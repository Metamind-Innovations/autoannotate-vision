import numpy as np
from typing import Literal, Optional, Dict, Any
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None  # type: ignore[assignment]


class ClusteringEngine:

    def __init__(
        self,
        method: Literal["kmeans", "hdbscan", "spectral", "dbscan"] = "kmeans",
        n_clusters: Optional[int] = None,
        reduce_dims: bool = True,
        target_dims: int = 50,
        random_state: int = 42,
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.reduce_dims = reduce_dims
        self.target_dims = target_dims
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.dim_reducer = None
        self.clusterer = None

    def _reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.reduce_dims or embeddings.shape[1] <= self.target_dims:
            return embeddings

        if embeddings.shape[1] > 300:
            # PCA requires n_components < min(n_samples, n_features)
            n_components = min(self.target_dims, embeddings.shape[0] - 1, embeddings.shape[1] - 1)
            self.dim_reducer = PCA(n_components=n_components, random_state=self.random_state)
        else:
            n_components = min(self.target_dims, embeddings.shape[1] - 1)
            self.dim_reducer = umap.UMAP(  # type: ignore[union-attr]
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=min(15, embeddings.shape[0] - 1),
            )

        reduced: np.ndarray = self.dim_reducer.fit_transform(embeddings)  # type: ignore[union-attr, attr-defined]  # noqa: E501
        return reduced

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        # Validate n_clusters BEFORE any processing
        if self.method in ["kmeans", "spectral"] and self.n_clusters is None:
            raise ValueError(f"n_clusters must be specified for {self.method}")

        # Validate minimum images per cluster (6 images per cluster minimum)
        n_samples = embeddings.shape[0]
        if self.method in ["kmeans", "spectral"] and self.n_clusters is not None:
            min_required = 6 * self.n_clusters
            if n_samples < min_required:
                raise ValueError(
                    f"Insufficient samples for {self.n_clusters} clusters. "
                    f"Have {n_samples} images but need at least {min_required} "
                    f"(minimum 6 images per cluster)."
                )
        elif n_samples < 6:
            # For auto-clustering methods (hdbscan, dbscan), require minimum 6 images total
            raise ValueError(
                f"Insufficient samples for clustering. "
                f"Have {n_samples} images but need at least 6 images minimum."
            )

        embeddings_scaled = self.scaler.fit_transform(embeddings)
        embeddings_reduced = self._reduce_dimensionality(embeddings_scaled)

        labels: np.ndarray
        if self.method == "kmeans":
            self.clusterer = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
            )
            labels = self.clusterer.fit_predict(embeddings_reduced)  # type: ignore[assignment, attr-defined]  # noqa: E501

        elif self.method == "hdbscan":
            min_cluster_size = max(5, embeddings.shape[0] // 100)
            self.clusterer = hdbscan.HDBSCAN(  # type: ignore[union-attr]
                min_cluster_size=min_cluster_size,
                min_samples=5,
                cluster_selection_epsilon=0.0,
                metric="euclidean",
            )
            labels = self.clusterer.fit_predict(embeddings_reduced)  # type: ignore[assignment, union-attr, attr-defined]  # noqa: E501

        elif self.method == "spectral":
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for spectral")
            self.clusterer = SpectralClustering(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                affinity="nearest_neighbors",
            )
            labels = self.clusterer.fit_predict(embeddings_reduced)  # type: ignore[assignment, attr-defined]  # noqa: E501

        elif self.method == "dbscan":
            eps = self._estimate_eps(embeddings_reduced)
            self.clusterer = DBSCAN(eps=eps, min_samples=5)
            labels = self.clusterer.fit_predict(embeddings_reduced)  # type: ignore[assignment, attr-defined]  # noqa: E501

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        return labels

    def _estimate_eps(self, embeddings: np.ndarray) -> float:
        from sklearn.neighbors import NearestNeighbors

        k = min(5, embeddings.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        distances = np.sort(distances[:, -1])

        knee_point = int(0.95 * len(distances))
        eps_value: float = float(distances[knee_point])
        return eps_value

    def get_cluster_stats(self, labels: np.ndarray) -> Dict[str, Any]:
        unique_labels = np.unique(labels)
        n_noise: int = int(np.sum(labels == -1))

        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(labels == label))

        return {
            "n_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "n_noise": n_noise,
            "cluster_sizes": cluster_sizes,
            "total_samples": len(labels),
        }

    def get_representative_indices(
        self, embeddings: np.ndarray, labels: np.ndarray, n_samples: int = 5
    ) -> Dict[int, np.ndarray]:
        representatives = {}
        embeddings_scaled = self.scaler.transform(embeddings)

        if self.dim_reducer is not None:
            embeddings_reduced = self.dim_reducer.transform(embeddings_scaled)
        else:
            embeddings_reduced = embeddings_scaled

        for label in np.unique(labels):
            if label == -1:
                continue

            mask = labels == label
            cluster_embeddings = embeddings_reduced[mask]
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            centroid = cluster_embeddings.mean(axis=0)

            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            closest_indices = np.argsort(distances)[: min(n_samples, len(indices))]
            representatives[int(label)] = indices[closest_indices]

        return representatives
