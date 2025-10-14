import pytest
import numpy as np
from sklearn.datasets import make_blobs

from autoannotate.core.clustering import ClusteringEngine


@pytest.fixture
def sample_embeddings():
    X, y = make_blobs(n_samples=100, n_features=128, centers=5, random_state=42)
    return X, y


@pytest.fixture
def high_dim_embeddings():
    X, y = make_blobs(n_samples=50, n_features=768, centers=3, random_state=42)
    return X, y


@pytest.fixture
def small_embeddings():
    X, y = make_blobs(n_samples=20, n_features=64, centers=3, random_state=42)
    return X, y


class TestClusteringEngine:
    
    def test_initialization_kmeans(self):
        clusterer = ClusteringEngine(method="kmeans", n_clusters=5)
        assert clusterer.method == "kmeans"
        assert clusterer.n_clusters == 5
    
    def test_initialization_hdbscan(self):
        clusterer = ClusteringEngine(method="hdbscan")
        assert clusterer.method == "hdbscan"
        assert clusterer.n_clusters is None
    
    def test_initialization_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            clusterer = ClusteringEngine(method="invalid")
            clusterer.fit_predict(np.random.randn(50, 128))
    
    def test_kmeans_clustering(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=5, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) <= 5
        assert labels.min() >= 0
    
    def test_hdbscan_clustering(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="hdbscan", reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
        assert -1 in labels or labels.min() >= 0
    
    def test_spectral_clustering(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="spectral", n_clusters=5, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) <= 5
    
    def test_dbscan_clustering(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="dbscan", reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
    
    def test_kmeans_missing_n_clusters(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        with pytest.raises(ValueError, match="n_clusters must be specified"):
            clusterer = ClusteringEngine(method="kmeans", n_clusters=None)
            clusterer.fit_predict(X)
    
    def test_dimensionality_reduction(self, high_dim_embeddings):
        X, y_true = high_dim_embeddings
        
        clusterer = ClusteringEngine(
            method="kmeans",
            n_clusters=3,
            reduce_dims=True,
            target_dims=50
        )
        
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
        assert clusterer.dim_reducer is not None
    
    def test_no_dimensionality_reduction(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(
            method="kmeans",
            n_clusters=5,
            reduce_dims=False
        )
        
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
        assert clusterer.dim_reducer is None
    
    def test_get_cluster_stats(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=5, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        stats = clusterer.get_cluster_stats(labels)
        
        assert 'n_clusters' in stats
        assert 'n_noise' in stats
        assert 'cluster_sizes' in stats
        assert 'total_samples' in stats
        assert stats['total_samples'] == len(X)
    
    def test_get_cluster_stats_with_noise(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="hdbscan", reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        stats = clusterer.get_cluster_stats(labels)
        
        assert stats['n_noise'] >= 0
        assert stats['n_clusters'] >= 0
    
    def test_get_representative_indices(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=5, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        representatives = clusterer.get_representative_indices(X, labels, n_samples=3)
        
        assert isinstance(representatives, dict)
        for cluster_id, indices in representatives.items():
            assert len(indices) <= 3
            assert all(0 <= idx < len(X) for idx in indices)
    
    def test_representative_indices_vary_n_samples(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=5, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        rep_3 = clusterer.get_representative_indices(X, labels, n_samples=3)
        rep_5 = clusterer.get_representative_indices(X, labels, n_samples=5)
        
        assert all(len(indices) <= 3 for indices in rep_3.values())
        assert all(len(indices) <= 5 for indices in rep_5.values())
    
    def test_representatives_are_closest_to_centroid(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=3, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        representatives = clusterer.get_representative_indices(X, labels, n_samples=1)
        
        for cluster_id, indices in representatives.items():
            mask = labels == cluster_id
            cluster_points = X[mask]
            centroid = cluster_points.mean(axis=0)
            
            rep_idx = indices[0]
            rep_point = X[rep_idx]
            
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            min_distance = distances.min()
            rep_distance = np.linalg.norm(rep_point - centroid)
            
            assert np.isclose(rep_distance, min_distance, atol=1e-5)
    
    def test_random_state_reproducibility(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer1 = ClusteringEngine(method="kmeans", n_clusters=5, random_state=42)
        labels1 = clusterer1.fit_predict(X)
        
        clusterer2 = ClusteringEngine(method="kmeans", n_clusters=5, random_state=42)
        labels2 = clusterer2.fit_predict(X)
        
        assert np.array_equal(labels1, labels2)
    
    def test_scaling_applied(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=5, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        assert clusterer.scaler is not None
        assert hasattr(clusterer.scaler, 'mean_')
        assert hasattr(clusterer.scaler, 'scale_')
    
    def test_eps_estimation_dbscan(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="dbscan", reduce_dims=False)
        
        embeddings_scaled = clusterer.scaler.fit_transform(X)
        eps = clusterer._estimate_eps(embeddings_scaled)
        
        assert eps > 0
        assert np.isfinite(eps)
    
    def test_small_dataset(self, small_embeddings):
        X, y_true = small_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=3, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) <= 3
    
    def test_single_cluster(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=1, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        assert len(np.unique(labels)) == 1
        assert all(labels == 0)
    
    def test_cluster_sizes_sum_to_total(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="kmeans", n_clusters=5, reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        stats = clusterer.get_cluster_stats(labels)
        total_in_clusters = sum(stats['cluster_sizes'].values())
        
        assert total_in_clusters + stats['n_noise'] == stats['total_samples']
    
    def test_pca_reducer_for_large_dims(self):
        X = np.random.randn(50, 500)
        
        clusterer = ClusteringEngine(
            method="kmeans",
            n_clusters=3,
            reduce_dims=True,
            target_dims=50
        )
        
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
        from sklearn.decomposition import PCA
        assert isinstance(clusterer.dim_reducer, PCA)
    
    def test_umap_reducer_for_moderate_dims(self):
        X = np.random.randn(50, 200)
        
        clusterer = ClusteringEngine(
            method="kmeans",
            n_clusters=3,
            reduce_dims=True,
            target_dims=50
        )
        
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == len(X)
        import umap
        assert isinstance(clusterer.dim_reducer, umap.UMAP)
    
    def test_target_dims_respected(self):
        X = np.random.randn(50, 500)
        target_dims = 30
        
        clusterer = ClusteringEngine(
            method="kmeans",
            n_clusters=3,
            reduce_dims=True,
            target_dims=target_dims
        )
        
        labels = clusterer.fit_predict(X)
        
        assert clusterer.dim_reducer.n_components == target_dims
    
    def test_hdbscan_min_cluster_size(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="hdbscan", reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        stats = clusterer.get_cluster_stats(labels)
        
        for cluster_id, size in stats['cluster_sizes'].items():
            assert size >= 5 or cluster_id == -1
    
    def test_no_representatives_for_noise(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        clusterer = ClusteringEngine(method="hdbscan", reduce_dims=False)
        labels = clusterer.fit_predict(X)
        
        representatives = clusterer.get_representative_indices(X, labels, n_samples=3)
        
        assert -1 not in representatives
    
    def test_different_n_clusters_values(self, sample_embeddings):
        X, y_true = sample_embeddings
        
        for n_clusters in [2, 5, 10]:
            clusterer = ClusteringEngine(
                method="kmeans",
                n_clusters=n_clusters,
                reduce_dims=False
            )
            labels = clusterer.fit_predict(X)
            
            assert len(np.unique(labels)) <= n_clusters