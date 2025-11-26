from pathlib import Path
from typing import Optional, Dict, Literal, List
import numpy as np
from PIL import Image

from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.core.organizer import DatasetOrganizer
from autoannotate.ui.interactive import InteractiveLabelingSession
from autoannotate.utils.image_loader import ImageLoader

__all__ = [
    "AutoAnnotator",
    "EmbeddingExtractor",
    "ClusteringEngine",
    "DatasetOrganizer",
    "InteractiveLabelingSession",
    "ImageLoader",
]


class AutoAnnotator:

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        model: Literal["clip", "dinov2", "dinov2-large", "siglip2"] = "dinov2",
        clustering_method: Literal["kmeans", "hdbscan", "spectral", "dbscan"] = "kmeans",
        n_clusters: Optional[int] = None,
        batch_size: int = 32,
        recursive: bool = False,
        reduce_dims: bool = True,
        lazy_loading: bool = True,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.recursive = recursive
        self.reduce_dims = reduce_dims
        self.lazy_loading = lazy_loading

        self.loader: Optional[ImageLoader] = None
        self.extractor: Optional[EmbeddingExtractor] = None
        self.clusterer: Optional[ClusteringEngine] = None
        self.organizer: Optional[DatasetOrganizer] = None

        self.images: Optional[List[Image.Image]] = None
        self.image_paths: Optional[List[Path]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.class_names: Optional[Dict[int, str]] = None

    def load_images(self):
        self.loader = ImageLoader(self.input_dir, recursive=self.recursive)
        self.images, self.image_paths = self.loader.load_images(lazy=self.lazy_loading)
        return self.images, self.image_paths

    def extract_embeddings(self):
        if self.image_paths is None:
            raise ValueError("Load images first using load_images()")

        self.extractor = EmbeddingExtractor(model_name=self.model, batch_size=self.batch_size)

        if self.lazy_loading:
            self.embeddings = self.extractor(self.image_paths)
        else:
            if self.images is None:
                raise ValueError("Images not loaded. Use lazy_loading=False with load_images()")
            self.embeddings = self.extractor(self.images)

        return self.embeddings

    def cluster(self):
        if self.embeddings is None:
            raise ValueError("Extract embeddings first using extract_embeddings()")

        self.clusterer = ClusteringEngine(
            method=self.clustering_method, n_clusters=self.n_clusters, reduce_dims=self.reduce_dims
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)
        return self.labels

    def get_cluster_stats(self) -> Dict:
        if self.labels is None:
            raise ValueError("Run clustering first using cluster()")
        if self.clusterer is None:
            raise ValueError("Run clustering first using cluster()")
        return self.clusterer.get_cluster_stats(self.labels)

    def get_representatives(self, n_samples: int = 5) -> Dict[int, np.ndarray]:
        if self.embeddings is None or self.labels is None:
            raise ValueError("Extract embeddings and cluster first")
        if self.clusterer is None:
            raise ValueError("Run clustering first using cluster()")

        return self.clusterer.get_representative_indices(
            self.embeddings, self.labels, n_samples=n_samples
        )

    def interactive_labeling(self, n_samples: int = 5):
        if self.labels is None:
            raise ValueError("Run clustering first")
        if self.image_paths is None:
            raise ValueError("Load images first using load_images()")

        representatives = self.get_representatives(n_samples=n_samples)
        stats = self.get_cluster_stats()

        session = InteractiveLabelingSession()
        session.display_cluster_stats(stats)

        self.class_names = session.label_all_clusters(
            self.image_paths, self.labels, representatives, stats, self.output_dir
        )

        session.display_labeling_summary(self.class_names, self.labels)
        return self.class_names

    def organize_dataset(
        self, class_names: Optional[Dict[int, str]] = None, copy_files: bool = True
    ):
        if class_names is None:
            if self.class_names is None:
                raise ValueError("Provide class_names or run interactive_labeling() first")
            class_names = self.class_names

        if self.image_paths is None or self.labels is None:
            raise ValueError("Load images and run clustering first")

        self.organizer = DatasetOrganizer(self.output_dir)
        metadata = self.organizer.organize_by_clusters(
            self.image_paths,
            self.labels,
            class_names,
            copy_files=copy_files,
            create_symlinks=not copy_files,
        )
        return metadata

    def create_splits(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
    ):
        if self.organizer is None:
            raise ValueError("Organize dataset first using organize_dataset()")

        return self.organizer.create_split(
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )

    def export_labels(self, format: str = "csv"):
        if self.organizer is None:
            raise ValueError("Organize dataset first using organize_dataset()")

        return self.organizer.export_labels_file(format=format)

    def run_full_pipeline(
        self,
        n_samples: int = 5,
        copy_files: bool = True,
        create_splits: bool = False,
        export_format: str = "csv",
    ):
        self.load_images()
        self.extract_embeddings()
        self.cluster()
        self.interactive_labeling(n_samples=n_samples)

        if self.class_names:
            self.organize_dataset(copy_files=copy_files)
            self.export_labels(format=export_format)

            if create_splits:
                self.create_splits()

        return {
            "n_images": len(self.images) if self.images else 0,
            "n_clusters": len(self.class_names) if self.class_names else 0,
            "output_dir": str(self.output_dir),
        }
