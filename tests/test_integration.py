import pytest
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import shutil

from autoannotate import AutoAnnotator
from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.utils.image_loader import ImageLoader

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


@pytest.fixture
def temp_image_dir():
    temp_dir = tempfile.mkdtemp()

    for i in range(20):
        img = Image.new("RGB", (224, 224), color=(i * 10, i * 5, i * 3))
        img.save(Path(temp_dir) / f"image_{i}.jpg")

    yield Path(temp_dir)

    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_image_loader(temp_image_dir):
    loader = ImageLoader(temp_image_dir, recursive=False)
    image_paths = loader.load_image_paths()

    assert len(image_paths) == 20
    assert all(p.suffix == ".jpg" for p in image_paths)

    images, paths = loader.load_images()
    assert len(images) == 20
    assert all(isinstance(img, Image.Image) for img in images)


def test_embedding_extractor(temp_image_dir):
    loader = ImageLoader(temp_image_dir)
    images, _ = loader.load_images()

    extractor = EmbeddingExtractor(model_name="dinov2", batch_size=4)
    embeddings = extractor.extract_batch(images[:5])

    assert embeddings.shape[0] == 5
    assert embeddings.shape[1] > 0
    assert np.isfinite(embeddings).all()


def test_clustering_kmeans(temp_image_dir):
    loader = ImageLoader(temp_image_dir)
    images, _ = loader.load_images()

    extractor = EmbeddingExtractor(model_name="dinov2", batch_size=8)
    embeddings = extractor.extract_batch(images)

    clusterer = ClusteringEngine(method="kmeans", n_clusters=3, reduce_dims=False)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == len(images)
    assert len(np.unique(labels)) <= 3

    stats = clusterer.get_cluster_stats(labels)
    assert stats["n_clusters"] <= 3
    assert stats["total_samples"] == len(images)


@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="HDBSCAN not installed")
def test_clustering_hdbscan(temp_image_dir):
    loader = ImageLoader(temp_image_dir)
    images, _ = loader.load_images()

    extractor = EmbeddingExtractor(model_name="dinov2", batch_size=8)
    embeddings = extractor.extract_batch(images)

    clusterer = ClusteringEngine(method="hdbscan", reduce_dims=True)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == len(images)

    stats = clusterer.get_cluster_stats(labels)
    assert stats["total_samples"] == len(images)


def test_representative_indices(temp_image_dir):
    loader = ImageLoader(temp_image_dir)
    images, _ = loader.load_images()

    extractor = EmbeddingExtractor(model_name="dinov2", batch_size=8)
    embeddings = extractor.extract_batch(images)

    clusterer = ClusteringEngine(method="kmeans", n_clusters=3, reduce_dims=False)
    labels = clusterer.fit_predict(embeddings)

    representatives = clusterer.get_representative_indices(embeddings, labels, n_samples=3)

    assert isinstance(representatives, dict)
    for cluster_id, indices in representatives.items():
        assert len(indices) <= 3
        assert all(0 <= idx < len(images) for idx in indices)


def test_full_pipeline(temp_image_dir, temp_output_dir):
    annotator = AutoAnnotator(
        input_dir=temp_image_dir,
        output_dir=temp_output_dir,
        model="dinov2",
        clustering_method="kmeans",
        n_clusters=3,
        batch_size=8,
        reduce_dims=False,
        lazy_loading=False,  # Disable lazy loading for test to load images into memory
    )

    images, paths = annotator.load_images()
    assert len(images) == 20

    embeddings = annotator.extract_embeddings()
    assert embeddings.shape[0] == 20

    labels = annotator.cluster()
    assert len(labels) == 20

    stats = annotator.get_cluster_stats()
    assert stats["n_clusters"] <= 3


def test_full_pipeline_with_lazy_loading(temp_image_dir, temp_output_dir):
    annotator = AutoAnnotator(
        input_dir=temp_image_dir,
        output_dir=temp_output_dir,
        model="dinov2",
        clustering_method="kmeans",
        n_clusters=3,
        batch_size=8,
        reduce_dims=False,
        lazy_loading=True,  # Enable lazy loading
    )

    images, paths = annotator.load_images()
    # With lazy loading, images should be empty
    assert len(images) == 0
    # But paths should contain all image files
    assert len(paths) == 20

    # Embeddings should still be extracted correctly from paths
    embeddings = annotator.extract_embeddings()
    assert embeddings.shape[0] == 20

    labels = annotator.cluster()
    assert len(labels) == 20

    stats = annotator.get_cluster_stats()
    assert stats["n_clusters"] <= 3


def test_dimension_reduction():
    embeddings = np.random.randn(50, 768)

    clusterer = ClusteringEngine(method="kmeans", n_clusters=5, reduce_dims=True, target_dims=50)

    labels = clusterer.fit_predict(embeddings)
    assert len(labels) == 50


def test_invalid_clustering_method():
    embeddings = np.random.randn(20, 50)

    clusterer = ClusteringEngine(method="invalid_method", n_clusters=3)
    with pytest.raises(ValueError, match="Unknown clustering method"):
        clusterer.fit_predict(embeddings)


def test_missing_n_clusters():
    embeddings = np.random.randn(20, 128)

    with pytest.raises(ValueError):
        clusterer = ClusteringEngine(method="kmeans", n_clusters=None)
        clusterer.fit_predict(embeddings)


def test_image_loader_empty_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            loader = ImageLoader(temp_dir)
            loader.load_image_paths()
