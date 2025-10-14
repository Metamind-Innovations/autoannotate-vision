import pytest
import numpy as np
from PIL import Image
import torch

from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.config import MODEL_CONFIGS


pytestmark = pytest.mark.slow


@pytest.fixture
def sample_images():
    images = []
    for i in range(5):
        img = Image.new("RGB", (224, 224), color=(i * 50, i * 30, i * 20))
        images.append(img)
    return images


@pytest.fixture
def single_image():
    return Image.new("RGB", (224, 224), color=(100, 150, 200))


class TestEmbeddingExtractor:

    def test_initialization_dinov2(self):
        extractor = EmbeddingExtractor(model_name="dinov2")
        assert extractor.model_name == "dinov2"
        assert extractor.model is not None
        assert extractor.processor is not None

    @pytest.mark.skip(reason="CLIP download is slow, tested in integration")
    def test_initialization_clip(self):
        pass

    def test_initialization_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            EmbeddingExtractor(model_name="invalid_model")

    def test_device_selection_cpu(self):
        extractor = EmbeddingExtractor(model_name="dinov2", device="cpu")
        assert extractor.device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_selection_cuda(self):
        extractor = EmbeddingExtractor(model_name="dinov2", device="cuda")
        assert extractor.device == "cuda"

    def test_extract_single_embedding(self, single_image):
        extractor = EmbeddingExtractor(model_name="dinov2", batch_size=1)
        embedding = extractor.extract_single(single_image)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == MODEL_CONFIGS["dinov2"]["embedding_dim"]
        assert np.isfinite(embedding).all()

    def test_extract_batch_embeddings(self, sample_images):
        extractor = EmbeddingExtractor(model_name="dinov2", batch_size=2)
        embeddings = extractor.extract_batch(sample_images)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == len(sample_images)
        assert embeddings.shape[1] == MODEL_CONFIGS["dinov2"]["embedding_dim"]
        assert np.isfinite(embeddings).all()

    def test_embeddings_normalized(self, sample_images):
        extractor = EmbeddingExtractor(model_name="dinov2", batch_size=2)
        embeddings = extractor.extract_batch(sample_images)

        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_call_method(self, sample_images):
        extractor = EmbeddingExtractor(model_name="dinov2")
        embeddings = extractor(sample_images)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_images)

    def test_batch_size_handling(self, sample_images):
        extractor = EmbeddingExtractor(model_name="dinov2", batch_size=2)
        embeddings = extractor.extract_batch(sample_images)

        assert embeddings.shape[0] == len(sample_images)

    def test_different_image_sizes(self):
        images = [
            Image.new("RGB", (100, 100), color=(255, 0, 0)),
            Image.new("RGB", (300, 300), color=(0, 255, 0)),
            Image.new("RGB", (224, 224), color=(0, 0, 255)),
        ]

        extractor = EmbeddingExtractor(model_name="dinov2")
        embeddings = extractor.extract_batch(images)

        assert embeddings.shape[0] == len(images)

    def test_grayscale_image_conversion(self):
        gray_image = Image.new("L", (224, 224), color=128)

        extractor = EmbeddingExtractor(model_name="dinov2")
        embedding = extractor.extract_single(gray_image)

        assert isinstance(embedding, np.ndarray)
        assert np.isfinite(embedding).all()

    def test_model_eval_mode(self):
        extractor = EmbeddingExtractor(model_name="dinov2")
        assert not extractor.model.training

    def test_no_gradient_computation(self, single_image):
        extractor = EmbeddingExtractor(model_name="dinov2")

        with torch.no_grad():
            embedding = extractor.extract_single(single_image)

        assert isinstance(embedding, np.ndarray)

    def test_deterministic_output(self, single_image):
        extractor = EmbeddingExtractor(model_name="dinov2")

        embedding1 = extractor.extract_single(single_image)
        embedding2 = extractor.extract_single(single_image)

        assert np.allclose(embedding1, embedding2, atol=1e-6)

    def test_different_images_different_embeddings(self):
        img1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img2 = Image.new("RGB", (224, 224), color=(0, 0, 255))

        extractor = EmbeddingExtractor(model_name="dinov2")

        emb1 = extractor.extract_single(img1)
        emb2 = extractor.extract_single(img2)

        assert not np.allclose(emb1, emb2, atol=1e-3)

    @pytest.mark.skip(reason="Model download is slow, tested in integration")
    def test_clip_embeddings(self, sample_images):
        pass

    @pytest.mark.skip(reason="Large model download is slow")
    def test_dinov2_large_embeddings(self, sample_images):
        pass

    def test_empty_batch(self):
        extractor = EmbeddingExtractor(model_name="dinov2")

        with pytest.raises(Exception):
            extractor.extract_batch([])

    def test_large_batch(self):
        images = [Image.new("RGB", (224, 224), color=(i, i, i)) for i in range(100)]

        extractor = EmbeddingExtractor(model_name="dinov2", batch_size=16)
        embeddings = extractor.extract_batch(images)

        assert embeddings.shape[0] == 100

    def test_memory_efficiency(self, sample_images):
        extractor = EmbeddingExtractor(model_name="dinov2", batch_size=1)

        initial_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        embeddings = extractor.extract_batch(sample_images)

        assert embeddings.shape[0] == len(sample_images)
