import pytest
import torch
from autoannotate.config import (
    SUPPORTED_IMAGE_FORMATS,
    MODEL_CONFIGS,
    CLUSTERING_CONFIGS,
    DEFAULT_CONFIG,
    CACHE_DIR,
    MIN_IMAGES_PER_CLUSTER,
    MAX_REPRESENTATIVE_SAMPLES,
    DEFAULT_IMAGE_MAX_SIZE,
    get_model_config,
    get_clustering_config,
    validate_split_ratios,
    get_device,
)


class TestConstants:
    def test_supported_image_formats(self):
        assert isinstance(SUPPORTED_IMAGE_FORMATS, set)
        assert ".jpg" in SUPPORTED_IMAGE_FORMATS
        assert ".png" in SUPPORTED_IMAGE_FORMATS
        assert ".jpeg" in SUPPORTED_IMAGE_FORMATS

    def test_model_configs_structure(self):
        assert isinstance(MODEL_CONFIGS, dict)
        assert "clip" in MODEL_CONFIGS
        assert "dinov2" in MODEL_CONFIGS
        assert "dinov2-large" in MODEL_CONFIGS

        for model_name, config in MODEL_CONFIGS.items():
            assert "name" in config
            assert "embedding_dim" in config
            assert "batch_size" in config
            assert "max_image_size" in config
            assert isinstance(config["embedding_dim"], int)
            assert isinstance(config["batch_size"], int)

    def test_clustering_configs_structure(self):
        assert isinstance(CLUSTERING_CONFIGS, dict)
        assert "kmeans" in CLUSTERING_CONFIGS
        assert "hdbscan" in CLUSTERING_CONFIGS
        assert "spectral" in CLUSTERING_CONFIGS
        assert "dbscan" in CLUSTERING_CONFIGS

        for method, config in CLUSTERING_CONFIGS.items():
            assert "requires_n_clusters" in config
            assert "handles_noise" in config
            assert "available" in config
            assert "default_params" in config
            assert isinstance(config["requires_n_clusters"], bool)
            assert isinstance(config["handles_noise"], bool)

    def test_default_config_structure(self):
        assert isinstance(DEFAULT_CONFIG, dict)
        assert "embedding" in DEFAULT_CONFIG
        assert "clustering" in DEFAULT_CONFIG
        assert "organization" in DEFAULT_CONFIG
        assert "interactive" in DEFAULT_CONFIG
        assert "export" in DEFAULT_CONFIG

        # Check embedding config
        assert "model" in DEFAULT_CONFIG["embedding"]
        assert "batch_size" in DEFAULT_CONFIG["embedding"]
        assert "device" in DEFAULT_CONFIG["embedding"]

        # Check clustering config
        assert "method" in DEFAULT_CONFIG["clustering"]
        assert "reduce_dims" in DEFAULT_CONFIG["clustering"]

    def test_cache_dir_exists(self):
        assert CACHE_DIR.exists()
        assert CACHE_DIR.is_dir()

    def test_min_images_per_cluster(self):
        assert isinstance(MIN_IMAGES_PER_CLUSTER, int)
        assert MIN_IMAGES_PER_CLUSTER > 0

    def test_max_representative_samples(self):
        assert isinstance(MAX_REPRESENTATIVE_SAMPLES, int)
        assert MAX_REPRESENTATIVE_SAMPLES > 0

    def test_default_image_max_size(self):
        assert isinstance(DEFAULT_IMAGE_MAX_SIZE, tuple)
        assert len(DEFAULT_IMAGE_MAX_SIZE) == 2
        assert all(isinstance(x, int) for x in DEFAULT_IMAGE_MAX_SIZE)


class TestGetModelConfig:
    def test_get_clip_config(self):
        config = get_model_config("clip")
        assert config["name"] == "openai/clip-vit-large-patch14"
        assert config["embedding_dim"] == 768

    def test_get_dinov2_config(self):
        config = get_model_config("dinov2")
        assert config["name"] == "facebook/dinov2-base"
        assert config["embedding_dim"] == 768

    def test_get_dinov2_large_config(self):
        config = get_model_config("dinov2-large")
        assert config["name"] == "facebook/dinov2-large"
        assert config["embedding_dim"] == 1024

    def test_get_invalid_model_config(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_config("invalid_model")

    def test_config_is_copy(self):
        config1 = get_model_config("clip")
        config2 = get_model_config("clip")

        # Modify one config
        config1["test_key"] = "test_value"

        # Should not affect the other copy
        assert "test_key" not in config2


class TestGetClusteringConfig:
    def test_get_kmeans_config(self):
        config = get_clustering_config("kmeans")
        assert config["requires_n_clusters"] is True
        assert config["handles_noise"] is False
        assert "n_init" in config["default_params"]

    def test_get_hdbscan_config(self):
        config = get_clustering_config("hdbscan")
        assert config["requires_n_clusters"] is False
        assert config["handles_noise"] is True
        assert "min_cluster_size" in config["default_params"]

    def test_get_spectral_config(self):
        config = get_clustering_config("spectral")
        assert config["requires_n_clusters"] is True
        assert config["handles_noise"] is False

    def test_get_dbscan_config(self):
        config = get_clustering_config("dbscan")
        assert config["requires_n_clusters"] is False
        assert config["handles_noise"] is True

    def test_get_invalid_clustering_config(self):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            get_clustering_config("invalid_method")

    def test_config_is_copy(self):
        config1 = get_clustering_config("kmeans")
        config2 = get_clustering_config("kmeans")

        # Modify one config
        config1["test_key"] = "test_value"

        # Should not affect the other copy
        assert "test_key" not in config2


class TestValidateSplitRatios:
    def test_valid_ratios(self):
        assert validate_split_ratios(0.7, 0.15, 0.15) is True

    def test_valid_ratios_sum_to_1(self):
        assert validate_split_ratios(0.6, 0.2, 0.2) is True

    def test_ratios_sum_not_1(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_split_ratios(0.5, 0.3, 0.1)

    def test_negative_ratio(self):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_split_ratios(-0.1, 0.6, 0.5)

    def test_zero_ratio(self):
        # Zero is valid
        assert validate_split_ratios(0.8, 0.2, 0.0) is True

    def test_all_equal_ratios(self):
        ratio = 1.0 / 3.0
        assert validate_split_ratios(ratio, ratio, ratio) is True


class TestGetDevice:
    def test_get_device_returns_string(self):
        device = get_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]

    def test_get_device_cpu_available(self):
        device = get_device()
        # CPU should always be available
        assert device is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_cuda(self):
        device = get_device()
        assert device == "cuda"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_get_device_mps(self):
        device = get_device()
        assert device == "mps"
