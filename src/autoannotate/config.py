from typing import Dict
from pathlib import Path
import os


SUPPORTED_IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".webp",
    ".svg",
}


MODEL_CONFIGS = {
    "clip": {
        "name": "openai/clip-vit-large-patch14",
        "embedding_dim": 768,
        "batch_size": 32,
        "max_image_size": (224, 224),
    },
    "dinov2": {
        "name": "facebook/dinov2-base",
        "embedding_dim": 768,
        "batch_size": 32,
        "max_image_size": (224, 224),
    },
    "dinov2-large": {
        "name": "facebook/dinov2-large",
        "embedding_dim": 1024,
        "batch_size": 16,
        "max_image_size": (224, 224),
    },
}


CLUSTERING_CONFIGS = {
    "kmeans": {
        "requires_n_clusters": True,
        "handles_noise": False,
        "available": True,
        "default_params": {"n_init": 10, "max_iter": 300, "tol": 1e-4},
    },
    "hdbscan": {
        "requires_n_clusters": False,
        "handles_noise": True,
        "available": False,  # Will be checked at runtime
        "default_params": {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
        },
    },
    "spectral": {
        "requires_n_clusters": True,
        "handles_noise": False,
        "available": True,
        "default_params": {"affinity": "nearest_neighbors", "n_neighbors": 10},
    },
    "dbscan": {
        "requires_n_clusters": False,
        "handles_noise": True,
        "available": True,
        "default_params": {"min_samples": 5, "metric": "euclidean"},
    },
}


DEFAULT_CONFIG = {
    "embedding": {
        "model": "dinov2",
        "batch_size": 32,
        "device": "auto",
    },
    "clustering": {
        "method": "kmeans",
        "n_clusters": None,
        "reduce_dims": True,
        "target_dims": 50,
        "random_state": 42,
    },
    "organization": {
        "copy_files": True,
        "create_symlinks": False,
        "rename_pattern": "{class_name}_{index}",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
    "interactive": {"n_representative_samples": 5, "show_images": False, "auto_accept": False},
    "export": {"format": "csv", "include_metadata": True, "export_embeddings": False},
}


CACHE_DIR = Path.home() / ".cache" / "autoannotate"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("AUTOANNOTATE_LOG_LEVEL", "INFO")


MIN_IMAGES_PER_CLUSTER = 1
MAX_REPRESENTATIVE_SAMPLES = 20
DEFAULT_IMAGE_MAX_SIZE = (512, 512)


def get_model_config(model_name: str) -> Dict:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. " f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name].copy()


def get_clustering_config(method: str) -> Dict:
    if method not in CLUSTERING_CONFIGS:
        raise ValueError(
            f"Unknown clustering method: {method}. "
            f"Available methods: {list(CLUSTERING_CONFIGS.keys())}"
        )
    return CLUSTERING_CONFIGS[method].copy()


def validate_split_ratios(train: float, val: float, test: float) -> bool:
    total = train + val + test
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total}. "
            f"Ratios: train={train}, val={val}, test={test}"
        )

    if any(ratio < 0 or ratio > 1 for ratio in [train, val, test]):
        raise ValueError("All split ratios must be between 0 and 1")

    return True


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
