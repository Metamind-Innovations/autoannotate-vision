from pathlib import Path
from typing import Optional, Union, List
import numpy as np
from PIL import Image

from autoannotate.config import (
    SUPPORTED_IMAGE_FORMATS,
    MODEL_CONFIGS,
    CLUSTERING_CONFIGS,
    MIN_IMAGES_PER_CLUSTER
)


class ValidationError(Exception):
    pass


def validate_directory(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_empty: bool = False
) -> Path:
    path = Path(path)
    
    if must_exist and not path.exists():
        raise ValidationError(f"Directory does not exist: {path}")
    
    if must_exist and not path.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")
    
    if must_be_empty and path.exists():
        if any(path.iterdir()):
            raise ValidationError(f"Directory is not empty: {path}")
    
    return path


def validate_image_file(image_path: Union[str, Path]) -> bool:
    path = Path(image_path)
    
    if not path.exists():
        return False
    
    if not path.is_file():
        return False
    
    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        return False
    
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_model_name(model_name: str) -> str:
    if model_name not in MODEL_CONFIGS:
        raise ValidationError(
            f"Invalid model name: {model_name}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    return model_name


def validate_clustering_method(method: str) -> str:
    if method not in CLUSTERING_CONFIGS:
        raise ValidationError(
            f"Invalid clustering method: {method}. "
            f"Available methods: {list(CLUSTERING_CONFIGS.keys())}"
        )
    return method


def validate_n_clusters(
    n_clusters: Optional[int],
    method: str,
    n_samples: int
) -> Optional[int]:
    config = CLUSTERING_CONFIGS[method]
    
    if config['requires_n_clusters'] and n_clusters is None:
        raise ValidationError(
            f"Clustering method '{method}' requires n_clusters to be specified"
        )
    
    if n_clusters is not None:
        if n_clusters < 2:
            raise ValidationError(
                f"n_clusters must be at least 2, got {n_clusters}"
            )
        
        if n_clusters > n_samples:
            raise ValidationError(
                f"n_clusters ({n_clusters}) cannot exceed number of samples ({n_samples})"
            )
        
        if n_clusters > n_samples // 2:
            import warnings
            warnings.warn(
                f"n_clusters ({n_clusters}) is more than half the number of samples ({n_samples}). "
                "This may result in very small clusters."
            )
    
    return n_clusters


def validate_batch_size(batch_size: int, n_samples: int) -> int:
    if batch_size < 1:
        raise ValidationError(
            f"batch_size must be at least 1, got {batch_size}"
        )
    
    if batch_size > n_samples:
        import warnings
        warnings.warn(
            f"batch_size ({batch_size}) is larger than number of samples ({n_samples}). "
            f"Setting batch_size to {n_samples}."
        )
        batch_size = n_samples
    
    return batch_size


def validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if not isinstance(embeddings, np.ndarray):
        raise ValidationError(
            f"Embeddings must be numpy array, got {type(embeddings)}"
        )
    
    if embeddings.ndim != 2:
        raise ValidationError(
            f"Embeddings must be 2D array (n_samples, n_features), got shape {embeddings.shape}"
        )
    
    if embeddings.shape[0] < 2:
        raise ValidationError(
            f"Need at least 2 samples for clustering, got {embeddings.shape[0]}"
        )
    
    if not np.isfinite(embeddings).all():
        raise ValidationError("Embeddings contain non-finite values (NaN or Inf)")
    
    return embeddings


def validate_labels(
    labels: np.ndarray,
    n_samples: int
) -> np.ndarray:
    if not isinstance(labels, np.ndarray):
        raise ValidationError(
            f"Labels must be numpy array, got {type(labels)}"
        )
    
    if labels.ndim != 1:
        raise ValidationError(
            f"Labels must be 1D array, got shape {labels.shape}"
        )
    
    if len(labels) != n_samples:
        raise ValidationError(
            f"Number of labels ({len(labels)}) must match number of samples ({n_samples})"
        )
    
    return labels


def validate_class_names(
    class_names: dict,
    labels: np.ndarray
) -> dict:
    if not isinstance(class_names, dict):
        raise ValidationError(
            f"class_names must be a dictionary, got {type(class_names)}"
        )
    
    if not class_names:
        raise ValidationError("class_names dictionary is empty")
    
    for label, name in class_names.items():
        if not isinstance(label, (int, np.integer)):
            raise ValidationError(
                f"Class label must be integer, got {type(label)} for label {label}"
            )
        
        if not isinstance(name, str):
            raise ValidationError(
                f"Class name must be string, got {type(name)} for class {label}"
            )
        
        if not name.strip():
            raise ValidationError(
                f"Class name cannot be empty for cluster {label}"
            )
        
        if label not in labels and label != -1:
            import warnings
            warnings.warn(
                f"Class label {label} ('{name}') not found in clustering results"
            )
    
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for label, name in class_names.items():
        if any(char in name for char in invalid_chars):
            raise ValidationError(
                f"Class name '{name}' contains invalid characters. "
                f"Avoid: {invalid_chars}"
            )
    
    return class_names


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> tuple:
    total = train_ratio + val_ratio + test_ratio
    
    if not (0.99 <= total <= 1.01):
        raise ValidationError(
            f"Split ratios must sum to 1.0, got {total:.4f}. "
            f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )
    
    if any(ratio < 0 for ratio in [train_ratio, val_ratio, test_ratio]):
        raise ValidationError("All split ratios must be non-negative")
    
    if any(ratio > 1 for ratio in [train_ratio, val_ratio, test_ratio]):
        raise ValidationError("All split ratios must be <= 1.0")
    
    return (train_ratio, val_ratio, test_ratio)


def validate_representative_samples(
    n_samples: int,
    cluster_size: int
) -> int:
    if n_samples < 1:
        raise ValidationError(
            f"n_samples must be at least 1, got {n_samples}"
        )
    
    if n_samples > cluster_size:
        import warnings
        warnings.warn(
            f"n_samples ({n_samples}) exceeds cluster size ({cluster_size}). "
            f"Setting n_samples to {cluster_size}."
        )
        n_samples = cluster_size
    
    return n_samples


def validate_export_format(format: str) -> str:
    valid_formats = ['csv', 'json']
    
    if format.lower() not in valid_formats:
        raise ValidationError(
            f"Invalid export format: {format}. "
            f"Valid formats: {valid_formats}"
        )
    
    return format.lower()


def validate_images_list(
    images: List,
    min_images: int = 2
) -> List:
    if not images:
        raise ValidationError("Images list is empty")
    
    if len(images) < min_images:
        raise ValidationError(
            f"Need at least {min_images} images, got {len(images)}"
        )
    
    for i, img in enumerate(images):
        if not isinstance(img, Image.Image):
            raise ValidationError(
                f"Image at index {i} is not a PIL Image object, got {type(img)}"
            )
    
    return images


def validate_device(device: str) -> str:
    valid_devices = ['cpu', 'cuda', 'mps', 'auto']
    
    if device not in valid_devices:
        raise ValidationError(
            f"Invalid device: {device}. Valid devices: {valid_devices}"
        )
    
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise ValidationError("CUDA is not available on this system")
    
    if device == 'mps':
        import torch
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise ValidationError("MPS is not available on this system")
    
    return device