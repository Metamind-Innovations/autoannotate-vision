# API Reference

Complete API documentation for AutoAnnotate-Vision.

## Core Classes

### `AutoAnnotator`

Main class for the auto-annotation pipeline.

```python
class AutoAnnotator(
    input_dir: Path,
    output_dir: Path,
    model: Literal["clip", "dinov2", "dinov2-large"] = "dinov2",
    clustering_method: Literal["kmeans", "hdbscan", "spectral", "dbscan"] = "kmeans",
    n_clusters: Optional[int] = None,
    batch_size: int = 32,
    recursive: bool = False,
    reduce_dims: bool = True
)
```

**Parameters:**
- `input_dir`: Directory containing input images
- `output_dir`: Directory for organized output
- `model`: Vision model for embeddings
- `clustering_method`: Clustering algorithm
- `n_clusters`: Number of clusters (required for kmeans/spectral)
- `batch_size`: Batch size for embedding extraction
- `recursive`: Search for images recursively
- `reduce_dims`: Apply dimensionality reduction

**Methods:**

#### `load_images()`
Load images from input directory.

**Returns:** `Tuple[List[Image.Image], List[Path]]`

#### `extract_embeddings()`
Extract embeddings from loaded images.

**Returns:** `np.ndarray` - Shape (n_images, embedding_dim)

#### `cluster()`
Perform clustering on embeddings.

**Returns:** `np.ndarray` - Cluster labels

#### `get_cluster_stats()`
Get clustering statistics.

**Returns:** `Dict` with keys:
- `n_clusters`: Number of clusters
- `n_noise`: Number of unclustered samples
- `cluster_sizes`: Dictionary of cluster sizes
- `total_samples`: Total number of samples

#### `interactive_labeling(n_samples: int = 5)`
Start interactive labeling session.

**Parameters:**
- `n_samples`: Number of representative samples per cluster

**Returns:** `Dict[int, str]` - Mapping of cluster IDs to class names

#### `organize_dataset(class_names: Optional[Dict[int, str]] = None, copy_files: bool = True)`
Organize dataset into labeled folders.

**Parameters:**
- `class_names`: Cluster ID to class name mapping
- `copy_files`: Copy files (True) or create symlinks (False)

**Returns:** `Dict` - Organization metadata

#### `create_splits(train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15)`
Create train/validation/test splits.

**Parameters:**
- `train_ratio`: Training set ratio
- `val_ratio`: Validation set ratio
- `test_ratio`: Test set ratio

**Returns:** `Dict` - Split information

#### `export_labels(format: str = "csv")`
Export labels to file.

**Parameters:**
- `format`: Export format ("csv" or "json")

**Returns:** `Path` - Path to labels file

#### `run_full_pipeline(**kwargs)`
Run complete annotation pipeline.

**Returns:** `Dict` - Pipeline results

---

## Embedding Module

### `EmbeddingExtractor`

Extract embeddings from images using pre-trained vision models.

```python
class EmbeddingExtractor(
    model_name: Literal["clip", "dinov2", "dinov2-large"] = "dinov2",
    device: Optional[str] = None,
    batch_size: int = 32
)
```

**Parameters:**
- `model_name`: Name of the vision model
- `device`: Device for computation ("cpu", "cuda", "mps", or "auto")
- `batch_size`: Batch size for processing

**Methods:**

#### `extract_single(image: Image.Image)`
Extract embedding from a single image.

**Returns:** `np.ndarray` - 1D embedding vector

#### `extract_batch(images: List[Image.Image])`
Extract embeddings from a batch of images.

**Returns:** `np.ndarray` - 2D array (n_images, embedding_dim)

#### `__call__(images: List[Image.Image])`
Alias for `extract_batch`.

---

## Clustering Module

### `ClusteringEngine`

Perform clustering on image embeddings.

```python
class ClusteringEngine(
    method: Literal["kmeans", "hdbscan", "spectral", "dbscan"] = "kmeans",
    n_clusters: Optional[int] = None,
    reduce_dims: bool = True,
    target_dims: int = 50,
    random_state: int = 42
)
```

**Parameters:**
- `method`: Clustering algorithm
- `n_clusters`: Number of clusters (for kmeans/spectral)
- `reduce_dims`: Apply dimensionality reduction
- `target_dims`: Target dimensionality
- `random_state`: Random seed

**Methods:**

#### `fit_predict(embeddings: np.ndarray)`
Fit clustering model and predict labels.

**Returns:** `np.ndarray` - Cluster labels

#### `get_cluster_stats(labels: np.ndarray)`
Compute clustering statistics.

**Returns:** `Dict` - Statistics dictionary

#### `get_representative_indices(embeddings: np.ndarray, labels: np.ndarray, n_samples: int = 5)`
Get representative sample indices for each cluster.

**Returns:** `Dict[int, np.ndarray]` - Cluster ID to indices mapping

---

## Organization Module

### `DatasetOrganizer`

Organize annotated images into structured folders.

```python
class DatasetOrganizer(output_dir: Path)
```

**Parameters:**
- `output_dir`: Output directory path

**Methods:**

#### `organize_by_clusters(image_paths, labels, class_names, copy_files=True, create_symlinks=False)`
Organize images by cluster assignments.

**Returns:** `Dict` - Organization metadata

#### `create_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)`
Create train/val/test splits.

**Returns:** `Dict` - Split information

#### `export_labels_file(format="csv")`
Export labels to CSV or JSON.

**Returns:** `Path` - Path to labels file

---

## Utilities

### `ImageLoader`

Load and validate images from directories.

```python
class ImageLoader(input_dir: Path, recursive: bool = False)
```

**Methods:**

#### `load_image_paths()`
Get list of valid image paths.

**Returns:** `List[Path]`

#### `load_images(max_size: Optional[Tuple[int, int]] = None)`
Load images as PIL Image objects.

**Returns:** `Tuple[List[Image.Image], List[Path]]`

#### `validate_image(image_path: Path)` (static)
Validate a single image file.

**Returns:** `bool`

---

## Interactive UI

### `InteractiveLabelingSession`

Interactive CLI for labeling clusters.

```python
class InteractiveLabelingSession()
```

**Methods:**

#### `label_all_clusters(image_paths, labels, representative_indices, cluster_stats)`
Run interactive labeling for all clusters.

**Returns:** `Dict[int, str]` - Cluster ID to class name mapping

#### `display_cluster_stats(stats: Dict)`
Display clustering statistics in terminal.

#### `display_labeling_summary(class_names: Dict, labels: np.ndarray)`
Display summary of labeling results.

---

## Configuration

### Available Models

| Model | Embedding Dim | Best For |
|-------|---------------|----------|
| `clip` | 768 | General images, text-image tasks |
| `dinov2` | 768 | General images, balanced speed/quality |
| `dinov2-large` | 1024 | High-quality, complex datasets |

### Clustering Methods

| Method | Auto-clusters | Handles Noise | Requires n_clusters |
|--------|---------------|---------------|---------------------|
| `kmeans` | ❌ | ❌ | ✅ |
| `hdbscan` | ✅ | ✅ | ❌ |
| `spectral` | ❌ | ❌ | ✅ |
| `dbscan` | ✅ | ✅ | ❌ |

---

## CLI Commands

### `autoannotate annotate`

Main annotation command.

```bash
autoannotate annotate INPUT_DIR OUTPUT_DIR [OPTIONS]
```

**Options:**
- `-n, --n-clusters INTEGER`: Number of clusters
- `-m, --method`: Clustering method
- `--model`: Embedding model
- `-b, --batch-size`: Batch size
- `-r, --recursive`: Search recursively
- `--reduce-dims / --no-reduce-dims`: Dimensionality reduction
- `--n-samples`: Representative samples per cluster
- `--copy / --symlink`: File handling method
- `--create-splits`: Create train/val/test splits
- `--export-format`: Labels export format

### `autoannotate validate`

Validate image files.

```bash
autoannotate validate INPUT_DIR [OPTIONS]
```

**Options:**
- `-r, --recursive`: Search recursively

---

## Error Handling

### `ValidationError`

Raised when input validation fails.

```python
from autoannotate.utils.validators import ValidationError

try:
    annotator = AutoAnnotator(...)
except ValidationError as e:
    print(f"Validation error: {e}")
```

---

## Examples

### Basic Pipeline

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=10
)

result = annotator.run_full_pipeline()
```

### Custom Workflow

```python
from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.utils.image_loader import ImageLoader

loader = ImageLoader("./images")
images, paths = loader.load_images()

extractor = EmbeddingExtractor(model_name="clip")
embeddings = extractor(images)

clusterer = ClusteringEngine(method="hdbscan")
labels = clusterer.fit_predict(embeddings)

stats = clusterer.get_cluster_stats(labels)
print(f"Found {stats['n_clusters']} clusters")
```

---

## Type Hints

The package uses type hints throughout. Import types:

```python
from typing import List, Dict, Optional, Tuple, Literal
from pathlib import Path
import numpy as np
from PIL import Image
```