# Tutorials

Step-by-step tutorials for using AutoAnnotate-Vision.

## Tutorial 1: Quick Start with CLI

### Prerequisites

```bash
pip install autoannotate-vision
```

### Step 1: Prepare Your Images

Organize your unlabeled images in a folder:

```
my_images/
├── img_001.jpg
├── img_002.jpg
├── img_003.jpg
└── ...
```

### Step 2: Run Auto-Annotation

```bash
autoannotate annotate \
    ./my_images \
    ./annotated_output \
    --n-clusters 5 \
    --method kmeans \
    --model dinov2 \
    --create-splits \
    --export-format csv
```

### Step 3: Interactive Labeling

The tool will show you representative samples from each cluster. For each cluster:

1. View the displayed sample images
2. Enter a class name (e.g., "cat", "dog", "car")
3. Or type "skip" to skip the cluster

### Step 4: Review Output

Your organized dataset will be in `./annotated_output`:

```
annotated_output/
├── metadata.json
├── labels.csv
├── cat/
│   ├── cat_1.jpg
│   └── cat_2.jpg
├── dog/
│   ├── dog_1.jpg
│   └── dog_2.jpg
└── splits/
    ├── train/
    ├── val/
    └── test/
```

---

## Tutorial 2: Python API - Full Pipeline

### Basic Usage

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./organized",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=10,
    batch_size=32
)

result = annotator.run_full_pipeline(
    n_samples=5,
    copy_files=True,
    create_splits=True,
    export_format="csv"
)

print(f"Processed {result['n_images']} images")
print(f"Created {result['n_clusters']} classes")
```

---

## Tutorial 3: Step-by-Step Custom Workflow

### Load and Explore Images

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./data/images",
    output_dir="./data/organized",
    model="dinov2-large",
    clustering_method="hdbscan"
)

images, image_paths = annotator.load_images()
print(f"Loaded {len(images)} images")
```

### Extract Embeddings

```python
embeddings = annotator.extract_embeddings()
print(f"Embeddings shape: {embeddings.shape}")
```

### Cluster Images

```python
labels = annotator.cluster()
stats = annotator.get_cluster_stats()

print(f"Number of clusters: {stats['n_clusters']}")
print(f"Unclustered images: {stats['n_noise']}")
print(f"Cluster sizes: {stats['cluster_sizes']}")
```

### Interactive Labeling

```python
class_names = annotator.interactive_labeling(n_samples=7)
print(f"Labeled {len(class_names)} classes")
```

### Organize and Export

```python
metadata = annotator.organize_dataset(copy_files=True)

annotator.create_splits(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

labels_file = annotator.export_labels(format="json")
print(f"Labels exported to: {labels_file}")
```

---

## Tutorial 4: Programmatic Labeling (No CLI)

### Automatic Labeling Without User Input

```python
from autoannotate import AutoAnnotator
import numpy as np

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=5
)

images, paths = annotator.load_images()
embeddings = annotator.extract_embeddings()
labels = annotator.cluster()

class_names = {
    0: "class_a",
    1: "class_b",
    2: "class_c",
    3: "class_d",
    4: "class_e"
}

annotator.organize_dataset(class_names=class_names)
annotator.export_labels(format="csv")
```

---

## Tutorial 5: Using Individual Components

### Embedding Extraction Only

```python
from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.utils.image_loader import ImageLoader

loader = ImageLoader("./images", recursive=True)
images, paths = loader.load_images()

extractor = EmbeddingExtractor(
    model_name="clip",
    batch_size=16
)

embeddings = extractor(images)

import numpy as np
np.save("embeddings.npy", embeddings)
```

### Clustering with Pre-computed Embeddings

```python
from autoannotate.core.clustering import ClusteringEngine
import numpy as np

embeddings = np.load("embeddings.npy")

clusterer = ClusteringEngine(
    method="hdbscan",
    reduce_dims=True,
    target_dims=50
)

labels = clusterer.fit_predict(embeddings)
stats = clusterer.get_cluster_stats(labels)

print(f"Found {stats['n_clusters']} clusters")
```

### Manual Organization

```python
from autoannotate.core.organizer import DatasetOrganizer
from pathlib import Path

image_paths = list(Path("./images").glob("*.jpg"))
labels = np.load("labels.npy")

class_names = {
    0: "category_1",
    1: "category_2",
    2: "category_3"
}

organizer = DatasetOrganizer(Path("./organized"))
metadata = organizer.organize_by_clusters(
    image_paths,
    labels,
    class_names,
    copy_files=True
)

organizer.create_split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

---

## Tutorial 6: Comparing Clustering Methods

### Experiment with Different Methods

```python
from autoannotate import AutoAnnotator
import pandas as pd

input_dir = "./images"
methods = ["kmeans", "hdbscan", "spectral"]

results = []

for method in methods:
    config = {"n_clusters": 10} if method in ["kmeans", "spectral"] else {}
    
    annotator = AutoAnnotator(
        input_dir=input_dir,
        output_dir=f"./output_{method}",
        clustering_method=method,
        **config
    )
    
    annotator.load_images()
    annotator.extract_embeddings()
    labels = annotator.cluster()
    stats = annotator.get_cluster_stats()
    
    results.append({
        "method": method,
        "n_clusters": stats['n_clusters'],
        "n_noise": stats['n_noise'],
        "cluster_sizes": stats['cluster_sizes']
    })

df = pd.DataFrame(results)
print(df)
```

---

## Tutorial 7: Working with Different Models

### Compare Vision Models

```python
from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.utils.image_loader import ImageLoader
import time

loader = ImageLoader("./images")
images, _ = loader.load_images()

models = ["clip", "dinov2", "dinov2-large"]

for model in models:
    print(f"\nTesting {model}...")
    
    extractor = EmbeddingExtractor(model_name=model, batch_size=16)
    
    start = time.time()
    embeddings = extractor(images)
    duration = time.time() - start
    
    print(f"  Shape: {embeddings.shape}")
    print(f"  Time: {duration:.2f}s")
    print(f"  Speed: {len(images)/duration:.2f} images/sec")
```

---

## Tutorial 8: Validation and Quality Control

### Validate Images Before Processing

```bash
autoannotate validate ./images --recursive
```

### Programmatic Validation

```python
from autoannotate.utils.image_loader import ImageLoader
from pathlib import Path

loader = ImageLoader("./images", recursive=True)
image_paths = loader.load_image_paths()

valid_images = []
invalid_images = []

for img_path in image_paths:
    if loader.validate_image(img_path):
        valid_images.append(img_path)
    else:
        invalid_images.append(img_path)

print(f"Valid: {len(valid_images)}")
print(f"Invalid: {len(invalid_images)}")

if invalid_images:
    print("\nInvalid files:")
    for img in invalid_images:
        print(f"  - {img}")
```

---

## Tutorial 9: Handling Large Datasets

### Memory-Efficient Processing

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./large_dataset",
    output_dir="./output",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=50,
    batch_size=8,
    reduce_dims=True
)

images, paths = annotator.load_images()
print(f"Processing {len(images)} images...")

embeddings = annotator.extract_embeddings()

labels = annotator.cluster()

representatives = annotator.get_representatives(n_samples=3)

class_names = annotator.interactive_labeling(n_samples=3)

annotator.organize_dataset(class_names=class_names, copy_files=False)
```

---

## Tutorial 10: Integration with ML Workflows

### Export for PyTorch DataLoader

```python
from autoannotate import AutoAnnotator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./dataset",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=10
)

result = annotator.run_full_pipeline(create_splits=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    root="./dataset/splits/train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

print(f"Classes: {train_dataset.classes}")
print(f"Samples: {len(train_dataset)}")
```

---

## Tutorial 11: Advanced Configuration

### Custom Clustering Parameters

```python
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.utils.image_loader import ImageLoader

loader = ImageLoader("./images")
images, paths = loader.load_images()

extractor = EmbeddingExtractor(model_name="dinov2-large", batch_size=16)
embeddings = extractor(images)

clusterer = ClusteringEngine(
    method="kmeans",
    n_clusters=20,
    reduce_dims=True,
    target_dims=100,
    random_state=42
)

labels = clusterer.fit_predict(embeddings)

representatives = clusterer.get_representative_indices(
    embeddings,
    labels,
    n_samples=10
)

print(f"Clusters: {len(representatives)}")
```

---

## Tutorial 12: Error Handling

### Robust Pipeline with Error Handling

```python
from autoannotate import AutoAnnotator
from autoannotate.utils.validators import ValidationError
import logging

logging.basicConfig(level=logging.INFO)

try:
    annotator = AutoAnnotator(
        input_dir="./images",
        output_dir="./output",
        model="dinov2",
        clustering_method="kmeans",
        n_clusters=10
    )
    
    result = annotator.run_full_pipeline()
    
    logging.info(f"Success! Processed {result['n_images']} images")
    
except ValidationError as e:
    logging.error(f"Validation error: {e}")
    
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    raise
```

---

## Best Practices

### 1. Choose the Right Model

- **CLIP**: Best for general images and when you need text-image alignment
- **DINOv2**: Balanced performance, good for most use cases
- **DINOv2-Large**: When quality is more important than speed

### 2. Select Clustering Method

- **K-means**: Fast, when you know the number of classes
- **HDBSCAN**: Automatic cluster detection, handles noise well
- **Spectral**: For complex cluster shapes
- **DBSCAN**: Density-based, good for irregular distributions

### 3. Optimize Performance

- Use appropriate `batch_size` for your GPU memory
- Enable `reduce_dims` for large embedding dimensions (>500)
- Use `recursive=False` unless you have nested folders
- Consider `create_symlinks=True` for large files

### 4. Quality Control

- Always validate images first
- Review representative samples carefully during labeling
- Check cluster sizes - very small clusters might indicate outliers
- Examine the `unclustered` folder for data quality issues

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory during embedding extraction
**Solution**: Reduce `batch_size` or use a smaller model

**Issue**: Too many small clusters
**Solution**: Reduce `n_clusters` or try HDBSCAN with higher `min_cluster_size`

**Issue**: Clusters don't make semantic sense
**Solution**: Try a different model or increase `n_samples` for better labeling

**Issue**: Slow processing
**Solution**: Ensure CUDA is available, increase `batch_size`, or use a faster model

---

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed documentation
- Check out [example scripts](https://github.com/yourusername/autoannotate-vision/tree/main/examples)
- Join discussions on [GitHub](https://github.com/yourusername/autoannotate-vision/discussions)