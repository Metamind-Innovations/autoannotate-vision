# Tutorials

Step-by-step tutorials for using AutoAnnotate-Vision.

## Tutorial 1: Quick Start with GUI

### The Easiest Way to Start!

1. **Launch the GUI:**
```bash
autoannotate-images
```

2. **Select Input Folder**
   - Click "Browse..." next to "Input Folder"
   - Navigate to your folder containing images
   - Click "Select Folder"

3. **Select Output Folder**
   - Click "Browse..." next to "Output Folder"
   - Choose where you want organized images
   - Click "Select Folder"

4. **Configure Settings**
   - Number of Classes: Set to how many categories you expect (e.g., 5)
   - Model: Choose "dinov2" (recommended)
   - Batch Size: Leave at 16 (or lower if memory issues)

5. **Click "Start Auto-Annotation"**

6. **Review HTML Previews**
   - Browser tabs will open showing sample images from each cluster
   - Review the images to understand what's in each cluster

7. **Name Clusters**
   - Return to terminal/command prompt
   - For each cluster, type a class name (e.g., "cats", "dogs")
   - Or type "skip" to skip a cluster

8. **Done!**
   - Images are organized into folders with **original filenames preserved**
   - Check the output folder for results

---

## Tutorial 2: CLI Quick Start

### Run from Command Line

```bash
autoannotate-images-cli annotate ./my_images ./output \
    --n-clusters 5 \
    --method kmeans \
    --model dinov2 \
    --create-splits
```

**What happens:**
1. Loads all images from `./my_images`
2. Extracts embeddings using DINOv2
3. Clusters into 5 groups
4. Opens HTML previews in browser
5. Asks you to name each cluster
6. Organizes images (original filenames kept!)
7. Creates train/val/test splits

---

## Tutorial 3: Python API - Full Pipeline

```python
from autoannotate import AutoAnnotator
from pathlib import Path

annotator = AutoAnnotator(
    input_dir=Path("./images"),
    output_dir=Path("./organized"),
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=5
)

result = annotator.run_full_pipeline(
    n_samples=5,
    create_splits=True
)

print(f"Processed {result['n_images']} images")
print(f"Created {result['n_clusters']} classes")
```

---

## Tutorial 4: Step-by-Step Control

For more control over each step:

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=5
)

images, paths = annotator.load_images()
print(f"Loaded {len(images)} images")

embeddings = annotator.extract_embeddings()
print(f"Embeddings: {embeddings.shape}")

labels = annotator.cluster()
stats = annotator.get_cluster_stats()
print(f"Found {stats['n_clusters']} clusters")

class_names = annotator.interactive_labeling(n_samples=6)

annotator.organize_dataset(copy_files=True)

annotator.create_splits()

annotator.export_labels(format="csv")
```

---

## Tutorial 5: Programmatic Labeling (No User Input)

Skip interactive labeling:

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=3
)

annotator.load_images()
annotator.extract_embeddings()
annotator.cluster()

class_names = {
    0: "cats",
    1: "dogs",
    2: "birds"
}

annotator.organize_dataset(class_names=class_names)
annotator.export_labels()
```

**Output structure:**
```
output/
├── cats/
│   ├── IMG_001.jpg  ✅ Original name!
│   └── my_cat.jpg   ✅ Original name!
├── dogs/
└── birds/
```

---

## Tutorial 6: Using Individual Components

### Extract Embeddings Only

```python
from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.utils.image_loader import ImageLoader

loader = ImageLoader("./images")
images, paths = loader.load_images()

extractor = EmbeddingExtractor(model_name="dinov2")
embeddings = extractor(images)

import numpy as np
np.save("embeddings.npy", embeddings)
```

### Cluster Pre-computed Embeddings

```python
from autoannotate.core.clustering import ClusteringEngine
import numpy as np

embeddings = np.load("embeddings.npy")

clusterer = ClusteringEngine(method="kmeans", n_clusters=5)
labels = clusterer.fit_predict(embeddings)

stats = clusterer.get_cluster_stats(labels)
print(f"Clusters: {stats}")
```

---

## Tutorial 7: HTML Preview Feature

The HTML preview feature automatically:
- Opens in your default browser
- Shows thumbnail grid of sample images
- Displays cluster statistics
- Uses responsive design
- Saves to output directory for later reference

You can also generate previews manually:

```python
from autoannotate.ui.html_preview import generate_cluster_preview_html
from pathlib import Path

html_path = generate_cluster_preview_html(
    cluster_id=0,
    image_paths=[Path("img1.jpg"), Path("img2.jpg")],
    indices=np.array([0, 1]),
    cluster_size=10
)

print(f"Preview saved to: {html_path}")
```

---

## Tutorial 8: Using Different Models

### CLIP for Text-Image Tasks

```python
annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="clip",  # Good for general images
    clustering_method="kmeans",
    n_clusters=5
)
```

### DINOv2-Large for High Quality

```python
annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="dinov2-large",  # Best quality, slower
    clustering_method="kmeans",
    n_clusters=5,
    batch_size=8  # Lower batch size for large model
)
```

### SigLIP2 for Best Results

```python
annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="siglip2",  # Latest Google model - Recommended
    clustering_method="kmeans",
    n_clusters=5
)
```

---

## Tutorial 9: Using HDBSCAN (Optional)

If you have HDBSCAN installed:

```bash
pip install autoannotate-vision[hdbscan]
```

Then:

```python
annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="dinov2",
    clustering_method="hdbscan",  # Auto-determines clusters!
    n_clusters=None  # Not needed for HDBSCAN
)

result = annotator.run_full_pipeline()
```

---

## Tutorial 10: Custom Train/Val/Test Splits

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    clustering_method="kmeans",
    n_clusters=10
)

annotator.run_full_pipeline()

annotator.create_splits(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

**Output:**
```
output/splits/
├── train/
│   ├── class1/
│   └── class2/
├── val/
└── test/
```

---

## Best Practices

### 1. Choose the Right Model

- **CLIP**: General images, faster
- **DINOv2**: Good for objects, balanced performance
- **DINOv2-Large**: When quality is critical
- **SigLIP2**: Latest Google model - **Recommended for best results**

### 2. Set Appropriate Batch Size

- GPU with 8GB: batch_size=16
- GPU with 4GB: batch_size=8
- CPU only: batch_size=4

### 3. Number of Clusters

- Start with expected number of classes
- Use HDBSCAN to auto-discover clusters
- Can always re-run with different numbers

### 4. Original Filenames

- **Always preserved** - no need to worry about losing original names
- Only folder structure changes
- Safe to use on important datasets

---

## Troubleshooting

**Issue**: Out of memory
**Fix**: Reduce batch_size

**Issue**: Clustering doesn't make sense
**Fix**: Try different model or increase n_clusters

**Issue**: HDBSCAN not available
**Fix**: `pip install autoannotate-vision[hdbscan]` or use kmeans/spectral

**Issue**: HTML preview doesn't open
**Fix**: Check your default browser settings

---

## Next Steps

- Read [API Reference](api_reference.md) for detailed documentation
- Check [GitHub](https://github.com/Metamind-Innovations/autoannotate-vision) for examples
- Report issues or request features

**Made for the RAIDO Project** 🚀

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
autoannotate-images-cli annotate \
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
class_names = annotator.interactive_labeling(n_samples=6)
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

models = ["clip", "dinov2", "dinov2-large", "siglip2"]

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
autoannotate-images-cli validate ./images --recursive
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
- **SigLIP2**: Latest Google model - **Recommended for best results**

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