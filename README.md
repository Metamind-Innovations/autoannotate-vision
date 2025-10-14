# autoannotate-vision

# AutoAnnotate-Vision 🎯

**State-of-the-art unsupervised auto-annotation SDK for image classification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AutoAnnotate-Vision automatically clusters and organizes unlabeled image datasets using cutting-edge vision models (CLIP, DINOv2) and advanced clustering algorithms. Perfect for bootstrapping image classification datasets.

## ✨ Features

- **SOTA Vision Embeddings**: CLIP, DINOv2, DINOv2-Large
- **Multiple Clustering Algorithms**: K-means, HDBSCAN, Spectral, DBSCAN
- **Interactive Labeling**: CLI-based workflow with representative samples
- **Automated Organization**: Folder structure + renamed files
- **Train/Val/Test Splits**: Automatic dataset splitting
- **Export Formats**: CSV, JSON label files
- **Programmatic API**: Full Python API for integration

## 🚀 Installation

### From PyPI (when published)
```bash
pip install autoannotate-vision
```

### From source
```bash
git clone https://github.com/yourusername/autoannotate-vision.git
cd autoannotate-vision
pip install -e .
```

### With development dependencies
```bash
pip install -e ".[dev]"
```

## 📖 Quick Start

### CLI Usage

```bash
autoannotate annotate \
    /path/to/images \
    /path/to/output \
    --n-clusters 10 \
    --method kmeans \
    --model dinov2 \
    --create-splits \
    --export-format csv
```

### Python API Usage

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./data/unlabeled",
    output_dir="./data/annotated",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=10
)

result = annotator.run_full_pipeline(
    n_samples=5,
    create_splits=True,
    export_format="csv"
)

print(f"Processed {result['n_images']} images into {result['n_clusters']} classes")
```

### Step-by-Step API

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./organized",
    model="dinov2",
    clustering_method="hdbscan",
    reduce_dims=True
)

images, paths = annotator.load_images()

embeddings = annotator.extract_embeddings()

labels = annotator.cluster()

stats = annotator.get_cluster_stats()
print(f"Found {stats['n_clusters']} clusters")

class_names = annotator.interactive_labeling(n_samples=5)

annotator.organize_dataset(copy_files=True)

annotator.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

annotator.export_labels(format="csv")
```

## 🎛️ CLI Commands

### `annotate` - Main annotation command
```bash
autoannotate annotate INPUT_DIR OUTPUT_DIR [OPTIONS]

Options:
  -n, --n-clusters INTEGER        Number of clusters (required for kmeans/spectral)
  -m, --method [kmeans|hdbscan|spectral|dbscan]
                                  Clustering method (default: kmeans)
  --model [clip|dinov2|dinov2-large]
                                  Embedding model (default: dinov2)
  -b, --batch-size INTEGER        Batch size for embeddings (default: 32)
  -r, --recursive                 Search images recursively
  --reduce-dims / --no-reduce-dims
                                  Apply dimensionality reduction (default: True)
  --n-samples INTEGER             Representative samples per cluster (default: 5)
  --copy / --symlink              Copy files or create symlinks (default: copy)
  --create-splits                 Create train/val/test splits
  --export-format [csv|json]      Export labels format (default: csv)
```

### `validate` - Validate image files
```bash
autoannotate validate INPUT_DIR [OPTIONS]

Options:
  -r, --recursive                 Search recursively
```

## 🏗️ Output Structure

```
output_dir/
├── metadata.json              # Full annotation metadata
├── labels.csv                 # Image labels in CSV format
├── classA/
│   ├── classA_1.jpg
│   ├── classA_2.jpg
│   └── ...
├── classB/
│   ├── classB_1.jpg
│   └── ...
├── unclustered/               # Images that couldn't be clustered
│   └── ...
└── splits/                    # Train/val/test splits (if created)
    ├── train/
    │   ├── classA/
    │   └── classB/
    ├── val/
    └── test/
```

## 🧠 Model Comparison

| Model | Speed | Quality | Memory | Best For |
|-------|-------|---------|--------|----------|
| **CLIP** | ⚡⚡ | ⭐⭐⭐ | 💾💾 | General images, text-image |
| **DINOv2** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💾💾 | General images, SOTA |
| **DINOv2-Large** | ⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾 | High-quality, complex data |

## 🎯 Clustering Method Comparison

| Method | Auto-clusters | Best For | Requires n_clusters |
|--------|---------------|----------|---------------------|
| **K-means** | ❌ | Fast, balanced clusters | ✅ |
| **HDBSCAN** | ✅ | Arbitrary shapes, noise handling | ❌ |
| **Spectral** | ❌ | Non-convex clusters | ✅ |
| **DBSCAN** | ✅ | Density-based, noise handling | ❌ |

## 📊 Advanced Examples

### Using HDBSCAN (auto-determines clusters)
```bash
autoannotate annotate ./images ./output \
    --method hdbscan \
    --model dinov2-large \
    --recursive
```

### Creating custom splits
```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    clustering_method="spectral",
    n_clusters=20
)

annotator.run_full_pipeline()

annotator.create_splits(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### Using as a library component
```python
from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from PIL import Image

images = [Image.open(f"img_{i}.jpg") for i in range(100)]

extractor = EmbeddingExtractor(model_name="dinov2", batch_size=16)
embeddings = extractor.extract_batch(images)

clusterer = ClusteringEngine(method="hdbscan", reduce_dims=True)
labels = clusterer.fit_predict(embeddings)

stats = clusterer.get_cluster_stats(labels)
print(f"Discovered {stats['n_clusters']} natural clusters")
```

## 🧪 Testing

```bash
pytest tests/

pytest tests/ --cov=autoannotate --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/transformers/), and [scikit-learn](https://scikit-learn.org/)
- Vision models: [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2)
- CLI powered by [Click](https://click.palletsprojects.com/) and [Rich](https://rich.readthedocs.io/)

## 📮 Contact

**MetaMind Innovations (MINDS)**  
Research & Development Team  
[Your contact information]

## 🗺️ Roadmap

- [ ] Web-based UI for labeling
- [ ] Support for video frame annotation
- [ ] Active learning integration
- [ ] Multi-modal clustering (image + text)
- [ ] Export to popular frameworks (COCO, YOLO, Pascal VOC)
- [ ] Confidence scoring for cluster assignments
- [ ] Hierarchical clustering support

---

Made with ❤️ for the RAIDO Project
