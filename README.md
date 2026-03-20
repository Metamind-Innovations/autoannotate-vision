# AutoAnnotate-Vision 🎯

**State-of-the-art unsupervised auto-annotation SDK for image classification with GUI**

[![Tests](https://github.com/Metamind-Innovations/autoannotate-vision/actions/workflows/tests.yml/badge.svg)](https://github.com/Metamind-Innovations/autoannotate-vision/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AutoAnnotate-Vision** automatically clusters and organizes unlabeled image datasets using cutting-edge vision models
(CLIP, DINOv2, SigLIP2). It features a **GUI** and **interactive HTML preview** with Plotly for visual cluster
inspection, as well as a **CLI tool**.

## ✨ Features

- 🎨 **Graphical User Interface**: Easy folder browsers and visual controls
- 🖼️ **HTML Image Preview**: View cluster samples in browse before labeling
- 🤖 **SOTA Vision Models**: CLIP, DINOv2, DINOv2-Large, SigLIP2
- 🔬 **Multiple Clustering**: K-means, Spectral, DBSCAN, HDBSCAN (optional)
- 📁 **Smart Organization**: Preserves original filenames
- ✂️ **Auto Splits**: Train/val/test dataset splitting
- 💾 **Export**: CSV, JSON formats
- 🔌 **Python API**: Full programmatic control

## 🚀 Installation

```bash
pip install autoannotate-vision
```

## 🎨 Quick Start - GUI

The easiest and most simplified way to use AutoAnnotate-Vision:

```bash
autoannotate-images
```

**Note:** Windows users need to have the latest C++ Redistributable installed which can be
found [here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version)

**Workflow:**

1. 📁 Select input folder with images
2. 📂 Select output folder
3. 🔢 Set number of classes
4. 🤖 Choose model (SigLIP2 or DINOv2 recommended)
5. ▶️ Click "Start Auto-Annotation"

The app will cluster images and open **HTML previews** in your browser showing sample images from each cluster for easy
labeling!

## 💻 CLI Usage

For extra commands and utilities.

```bash
autoannotate-images-cli annotate /path/to/images /path/to/output \
    --n-clusters 10 \
    --method kmeans \
    --model siglip2 \
    --create-splits
```

**Available models:** `clip`, `dinov2`, `dinov2-large`, `siglip2`

### Command Arguments

The `autoannotate-images-cli annotate` command accepts the following arguments:

**Required Arguments:**

- `INPUT_DIR` - Path to the directory containing images to annotate
- `OUTPUT_DIR` - Path where annotated images and metadata will be saved

**Optional Arguments:**

- `-n, --n-clusters INTEGER` - Number of clusters to create (required for kmeans/spectral methods)
- `-m, --method [kmeans|hdbscan|spectral|dbscan]` - Clustering algorithm to use (default: kmeans)
- `--model [clip|dinov2|dinov2-large|siglip2]` - Vision model for embeddings (default: siglip2)
- `-b, --batch-size INTEGER` - Batch size for embedding extraction (default: 32)
- `-r, --recursive` - Search for images in subdirectories recursively
- `--reduce-dims / --no-reduce-dims` - Apply dimensionality reduction before clustering
- `--n-samples INTEGER` - Number of representative samples per cluster for preview
- `--copy / --symlink` - Copy image files or create symbolic links (default: copy)
- `--create-splits` - Automatically create train/val/test dataset splits
- `--export-format [csv|json]` - Format for exporting labels (default: csv)

**Examples:**

```bash
# Basic usage with 5 clusters
autoannotate-images-cli annotate ./my_images ./output --n-clusters 5

# Use DBSCAN
autoannotate-images-cli annotate ./my_images ./output --method dbscan

# Use larger batch size with dimensionality reduction
autoannotate-images-cli annotate ./my_images ./output \
    --n-clusters 10 \
    --batch-size 64 \
    --reduce-dims

```

## 🐍 Python API

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="siglip2",  # or "dinov2", "dinov2-large", "clip"
    clustering_method="kmeans",
    n_clusters=5,
    batch_size=32
)

result = annotator.run_full_pipeline(create_splits=True)
print(f"Processed {result['n_images']} images into {result['n_clusters']} classes")
```

**Available models:** `clip`, `dinov2`, `dinov2-large`, `siglip2`
**Available clustering methods:** `kmeans`, `hdbscan`, `spectral`, `dbscan`

## 📁 Output Structure

```
output/
├── metadata.json
├── labels.csv
├── cats/              # Your class names
│   ├── IMG_001.jpg   # Original filenames preserved!
│   └── ...
├── dogs/
└── splits/            # train/val/test. Availabe only through CLI --create-splits
    ├── train/
    ├── val/
    └── test/
```

## 🧠 Model Comparison

| Model        | Speed | Quality | Notes                                        |
|--------------|-------|---------|----------------------------------------------|
| CLIP         | ⚡⚡    | ⭐⭐⭐     | General-purpose, good for diverse datasets   |
| DINOv2       | ⚡⚡⚡   | ⭐⭐⭐⭐    | Fast, self-supervised, excellent for objects |
| DINOv2-Large | ⚡     | ⭐⭐⭐⭐⭐   | Best quality, slower, great for fine details |
| SigLIP2      | ⚡⚡    | ⭐⭐⭐⭐⭐   | Latest Google model - **Recommended** 🌟     |

**Recommendation:** Start with **SigLIP2** for best results, or **DINOv2** for faster processing.

## 🔧 Features

- ✅ **Fast Image Processing**: All models use optimized processors (`use_fast=True`) for better performance
- ✅ **Normalized Embeddings**: All embeddings are L2-normalized for consistent similarity measurements
- ✅ **Batch Processing**: Efficient batch processing with configurable batch sizes
- ✅ **GPU Support**: Automatic GPU detection and usage when available
- ✅ **Progress Tracking**: Real-time progress bars for all operations
- ✅ **HTML Previews**: Interactive HTML preview for visual cluster inspection before labeling

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. All actions
   from [tests.yml](https://github.com/Metamind-Innovations/autoannotate-vision/blob/main/.github/workflows/tests.yml)
   should pass
4. Push and create PR

## 📄 License

MIT License - see [LICENSE](https://github.com/Metamind-Innovations/autoannotate-vision/blob/main/LICENSE) file.

## 🙏 Acknowledgments

Built with `PyTorch`, `Transformers`, `scikit-learn` and more. Vision models: CLIP, DINOv2, SigLIP2.

**Made for the [RAIDO Project](https://raido-project.eu/), from [MetaMind Innovations](https://metamind.gr/)**

🇪🇺 RAIDO project has received funding from the European Union's Horizon Europe research and innovation programme under **Grant Agreement No. 101135800**.

*Disclaimer: Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Commission. Neither the European Union nor the European Commission can be held responsible for them.*

---

**Sister Project**: [AutoAnnotate-Timeseries](https://github.com/Metamind-Innovations/autoannotate-timeseries) - For
time series auto-annotation
