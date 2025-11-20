# AutoAnnotate-Vision 🎯

**State-of-the-art unsupervised auto-annotation SDK for image classification with GUI**

[![Tests](https://github.com/Metamind-Innovations/autoannotate-vision/actions/workflows/tests.yml/badge.svg)](https://github.com/Metamind-Innovations/autoannotate-vision/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AutoAnnotate-Vision automatically clusters and organizes unlabeled image datasets using cutting-edge vision models
(CLIP, DINOv2, SigLIP2). Features a **graphical user interface** for easy use and **HTML preview** for visual cluster
inspection.

## ✨ Features

- 🎨 **Graphical User Interface**: Easy folder browsers and visual controls
- 🖼️ **HTML Image Preview**: View cluster samples in browse before labeling
- 🤖 **SOTA Vision Models**: CLIP, DINOv2, DINOv2-Large, SigLIP2
- 🔬 **Multiple Clustering**: K-means, Spectral, DBSCAN
- 📁 **Smart Organization**: Preserves original filenames
- ✂️ **Auto Splits**: Train/val/test dataset splitting
- 💾 **Export**: CSV, JSON formats
- 🔌 **Python API**: Full programmatic control

## 🚀 Installation

```bash
pip install autoannotate-vision
```

Or from source:

```bash
git clone https://github.com/Metamind-Innovations/autoannotate-vision.git
cd autoannotate-vision
pip install -e .
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
└── splits/            # train/val/test
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
| SigLIP2      | ⚡⚡⚡   | ⭐⭐⭐⭐⭐   | Latest Google model - **Recommended** 🌟     |

**Recommendation:** Start with **SigLIP2** for best results, or **DINOv2** for faster processing.

## 🔧 Features & Improvements

- ✅ **Fast Image Processing**: All models use optimized processors (`use_fast=True`) for better performance
- ✅ **Normalized Embeddings**: All embeddings are L2-normalized for consistent similarity measurements
- ✅ **Batch Processing**: Efficient batch processing with configurable batch sizes
- ✅ **GPU Support**: Automatic GPU detection and usage when available
- ✅ **Progress Tracking**: Real-time progress bars for all operations
- ✅ **HTML Previews**: Interactive HTML preview for visual cluster inspection before labeling

## 🔍 Pre-Push Checklist

Before pushing code:

```bash
# Format code
black src/autoannotate tests

# Run tests
pytest tests/ -v

# Typing
mypy src/autoannotate --ignore-missing-imports
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. **Format with Black**: `black src/autoannotate tests`
4. **Check typing with mypy**: `mypy src/autoannotate --ignore-missing-imports`
5. **Run tests**: `pytest tests/ -v`
6. Push and create PR

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built with PyTorch, Transformers, scikit-learn. Vision models: CLIP, DINOv2, SigLIP2.

**Made for the [RAIDO Project](https://raido-project.eu/), from [MetaMind Innovations](https://metamind.gr/)**