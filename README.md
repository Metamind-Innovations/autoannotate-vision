# AutoAnnotate-Vision 🎯

**State-of-the-art unsupervised auto-annotation SDK for image classification with GUI**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AutoAnnotate-Vision automatically clusters and organizes unlabeled image datasets using cutting-edge vision models (CLIP, DINOv2). Features a **graphical user interface** for easy use and **HTML preview** for visual cluster inspection.

## ✨ Features

- 🎨 **Graphical User Interface**: Easy folder browsers and visual controls
- 🖼️ **HTML Image Preview**: View cluster samples in browser before labeling
- 🤖 **SOTA Vision Models**: CLIP, DINOv2, DINOv2-Large
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

The easiest way to use AutoAnnotate-Vision:

```bash
python run_autoannotate_gui.py
```

**Workflow:**
1. 📁 Select input folder with images
2. 📂 Select output folder  
3. 🔢 Set number of classes
4. 🤖 Choose model (dinov2 recommended)
5. ▶️ Click "Start Auto-Annotation"

The app will cluster images and open **HTML previews** in your browser showing sample images from each cluster for easy labeling!

## 💻 CLI Usage

```bash
autoannotate annotate /path/to/images /path/to/output \
    --n-clusters 10 \
    --method kmeans \
    --model dinov2 \
    --create-splits
```

## 🐍 Python API

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=5
)

result = annotator.run_full_pipeline(create_splits=True)
print(f"Processed {result['n_images']} images")
```

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

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| CLIP | ⚡⚡ | ⭐⭐⭐ | General images |
| DINOv2 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Recommended |
| DINOv2-Large | ⚡ | ⭐⭐⭐⭐⭐ | High-quality |

## 🔍 Pre-Push Checklist

Before pushing code:

```bash
# Format code
black src/autoannotate tests

# Run tests
pytest tests/ -v

# Check everything at once
black --check src/autoannotate tests && pytest tests/ -v
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. **Format with Black**: `black src/autoannotate tests`
4. **Run tests**: `pytest tests/ -v`
5. Push and create PR

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built with PyTorch, Transformers, scikit-learn. Vision models: CLIP, DINOv2.

**Made for the [RAIDO Project](https://raido-project.eu/)**