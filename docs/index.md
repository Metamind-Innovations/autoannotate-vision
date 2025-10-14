# AutoAnnotate-Vision Documentation

Welcome to **AutoAnnotate-Vision** - a state-of-the-art SDK for unsupervised image auto-annotation with a graphical user interface!

## What is AutoAnnotate-Vision?

AutoAnnotate-Vision automatically clusters and organizes unlabeled image datasets using cutting-edge vision models (CLIP, DINOv2). It features:

- 🎨 **Graphical User Interface** for easy folder selection and configuration
- 🖼️ **HTML Preview** that opens sample images in your browser before labeling
- 📁 **Smart Organization** that preserves original filenames
- 🤖 **SOTA Models** including DINOv2 and CLIP
- ✂️ **Auto Splits** for train/val/test datasets

## Quick Start

### Using the GUI (Easiest!)

```bash
python run_autoannotate_gui.py
```

1. Select input folder with images
2. Select output folder
3. Set number of classes
4. Click "Start Auto-Annotation"
5. Review HTML previews in browser
6. Name each cluster
7. Done! Images organized with original filenames preserved

### Using the CLI

```bash
autoannotate annotate /path/to/images /path/to/output \
    --n-clusters 10 \
    --method kmeans \
    --model dinov2
```

### Using Python API

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./output",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=5
)

result = annotator.run_full_pipeline()
```

## Key Features

### HTML Image Preview

When labeling clusters, AutoAnnotate automatically opens beautiful HTML previews in your browser showing representative sample images from each cluster. No more guessing what's in each cluster!

### Original Filenames Preserved

Unlike other tools, AutoAnnotate **preserves your original filenames**. Images are organized into class folders without renaming:

```
output/
├── cats/
│   ├── IMG_001.jpg  ✅ Original name!
│   ├── my_cat.jpg   ✅ Original name!
└── dogs/
    ├── photo_5.jpg  ✅ Original name!
```

### Optional Dependencies

- **HDBSCAN** is optional - install with `pip install autoannotate-vision[hdbscan]`
- Core functionality works with just K-means, Spectral, and DBSCAN

## Use Cases

- **Research Projects**: Quickly organize unlabeled datasets
- **Data Preparation**: Bootstrap annotation for ML projects
- **Dataset Exploration**: Discover natural groupings in images
- **Quality Control**: Identify outliers and issues

## Installation

```bash
pip install autoannotate-vision
```

Or from source:
```bash
git clone https://github.com/Metamind-Innovations/autoannotate-vision.git
cd autoannotate-vision
pip install -e .
```

## Navigation

- **[API Reference](api_reference.md)**: Complete API documentation
- **[Tutorials](tutorials.md)**: Step-by-step guides

## Support

- **GitHub**: [Metamind-Innovations/autoannotate-vision](https://github.com/Metamind-Innovations/autoannotate-vision)
- **Issues**: [Report bugs or request features](https://github.com/Metamind-Innovations/autoannotate-vision/issues)

## License

MIT License - see [LICENSE](https://github.com/Metamind-Innovations/autoannotate-vision/blob/main/LICENSE)

**Made for the [RAIDO Project](https://raido-project.eu/)**