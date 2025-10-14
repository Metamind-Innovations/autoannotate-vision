# AutoAnnotate-Vision Documentation

Welcome to the **AutoAnnotate-Vision** documentation! This SDK provides state-of-the-art unsupervised auto-annotation for image classification tasks.

## What is AutoAnnotate-Vision?

AutoAnnotate-Vision is a Python SDK that automatically clusters and organizes unlabeled image datasets using cutting-edge vision models (CLIP, DINOv2) and advanced clustering algorithms. It's designed to bootstrap image classification datasets quickly and efficiently.

## Key Features

- **🎯 SOTA Vision Embeddings**: Support for CLIP, DINOv2, and DINOv2-Large models
- **🔬 Multiple Clustering Algorithms**: K-means, HDBSCAN, Spectral Clustering, DBSCAN
- **💬 Interactive Labeling**: CLI-based workflow with representative sample visualization
- **📁 Automated Organization**: Automatic folder structure creation and file renaming
- **✂️ Dataset Splitting**: Built-in train/val/test split functionality
- **💾 Export Formats**: CSV and JSON label file generation
- **🔌 Programmatic API**: Full Python API for custom workflows

## Quick Start

### Installation

```bash
pip install autoannotate-vision
```

### Basic Usage

```bash
autoannotate annotate \
    /path/to/images \
    /path/to/output \
    --n-clusters 10 \
    --method kmeans \
    --model dinov2
```

### Python API

```python
from autoannotate import AutoAnnotator

annotator = AutoAnnotator(
    input_dir="./images",
    output_dir="./organized",
    model="dinov2",
    clustering_method="kmeans",
    n_clusters=10
)

result = annotator.run_full_pipeline(create_splits=True)
```

## Use Cases

AutoAnnotate-Vision is perfect for:

- **Research Projects**: Quickly organize unlabeled datasets for experiments
- **Data Preparation**: Bootstrap annotation for large-scale ML projects
- **Dataset Exploration**: Discover natural groupings in image collections
- **Quality Control**: Identify outliers and data quality issues
- **Transfer Learning**: Prepare datasets for fine-tuning vision models

## Architecture

```
Input Images → Embedding Extraction → Clustering → Interactive Labeling → Organized Dataset
                     ↓                     ↓              ↓                    ↓
                  CLIP/DINOv2      K-means/HDBSCAN   CLI Interface      Folders + Metadata
```

## Navigation

- **[API Reference](api_reference.md)**: Complete API documentation
- **[Tutorials](tutorials.md)**: Step-by-step guides and examples
- **[GitHub Repository](https://github.com/yourusername/autoannotate-vision)**: Source code and issues

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/autoannotate-vision/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/autoannotate-vision/discussions)
- **Email**: research@metamind-innovations.com

## License

AutoAnnotate-Vision is released under the MIT License. See [LICENSE](https://github.com/yourusername/autoannotate-vision/blob/main/LICENSE) for details.

## Citation

If you use AutoAnnotate-Vision in your research, please cite:

```bibtex
@software{autoannotate_vision,
  title={AutoAnnotate-Vision: SOTA Unsupervised Auto-Annotation for Image Classification},
  author={MetaMind Innovations},
  year={2025},
  url={https://github.com/yourusername/autoannotate-vision}
}
```

## Acknowledgments

Built for the **RAIDO Project** (HORIZON-CL4-2023-HUMAN-01-CNECT) by MetaMind Innovations (MINDS).

Powered by:
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [scikit-learn](https://scikit-learn.org/)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [UMAP](https://umap-learn.readthedocs.io/)