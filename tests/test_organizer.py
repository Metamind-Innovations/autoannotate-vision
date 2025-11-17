import pytest
import json
import numpy as np
from PIL import Image
import shutil

from autoannotate.core.organizer import DatasetOrganizer


@pytest.fixture
def temp_images(tmp_path):
    """Create temporary test images."""
    image_paths = []
    for i in range(20):
        img_path = tmp_path / f"image_{i}.jpg"
        img = Image.new("RGB", (100, 100), color=(i * 10, i * 5, i * 3))
        img.save(img_path)
        image_paths.append(img_path)
    return image_paths


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    out_dir = tmp_path / "output"
    return out_dir


@pytest.fixture
def sample_labels():
    """Generate sample clustering labels."""
    # 20 images: 8 in cluster 0, 7 in cluster 1, 3 in cluster 2, 2 as noise (-1)
    return np.array([0, 0, 1, 0, 1, 2, 0, 1, -1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, -1])


@pytest.fixture
def class_names():
    """Sample class names for clusters."""
    return {0: "cats", 1: "dogs", 2: "birds"}


class TestDatasetOrganizer:
    def test_initialization(self, output_dir):
        organizer = DatasetOrganizer(output_dir)
        assert organizer.output_dir == output_dir
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_organize_by_clusters_copy_files(
        self, temp_images, output_dir, sample_labels, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(
            temp_images, sample_labels, class_names, copy_files=True, create_symlinks=False
        )

        # Check metadata structure
        assert "metadata" in metadata
        assert "classes" in metadata
        assert metadata["metadata"]["total_images"] == 20
        assert metadata["metadata"]["n_classes"] == 3

        # Check class directories were created
        assert (output_dir / "cats").exists()
        assert (output_dir / "dogs").exists()
        assert (output_dir / "birds").exists()

        # Check unclustered directory for noise points
        assert (output_dir / "unclustered").exists()

        # Check files were copied
        assert len(list((output_dir / "cats").glob("*"))) == 8
        assert len(list((output_dir / "dogs").glob("*"))) == 7
        assert len(list((output_dir / "birds").glob("*"))) == 3
        assert len(list((output_dir / "unclustered").glob("*"))) == 2

    def test_metadata_json_created(self, temp_images, output_dir, sample_labels, class_names):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(temp_images, sample_labels, class_names, copy_files=True)

        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "metadata" in metadata
        assert "classes" in metadata
        assert "created_at" in metadata["metadata"]

    def test_class_metadata_content(self, temp_images, output_dir, sample_labels, class_names):
        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(
            temp_images, sample_labels, class_names, copy_files=True
        )

        # Check class metadata
        assert "cats" in metadata["classes"]
        assert "dogs" in metadata["classes"]
        assert "birds" in metadata["classes"]

        # Check counts
        assert metadata["classes"]["cats"]["count"] == 8
        assert metadata["classes"]["dogs"]["count"] == 7
        assert metadata["classes"]["birds"]["count"] == 3

        # Check image info
        assert "images" in metadata["classes"]["cats"]
        assert len(metadata["classes"]["cats"]["images"]) == 8

    def test_duplicate_filenames_handling(self, temp_images, output_dir, class_names):
        # Create duplicate filenames
        duplicate_path = temp_images[0].parent / temp_images[0].name
        labels = np.array([0, 0])  # Both images in same cluster

        # Create second image with same name in different location
        temp_dir2 = temp_images[0].parent / "subdir"
        temp_dir2.mkdir()
        duplicate_img = temp_dir2 / temp_images[0].name
        img = Image.new("RGB", (100, 100))
        img.save(duplicate_img)

        paths = [temp_images[0], duplicate_img]

        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(paths, labels, class_names, copy_files=True)

        # Check that both files exist (one renamed)
        cats_files = list((output_dir / "cats").glob("*"))
        assert len(cats_files) == 2

    def test_create_split_invalid_ratios(self, temp_images, output_dir, sample_labels, class_names):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(temp_images, sample_labels, class_names, copy_files=True)

        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            organizer.create_split(train_ratio=0.5, val_ratio=0.3, test_ratio=0.1)

    def test_export_labels_csv(self, temp_images, output_dir, sample_labels, class_names):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(temp_images, sample_labels, class_names, copy_files=True)

        output_path = organizer.export_labels_file(format="csv")

        assert output_path.exists()
        assert output_path.name == "labels.csv"

        # Check CSV content
        with open(output_path) as f:
            lines = f.readlines()

        assert lines[0].strip() == "image_path,class_name,cluster_id"
        assert len(lines) > 1  # Header + data lines

    def test_export_labels_json(self, temp_images, output_dir, sample_labels, class_names):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(temp_images, sample_labels, class_names, copy_files=True)

        output_path = organizer.export_labels_file(format="json")

        assert output_path.exists()
        assert output_path.name == "labels.json"

        # Check JSON content
        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0
        assert "image_path" in data[0]
        assert "class_name" in data[0]
        assert "cluster_id" in data[0]

    def test_export_labels_invalid_format(
        self, temp_images, output_dir, sample_labels, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(temp_images, sample_labels, class_names, copy_files=True)

        with pytest.raises(ValueError, match="Unsupported format"):
            organizer.export_labels_file(format="xml")

    def test_export_labels_without_organization(self, output_dir):
        organizer = DatasetOrganizer(output_dir)

        with pytest.raises(FileNotFoundError, match="metadata.json not found"):
            organizer.export_labels_file(format="csv")

    def test_empty_class_handling(self, temp_images, output_dir):
        # Only one cluster with all images
        labels = np.array([0] * 20)
        class_names = {0: "all_images"}

        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(temp_images, labels, class_names, copy_files=True)

        assert len(metadata["classes"]) == 1
        assert metadata["classes"]["all_images"]["count"] == 20
