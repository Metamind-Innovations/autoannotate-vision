import pytest
from PIL import Image

from autoannotate.utils.image_loader import ImageLoader, SUPPORTED_FORMATS


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary directory with test images."""
    for i in range(10):
        img = Image.new("RGB", (200, 200), color=(i * 25, i * 10, i * 5))
        img.save(tmp_path / f"image_{i}.jpg")
    return tmp_path


@pytest.fixture
def nested_image_dir(tmp_path):
    """Create nested directories with images."""
    # Root level
    for i in range(3):
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / f"root_{i}.jpg")

    # Subfolder level 1
    sub1 = tmp_path / "sub1"
    sub1.mkdir()
    for i in range(3):
        img = Image.new("RGB", (100, 100))
        img.save(sub1 / f"sub1_{i}.png")

    # Subfolder level 2
    sub2 = sub1 / "sub2"
    sub2.mkdir()
    for i in range(2):
        img = Image.new("RGB", (100, 100))
        img.save(sub2 / f"sub2_{i}.bmp")

    return tmp_path


class TestImageLoader:
    def test_initialization_valid_directory(self, temp_image_dir):
        loader = ImageLoader(temp_image_dir, recursive=False)
        assert loader.input_dir == temp_image_dir
        assert loader.recursive is False

    def test_initialization_directory_not_exists(self):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            ImageLoader("/nonexistent/path")

    def test_initialization_path_is_file(self, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(NotADirectoryError, match="Not a directory"):
            ImageLoader(file_path)

    def test_load_image_paths_non_recursive(self, temp_image_dir):
        loader = ImageLoader(temp_image_dir, recursive=False)
        paths = loader.load_image_paths()

        assert len(paths) == 10
        assert all(p.suffix == ".jpg" for p in paths)
        assert all(p.is_file() for p in paths)

    def test_load_image_paths_recursive(self, nested_image_dir):
        loader = ImageLoader(nested_image_dir, recursive=True)
        paths = loader.load_image_paths()

        # 3 root + 3 sub1 + 2 sub2 = 8 total
        assert len(paths) == 8

    def test_load_image_paths_non_recursive_no_subfolders(self, nested_image_dir):
        loader = ImageLoader(nested_image_dir, recursive=False)
        paths = loader.load_image_paths()

        # Only 3 root images
        assert len(paths) == 3

    def test_load_image_paths_sorted(self, temp_image_dir):
        loader = ImageLoader(temp_image_dir)
        paths = loader.load_image_paths()

        # Should be sorted
        assert paths == sorted(paths)

    def test_load_image_paths_empty_directory(self, tmp_path):
        loader = ImageLoader(tmp_path)

        with pytest.raises(ValueError, match="No valid images found"):
            loader.load_image_paths()

    def test_load_image_paths_no_images(self, tmp_path):
        # Create non-image files
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.pdf").write_bytes(b"pdf")

        loader = ImageLoader(tmp_path)

        with pytest.raises(ValueError, match="No valid images found"):
            loader.load_image_paths()

    def test_load_images_success(self, temp_image_dir):
        loader = ImageLoader(temp_image_dir)
        images, paths = loader.load_images()

        assert len(images) == 10
        assert len(paths) == 10
        assert all(isinstance(img, Image.Image) for img in images)
        assert all(img.mode == "RGB" for img in images)

    def test_load_images_with_max_size(self, temp_image_dir):
        loader = ImageLoader(temp_image_dir)
        images, paths = loader.load_images(max_size=(50, 50))

        assert len(images) == 10
        # Images should be resized to fit within max_size
        for img in images:
            assert img.width <= 50
            assert img.height <= 50

    def test_load_images_various_formats(self, tmp_path):
        # Create images in different formats
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / "test.jpg")
        img.save(tmp_path / "test.png")
        img.save(tmp_path / "test.bmp")
        img.save(tmp_path / "test.gif")

        loader = ImageLoader(tmp_path)
        images, paths = loader.load_images()

        assert len(images) == 4

    def test_load_images_skip_corrupted(self, tmp_path):
        # Create valid image
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / "valid.jpg")

        # Create corrupted "image"
        (tmp_path / "corrupted.jpg").write_bytes(b"not an image")

        loader = ImageLoader(tmp_path)
        images, paths = loader.load_images()

        # Should only load the valid image
        assert len(images) == 1
        assert len(paths) == 1

    def test_load_images_all_corrupted(self, tmp_path):
        # Create only corrupted images
        (tmp_path / "corrupted1.jpg").write_bytes(b"corrupted")
        (tmp_path / "corrupted2.png").write_bytes(b"also corrupted")

        loader = ImageLoader(tmp_path)

        with pytest.raises(ValueError, match="No valid images could be loaded"):
            loader.load_images()

    def test_validate_image_valid(self, tmp_path):
        img_path = tmp_path / "valid.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        assert ImageLoader.validate_image(img_path) is True

    def test_validate_image_corrupted(self, tmp_path):
        img_path = tmp_path / "corrupted.jpg"
        img_path.write_bytes(b"corrupted data")

        assert ImageLoader.validate_image(img_path) is False

    def test_validate_image_nonexistent(self):
        assert ImageLoader.validate_image("/nonexistent/image.jpg") is False

    def test_supported_formats_constant(self):
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".jpeg" in SUPPORTED_FORMATS
        assert ".png" in SUPPORTED_FORMATS
        assert ".bmp" in SUPPORTED_FORMATS
        assert ".gif" in SUPPORTED_FORMATS
        assert ".tiff" in SUPPORTED_FORMATS
        assert ".webp" in SUPPORTED_FORMATS

    def test_load_images_grayscale_conversion(self, tmp_path):
        # Create grayscale image
        gray_img = Image.new("L", (100, 100), color=128)
        gray_img.save(tmp_path / "gray.png")

        loader = ImageLoader(tmp_path)
        images, paths = loader.load_images()

        # Should convert to RGB
        assert len(images) == 1
        assert images[0].mode == "RGB"

    def test_load_images_rgba_conversion(self, tmp_path):
        # Create RGBA image
        rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        rgba_img.save(tmp_path / "rgba.png")

        loader = ImageLoader(tmp_path)
        images, paths = loader.load_images()

        # Should convert to RGB
        assert len(images) == 1
        assert images[0].mode == "RGB"

    def test_path_types(self, temp_image_dir):
        # Test with Path object
        loader1 = ImageLoader(temp_image_dir)
        paths1 = loader1.load_image_paths()

        # Test with string
        loader2 = ImageLoader(str(temp_image_dir))
        paths2 = loader2.load_image_paths()

        assert len(paths1) == len(paths2)

    def test_case_insensitive_extensions(self, tmp_path):
        # Create images with uppercase extensions
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / "test1.JPG")
        img.save(tmp_path / "test2.PNG")
        img.save(tmp_path / "test3.BMP")

        loader = ImageLoader(tmp_path)
        paths = loader.load_image_paths()

        # Should find all images regardless of case
        assert len(paths) == 3
