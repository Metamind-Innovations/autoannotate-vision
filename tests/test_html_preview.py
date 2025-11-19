import pytest
import numpy as np
from PIL import Image

from autoannotate.ui.html_preview import generate_cluster_preview_html, open_html_in_browser


@pytest.fixture
def temp_images(tmp_path):
    """Create temporary test images."""
    image_paths = []
    for i in range(10):
        img_path = tmp_path / f"image_{i}.jpg"
        img = Image.new("RGB", (100, 100), color=(i * 20, i * 10, i * 5))
        img.save(img_path)
        image_paths.append(img_path)
    return image_paths


class TestGenerateClusterPreviewHTML:
    def test_generate_html_default_output(self, temp_images):
        indices = np.array([0, 2, 4])
        cluster_size = 5

        html_path = generate_cluster_preview_html(
            cluster_id=0, image_paths=temp_images, indices=indices, cluster_size=cluster_size
        )

        # Check default path
        assert html_path.exists()
        assert html_path.name == "cluster_0_preview.html"

        # Clean up
        html_path.unlink()

    def test_generate_html_custom_output(self, temp_images, tmp_path):
        indices = np.array([1, 3, 5])
        cluster_size = 7
        output_path = tmp_path / "custom_preview.html"

        html_path = generate_cluster_preview_html(
            cluster_id=1,
            image_paths=temp_images,
            indices=indices,
            cluster_size=cluster_size,
            output_path=output_path,
        )

        assert html_path == output_path
        assert html_path.exists()

    def test_html_content_structure(self, temp_images, tmp_path):
        indices = np.array([0, 1, 2])
        output_path = tmp_path / "test.html"

        html_path = generate_cluster_preview_html(
            cluster_id=0,
            image_paths=temp_images,
            indices=indices,
            cluster_size=5,
            output_path=output_path,
        )

        # Read HTML content
        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        # Check HTML structure
        assert "<!DOCTYPE html>" in html_content
        assert "<html>" in html_content
        assert "</html>" in html_content
        assert "Cluster 0" in html_content

    def test_html_includes_cluster_info(self, temp_images, tmp_path):
        indices = np.array([0, 2])
        cluster_size = 8
        output_path = tmp_path / "test.html"

        html_path = generate_cluster_preview_html(
            cluster_id=3,
            image_paths=temp_images,
            indices=indices,
            cluster_size=cluster_size,
            output_path=output_path,
        )

        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        # Check cluster information
        assert "Cluster 3" in html_content
        assert str(cluster_size) in html_content
        assert str(len(indices)) in html_content

    def test_html_includes_images(self, temp_images, tmp_path):
        indices = np.array([0, 1, 2])
        output_path = tmp_path / "test.html"

        html_path = generate_cluster_preview_html(
            cluster_id=0,
            image_paths=temp_images,
            indices=indices,
            cluster_size=5,
            output_path=output_path,
        )

        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        # Check that image paths are included
        for idx in indices:
            assert temp_images[idx].name in html_content

        # Check for image tags
        assert "<img" in html_content
        assert 'src="' in html_content

    def test_html_with_single_image(self, temp_images, tmp_path):
        indices = np.array([0])
        output_path = tmp_path / "single.html"

        html_path = generate_cluster_preview_html(
            cluster_id=0,
            image_paths=temp_images,
            indices=indices,
            cluster_size=1,
            output_path=output_path,
        )

        assert html_path.exists()

        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        assert "Sample 1" in html_content

    def test_html_with_many_images(self, temp_images, tmp_path):
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        output_path = tmp_path / "many.html"

        html_path = generate_cluster_preview_html(
            cluster_id=0,
            image_paths=temp_images,
            indices=indices,
            cluster_size=10,
            output_path=output_path,
        )

        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        # Should include all 10 images
        assert "Sample 10" in html_content

    def test_html_encoding(self, temp_images, tmp_path):
        indices = np.array([0])
        output_path = tmp_path / "encoding.html"

        html_path = generate_cluster_preview_html(
            cluster_id=0,
            image_paths=temp_images,
            indices=indices,
            cluster_size=1,
            output_path=output_path,
        )

        # Should be able to read with utf-8 encoding
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
            assert content is not None

    def test_html_with_special_characters_in_filename(self, tmp_path):
        # Create image with special characters in name
        img_path = tmp_path / "image (1).jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        indices = np.array([0])
        output_path = tmp_path / "special.html"

        html_path = generate_cluster_preview_html(
            cluster_id=0,
            image_paths=[img_path],
            indices=indices,
            cluster_size=1,
            output_path=output_path,
        )

        assert html_path.exists()

    def test_different_cluster_ids(self, temp_images, tmp_path):
        indices = np.array([0, 1])

        for cluster_id in [0, 1, 5, 10, 100]:
            output_path = tmp_path / f"cluster_{cluster_id}.html"

            html_path = generate_cluster_preview_html(
                cluster_id=cluster_id,
                image_paths=temp_images,
                indices=indices,
                cluster_size=3,
                output_path=output_path,
            )

            with open(html_path, encoding="utf-8") as f:
                html_content = f.read()

            assert f"Cluster {cluster_id}" in html_content


class TestOpenHTMLInBrowser:
    def test_open_html_in_browser(self, tmp_path, monkeypatch):
        # Create a test HTML file
        html_path = tmp_path / "test.html"
        html_path.write_text("<html><body>Test</body></html>")

        # Mock webbrowser.open to avoid actually opening browser
        opened_url = None

        def mock_open(url):
            nonlocal opened_url
            opened_url = url
            return True

        import webbrowser

        monkeypatch.setattr(webbrowser, "open", mock_open)

        # Call the function
        open_html_in_browser(html_path)

        # Check that webbrowser.open was called with correct URL
        assert opened_url is not None
        assert html_path.as_uri() in opened_url

    def test_open_nonexistent_file(self, tmp_path, monkeypatch):
        html_path = tmp_path / "nonexistent.html"

        # Mock webbrowser to avoid errors
        def mock_open(url):
            return True

        import webbrowser

        monkeypatch.setattr(webbrowser, "open", mock_open)

        # Should not raise error even if file doesn't exist
        # (browser handles the error)
        open_html_in_browser(html_path)
