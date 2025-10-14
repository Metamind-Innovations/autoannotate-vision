from pathlib import Path
from typing import List, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


class ImageLoader:

    def __init__(self, input_dir: Path, recursive: bool = False):
        self.input_dir = Path(input_dir)
        self.recursive = recursive

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")

    def load_image_paths(self) -> List[Path]:
        if self.recursive:
            image_paths = [
                p
                for p in self.input_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
            ]
        else:
            image_paths = [
                p
                for p in self.input_dir.glob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
            ]

        if not image_paths:
            raise ValueError(f"No valid images found in {self.input_dir}")

        logger.info(f"Found {len(image_paths)} images")
        return sorted(image_paths)

    def load_images(self, max_size: Tuple[int, int] = None) -> Tuple[List[Image.Image], List[Path]]:
        image_paths = self.load_image_paths()
        images = []
        valid_paths = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")

                if max_size:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                images.append(img)
                valid_paths.append(img_path)

            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue

        if not images:
            raise ValueError("No valid images could be loaded")

        logger.info(f"Successfully loaded {len(images)} images")
        return images, valid_paths

    @staticmethod
    def validate_image(image_path: Path) -> bool:
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
