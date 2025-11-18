import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoProcessor
from PIL import Image
from typing import List, Literal, Optional
import numpy as np
from tqdm import tqdm


class EmbeddingExtractor:

    def __init__(
        self,
        model_name: Literal["clip", "dinov2", "dinov2-large", "siglip2"] = "dinov2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._load_model()

    def _load_model(self):
        if self.model_name == "clip":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14", use_fast=True
            )
        elif self.model_name == "dinov2":
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base", use_fast=True
            )
        elif self.model_name == "dinov2-large":
            self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-large", use_fast=True
            )
        elif self.model_name == "siglip2":
            self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(
                self.device
            )
            self.processor = AutoProcessor.from_pretrained(
                "google/siglip2-base-patch16-224", use_fast=True
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.model.eval()

    def extract_single(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            if self.model_name in ["clip", "siglip2"]:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                embedding = self.model.get_image_features(**inputs)
            else:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0]

            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        embeddings = []

        for i in tqdm(range(0, len(images), self.batch_size), desc="Extracting embeddings"):
            batch = images[i : i + self.batch_size]

            with torch.no_grad():
                if self.model_name in ["clip", "siglip2"]:
                    inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(
                        self.device
                    )
                    batch_embeddings = self.model.get_image_features(**inputs)
                else:
                    inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0]

                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

        result: np.ndarray = np.vstack(embeddings)
        return result

    def __call__(self, images: List[Image.Image]) -> np.ndarray:
        return self.extract_batch(images)
