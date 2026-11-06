"""
image_encoder.py
================
Abstract interface and concrete implementations for the Image Encoder component.

Current implementations:
  - MockImageEncoder: returns a zero-vector, used when no real model is available.

Future adapters (to be implemented in Phase 2+):
  - CLIPEncoder: Uses OpenAI CLIP to produce joint image-text embeddings.
  - ViTEncoder: Uses a standalone Vision Transformer for image-only embeddings.

The encoder output (image_embedding) feeds into:
  1. CaptionGenerator   (Phase 1 — current)
  2. KnowledgeRetriever (Phase 2 — future)
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageEncoder(ABC):
    """Abstract base class for all image encoders.

    Any encoder must implement `encode()`, which accepts a PIL Image
    and returns an embedding vector (numpy array).
    """

    @abstractmethod
    def encode(self, image: Image.Image) -> Any:
        """Encode a PIL image into an embedding vector.

        Args:
            image: A PIL.Image.Image object (should already be preprocessed).

        Returns:
            Embedding representation of the image (e.g. numpy ndarray).
        """
        raise NotImplementedError


class MockImageEncoder(ImageEncoder):
    """Mock encoder that returns a zero-vector.

    Used as a placeholder when no real encoder model is available.
    Downstream components (CaptionGenerator, KnowledgeRetriever) can
    still be wired up and tested against this dummy embedding.

    Output shape: (embedding_dim,) numpy float32 array of zeros.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        logger.warning(
            "MockImageEncoder is active. "
            "Replace with CLIPEncoder or ViTEncoder for real embeddings."
        )

    def encode(self, image: Image.Image) -> np.ndarray:
        """Return a zero-vector as a mock embedding."""
        logger.debug("MockImageEncoder.encode() called — returning zero vector.")
        return np.zeros(self.embedding_dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# Future adapters (not yet implemented)
# ---------------------------------------------------------------------------

# class CLIPEncoder(ImageEncoder):
#     """Uses OpenAI CLIP to produce joint image-text embeddings.
#     Install: pip install openai-clip
#     """
#     def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
#         import clip
#         self.model, self.preprocess = clip.load(model_name, device=device)
#         self.device = device
#
#     def encode(self, image: Image.Image) -> np.ndarray:
#         import torch
#         img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             features = self.model.encode_image(img_tensor)
#         return features.cpu().numpy().squeeze()


# class ViTEncoder(ImageEncoder):
#     """Uses a standalone Vision Transformer (google/vit-base-patch16-224).
#     Install: pip install transformers torch
#     """
#     def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = "cpu"):
#         from transformers import ViTModel, ViTFeatureExtractor
#         self.extractor = ViTFeatureExtractor.from_pretrained(model_name)
#         self.model = ViTModel.from_pretrained(model_name).to(device)
#         self.device = device
#
#     def encode(self, image: Image.Image) -> np.ndarray:
#         import torch
#         inputs = self.extractor(images=image, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return outputs.pooler_output.cpu().numpy().squeeze()
