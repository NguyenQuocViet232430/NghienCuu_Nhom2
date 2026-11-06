"""
caption_generator.py
====================
Abstract interface and concrete implementations for the Caption Generator.

Implementations:
  - MockCaptionGenerator: Placeholder used when no trained model is available.
  - ViTGPT2CaptionGenerator: Adapter for `nlpconnect/vit-gpt2-image-captioning`,
    a real pre-trained model that generates captions from PIL images.
    Requires: pip install torch transformers

Architecture note:
  In Phase 1, CaptionGenerator receives a PIL Image directly (since ViTGPT2 has
  its own internal image processor). In Phase 2, when KnowledgeFusion is added,
  the interface will be extended to also accept fused embeddings.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class CaptionGenerator(ABC):
    """Abstract base class for all caption generators.

    Implementations may accept either:
      - A raw PIL image (for models with internal preprocessing, e.g. ViT-GPT2)
      - An embedding vector (for models that consume pre-computed features)

    In Phase 2, the interface will be extended with a `generate_with_knowledge`
    method that also accepts fused knowledge vectors.
    """

    @abstractmethod
    def generate(self, image: Image.Image) -> str:
        """Generate a natural-language caption.

        Args:
            image: Preprocessed PIL.Image.Image.

        Returns:
            A string caption.
        """
        raise NotImplementedError


class MockCaptionGenerator(CaptionGenerator):
    """Placeholder generator used when no trained model is available.

    Returns a clearly-labelled message so the UI never misleads the user
    into thinking a real model produced the output.
    """

    def generate(self, image: Image.Image) -> str:
        logger.warning(
            "MockCaptionGenerator is active. "
            "No trained model is loaded. Install transformers+torch to use ViTGPT2CaptionGenerator."
        )
        return (
            "[Baseline — Model not loaded] "
            "Caption generator is not configured. "
            "Install 'torch' and 'transformers', then restart the server to enable real inference."
        )


class ViTGPT2CaptionGenerator(CaptionGenerator):
    """Adapter for the pre-trained `nlpconnect/vit-gpt2-image-captioning` model.

    This is a real encoder-decoder model (ViT image encoder + GPT-2 text decoder)
    trained on MS-COCO. It produces genuine captions without any fine-tuning.

    Installation:
        pip install torch transformers

    Usage:
        gen = ViTGPT2CaptionGenerator(device="cpu")
        caption = gen.generate(pil_image)
    """

    MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

    def __init__(self, device: str = "cpu"):
        """Load the ViT-GPT2 model and processor from HuggingFace Hub.

        Args:
            device: 'cpu' or 'cuda'. Defaults to 'cpu'.

        Raises:
            ImportError: If torch or transformers are not installed.
            OSError: If the model cannot be downloaded.
        """
        try:
            from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "torch and transformers are required for ViTGPT2CaptionGenerator. "
                "Run: pip install torch transformers"
            ) from e

        import torch

        logger.info(f"Loading ViTGPT2 model '{self.MODEL_NAME}' on device '{device}'...")
        self.device = torch.device(device)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.processor = ViTImageProcessor.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        logger.info("ViTGPT2 model loaded successfully.")

    def generate(self, image: Image.Image) -> str:
        """Generate a caption for the given PIL image.

        Args:
            image: A PIL.Image.Image (RGB).

        Returns:
            Generated caption string.
        """
        import torch

        # The ViTImageProcessor handles its own internal resizing & normalization
        pixel_values = self.processor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                early_stopping=True,
            )

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
