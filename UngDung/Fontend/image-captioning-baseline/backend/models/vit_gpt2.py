import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

class ViTGPT2Captioner:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_name = "nlpconnect/vit-gpt2-image-captioning"
        self.loaded = False

    def load(self):
        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # Load components
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
        
        # Performance optimization: eval mode, no gradients needed
        self.model.eval()
        self.loaded = True

    def generate(self, image: Image.Image) -> str:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")
        
        # Preprocess
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Inference mode disables gradient calculation, optimizing memory and speed
        with torch.inference_mode():
            output_ids = self.model.generate(
                pixel_values,
                max_length=16,
                num_beams=4
            )
        
        # Decode and clean output
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        caption = preds[0].strip()
        return caption

    def is_loaded(self):
        return self.loaded
