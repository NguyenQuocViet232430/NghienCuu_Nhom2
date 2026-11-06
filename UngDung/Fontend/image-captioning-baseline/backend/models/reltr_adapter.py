"""Optional local RelTR adapter. Settings match the group's Lv3/rel-tr.py.

Uses models/reltr by default; RELTR_REPO and RELTR_CHECKPOINT override it.
No detector downloads at startup.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import torch

CLASSES = ('N/A airplane animal arm bag banana basket beach bear bed bench bike bird board boat book boot bottle '
           'bowl box boy branch building bus cabinet cap car cat chair child clock coat counter cow cup curtain '
           'desk dog door drawer ear elephant engine eye face fence finger flag flower food fork fruit giraffe '
           'girl glass glove guy hair hand handle hat head helmet hill horse house jacket jean kid kite lady lamp '
           'laptop leaf leg letter light logo man men motorcycle mountain mouth neck nose number orange pant '
           'paper paw people person phone pillow pizza plane plant plate player pole post pot racket railing rock '
           'roof room screen seat sheep shelf shirt shoe short sidewalk sign sink skateboard ski skier sneaker '
           'snow sock stand street surfboard table tail tie tile tire toilet towel tower track train tree truck '
           'trunk umbrella vase vegetable vehicle wave wheel window windshield wing wire woman zebra').split()


class RelTRDetector:
    @classmethod
    def from_environment(cls, device):
        root = Path(__file__).resolve().parents[4] / "models" / "reltr"
        repo = os.environ.get("RELTR_REPO") or str(root / "RelTR")
        weights = os.environ.get("RELTR_CHECKPOINT") or str(root / "checkpoint0149.pth")
        if not Path(repo).is_dir() or not Path(weights).is_file():
            raise FileNotFoundError("run scripts/setup-reltr.ps1 or set RELTR_REPO and RELTR_CHECKPOINT")
        return cls(Path(repo), Path(weights), device)

    def __init__(self, repo, checkpoint, device):
        repo = repo.resolve()
        if not (repo / "models" / "reltr.py").is_file() or not checkpoint.is_file():
            raise FileNotFoundError("RelTR repository or checkpoint is missing")
        for name in ("models", "util"):
            module = sys.modules.get(name)
            if module is not None and repo not in Path(module.__file__).resolve().parents:
                raise RuntimeError(f"{name} already imported from another repository; restart the server")
        sys.path.insert(0, str(repo))
        try:
            from models import backbone as bb
            from models.backbone import Backbone, Joiner
            from models.position_encoding import PositionEmbeddingSine
            from models.transformer import Transformer
            from models.reltr import RelTR
        finally:
            sys.path.remove(str(repo))
        import torchvision.transforms as transforms
        original_is_main = bb.is_main_process
        bb.is_main_process = lambda: False
        try:
            backbone = Joiner(Backbone("resnet50", False, False, False), PositionEmbeddingSine(128, normalize=True))
        finally:
            bb.is_main_process = original_is_main
        backbone.num_channels = 2048
        transformer = Transformer(d_model=256, dropout=0.1, nhead=8, dim_feedforward=2048,
                                  num_encoder_layers=6, num_decoder_layers=6,
                                  normalize_before=False, return_intermediate_dec=True)
        model = RelTR(backbone, transformer, num_classes=151, num_rel_classes=51,
                      num_entities=100, num_triplets=200)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved.get("model", saved), strict=True)
        self.model = model.requires_grad_(False).to(device).eval()
        self.device = device
        self.transform = transforms.Compose([transforms.Resize(800), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @torch.inference_mode()
    def detect(self, image):
        outputs = self.model(self.transform(image.convert("RGB")).unsqueeze(0).to(self.device))
        relation = outputs["rel_logits"].softmax(-1)[0, :, :-1]
        subject = outputs["sub_logits"].softmax(-1)[0, :, :-1]
        obj = outputs["obj_logits"].softmax(-1)[0, :, :-1]
        if subject.shape != (200, 151) or obj.shape != (200, 151) or relation.shape != (200, 51):
            raise ValueError("Unexpected RelTR Visual Genome output shape")
        if not all(torch.isfinite(x).all() for x in (subject, obj, relation)):
            raise ValueError("Non-finite RelTR output")
        rp, ri = relation.max(-1)
        sp, si = subject.max(-1)
        op, oi = obj.max(-1)
        positions = torch.where((rp > 0.25) & (sp > 0.25) & (op > 0.25))[0]
        positions = positions[torch.argsort(-(rp[positions] * sp[positions] * op[positions]))]
        unique = {}
        for position in positions.tolist():
            key = (int(si[position]), int(ri[position]), int(oi[position]))
            unique.setdefault(key, float((sp[position] + rp[position] + op[position]) / 3))
        top = sorted(unique, key=unique.get, reverse=True)[:10]
        return sorted({CLASSES[idx] for s, _, o in top for idx in (s, o) if CLASSES[idx] != "N/A"})
