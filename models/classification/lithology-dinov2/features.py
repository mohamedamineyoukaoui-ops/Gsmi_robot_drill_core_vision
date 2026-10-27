"""
DINOv2 feature extractor (frozen).

DINOv2 (Meta, 2023) is a Vision Transformer trained without labels on 142 M
images. Its features describe texture, colour and grain without having been
trained on any rock label, so they carry no shortcut from a previous rock
dataset. Only a small classifier on top of them is trained (train.py).
"""

from __future__ import annotations

import numpy as np
import torch

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"   # ViT-S/14, 21 M parameters, 384-d output
IMG = 224

_model = None
_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_backbone():
    global _model
    if _model is None:
        import timm
        _model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=IMG)
        _model = _model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return _model


@torch.no_grad()
def embed_bgr(patches: list[np.ndarray], batch: int = 64) -> np.ndarray:
    """BGR uint8 patches (any size) -> L2-normalised 384-d features."""
    import cv2
    m = get_backbone()
    dev = next(m.parameters()).device
    out = []
    for i in range(0, len(patches), batch):
        x = np.stack([cv2.resize(p, (IMG, IMG), interpolation=cv2.INTER_AREA)[:, :, ::-1] for p in patches[i:i + batch]])
        t = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float() / 255.0
        f = m(((t - _mean) / _std).to(dev))
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)
