"""
Rock type from the image: DINOv2 features + a trained linear head.

The model was trained on square tiles of rock surface, so it must be fed
square tiles cut inside the detected columns, never a whole tray photo.

    tile (BGR, any size) -> resize 224x224, RGB, ImageNet normalisation
                         -> DINOv2 ViT-S/14 (frozen)  -> 384-d feature
                         -> L2 normalisation
                         -> logistic-regression head  -> 6 class probabilities

Tiles mostly showing "no rock" (tray wall, label, empty trough) are dropped,
and the remaining tile probabilities are averaged into one vector.

Two corrections, both measured on the test set:
  - tiles are cut from the core band only: with a fixed 12 % margin the
    tray walls stayed in the tiles and 86 % of core tiles came out "no rock"
    (24 % once limited to the core);
  - a colour cast common to all tiles of an image (e.g. multispectral VNIR
    trays) is shifted back to the training colour range: it cost 3 points of
    accuracy, and the shift recovers all of it. Brightness is not touched:
    it is a real rock cue.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline import weights

CLASSES = ["limestone", "dolomitic_limestone", "marl", "flint", "out_of_series", "no_rock"]
IMAGE_SIZE = 224
BACKBONE = "vit_small_patch14_dinov2.lvd142m"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Median LAB a*, b* of the training rock images, and the tolerated spread.
TRAIN_A, TRAIN_B = 1.4, 3.7
CAST_TOL_A, CAST_TOL_B = 4.0, 6.0

_cache: dict = {}


def available() -> bool:
    d = weights.path(weights.LITHOLOGY_DIR)
    return (d / "dinov2_vits14_224.pt").exists() and (d / "dinov2_head_A.npz").exists()


def _backbone():
    if "backbone" not in _cache:
        import timm
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bb = timm.create_model(BACKBONE, pretrained=False, num_classes=0, img_size=IMAGE_SIZE)
        bb.load_state_dict(torch.load(weights.path(weights.LITHOLOGY_DIR) / "dinov2_vits14_224.pt", map_location="cpu"))
        _cache["backbone"] = bb.eval().to(device)
    return _cache["backbone"]


def _head() -> tuple[np.ndarray, np.ndarray]:
    if "head" not in _cache:
        h = np.load(weights.path(weights.LITHOLOGY_DIR) / "dinov2_head_A.npz")
        _cache["head"] = (h["coef"], h["intercept"])          # (6, 384), (6,)
    return _cache["head"]


def match_training_colour(tiles: list[np.ndarray]) -> list[np.ndarray]:
    """Remove a colour cast common to all tiles when it lies outside the training range."""
    if not tiles:
        return tiles
    labs = [cv2.cvtColor(t, cv2.COLOR_BGR2LAB).astype(np.float32) for t in tiles]
    a = float(np.mean([lab[..., 1].mean() for lab in labs])) - 128.0
    b = float(np.mean([lab[..., 2].mean() for lab in labs])) - 128.0
    if abs(a - TRAIN_A) <= CAST_TOL_A and abs(b - TRAIN_B) <= CAST_TOL_B:
        return tiles
    out = []
    for lab in labs:
        lab[..., 1] -= a - TRAIN_A
        lab[..., 2] -= b - TRAIN_B
        out.append(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))
    return out


def predict_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    """Probabilities [N, 6] in CLASSES order."""
    import torch

    if not tiles:
        return np.zeros((0, len(CLASSES)))
    tiles = match_training_colour(tiles)
    bb = _backbone()
    x = np.stack([cv2.resize(t, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1] for t in tiles])
    x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    with torch.no_grad():
        f = bb(torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(next(bb.parameters()).device))
        f = torch.nn.functional.normalize(f, dim=1).cpu().numpy()

    coef, intercept = _head()
    logits = f @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)           # numerically stable softmax
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def aggregate(probs: np.ndarray, no_rock_cut: float = 0.5) -> dict[str, float]:
    """Average of the rock tiles (all tiles if every one is 'no rock')."""
    if probs.size == 0:
        return {}
    rock = probs[:, CLASSES.index("no_rock")] < no_rock_cut
    m = (probs[rock] if rock.any() else probs).mean(axis=0)
    m = m / max(m.sum(), 1e-9)
    return {c: float(m[i]) for i, c in enumerate(CLASSES)}


def core_x_range(column_bgr: np.ndarray) -> tuple[int, int]:
    """x-range of the core inside a column crop, shrunk by 4 % on each side."""
    from pipeline.pieces import core_band
    xl, xr = core_band(column_bgr, column_bgr.shape[1])
    pad = int(0.04 * (xr - xl))
    return xl + pad, xr - pad


def tiles_in_columns(bgr: np.ndarray, columns: list[dict], max_tiles: int = 96) -> list[np.ndarray]:
    """Square tiles inside the core band of each column, half-overlapping along the column."""
    tiles = []
    for c in columns:
        if c["x_end"] - c["x_start"] < 8 or c["y_end"] - c["y_start"] < 8:
            continue
        xl, xr = core_x_range(bgr[c["y_start"]:c["y_end"], c["x_start"]:c["x_end"]])
        a, b = c["x_start"] + xl, c["x_start"] + xr
        s = b - a
        if s < 16:
            continue
        for y in range(c["y_start"], max(c["y_start"] + 1, c["y_end"] - s + 1), max(1, s // 2)):
            t = bgr[y:y + s, a:b]
            if t.shape[:2] == (s, s):
                tiles.append(t)
    if len(tiles) > max_tiles:
        tiles = [tiles[i] for i in np.linspace(0, len(tiles) - 1, max_tiles).round().astype(int)]
    return tiles


def tiles_in_grid(bgr: np.ndarray, n: int = 4) -> list[np.ndarray]:
    """When no column was found: a regular grid of square tiles."""
    h, w = bgr.shape[:2]
    s = max(16, min(h, w) // n)
    return [bgr[y:y + s, x:x + s] for y in range(0, h - s + 1, s) for x in range(0, w - s + 1, s)]


def classify(bgr: np.ndarray, columns: list[dict], hires_bgr: np.ndarray | None = None) -> dict[str, float]:
    """Rock-type probabilities of a tray. hires_bgr: the same layout at higher
    resolution (stitched scans: tiles cut from the 140 px/column RQD layout
    were too small and cost ~10 points of accuracy); column boxes are scaled onto it."""
    if hires_bgr is not None:
        sx, sy = hires_bgr.shape[1] / bgr.shape[1], hires_bgr.shape[0] / bgr.shape[0]
        if abs(sx - sy) <= 0.02 * sx:                 # same layout, just bigger
            bgr = hires_bgr
            columns = [{**c, "x_start": int(c["x_start"] * sx), "x_end": int(c["x_end"] * sx),
                        "y_start": int(c["y_start"] * sy), "y_end": int(c["y_end"] * sy)} for c in columns]
    tiles = tiles_in_columns(bgr, columns) if columns else tiles_in_grid(bgr)
    return aggregate(predict_tiles(tiles))
