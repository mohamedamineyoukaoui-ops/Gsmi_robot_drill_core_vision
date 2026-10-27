"""
Step 1 — crack mask of the whole tray with the U-Net.

The U-Net was trained with every tray brought to one physical resolution,
10 px/cm, so a crack has the same width in pixels in every image. The tray is
resized to that resolution (its own px/cm is known), segmented in horizontal
bands of 1024 rows that overlap by 256 rows (merged with a pixel-wise max: a
crack seen in either band is kept), and the probabilities are resized back to
the original image.

Then: probability > 0.45 -> crack, followed by a 3x3 morphological close
(fills 1-pixel holes) and open (removes isolated noise pixels).

Manual corrections from the user are applied to a copy of this mask later:
drawn lines become cracks, drawn boxes erase cracks.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline import weights

WORK_PX_PER_CM = 10.0          # the resolution the U-Net was trained at
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_model = None


def _get_model():
    global _model
    if _model is None:
        import segmentation_models_pytorch as smp
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(torch.load(str(weights.path(weights.CRACKS)), map_location=device, weights_only=True))
        _model = model.to(device).eval()
    return _model


def _predict_probs(rgb: np.ndarray) -> np.ndarray:
    """Crack probability for every pixel of an RGB array whose sides are multiples of 32."""
    import torch

    model = _get_model()
    device = next(model.parameters()).device
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(tensor))[0, 0].cpu().numpy()


def segment_cracks(
    bgr_image: np.ndarray,
    px_per_cm: float | None = None,
    threshold: float = 0.45,
    chunk_h: int = 1024,
    overlap: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """(mask uint8 0/255, probabilities float32) for the whole image, at its own size.

    px_per_cm: the image's scale. Without it the image is segmented as it is.
    """
    if not weights.exists(weights.CRACKS):
        raise FileNotFoundError(f"U-Net weights not found: {weights.path(weights.CRACKS)}")

    h, w = bgr_image.shape[:2]
    factor = WORK_PX_PER_CM / px_per_cm if px_per_cm and px_per_cm > 0 else 1.0
    work = bgr_image if abs(factor - 1) < 1e-3 else cv2.resize(
        bgr_image, (max(32, round(w * factor)), max(32, round(h * factor))),
        interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)

    wh, ww = work.shape[:2]
    probs = np.zeros((wh, ww), np.float32)
    for y in range(0, wh, max(1, chunk_h - overlap)):
        y_end = min(y + chunk_h, wh)
        chunk = cv2.cvtColor(work[y:y_end], cv2.COLOR_BGR2RGB)
        ch, cw = chunk.shape[:2]
        # The U-Net downsamples 5 times (2^5 = 32): pad both sides to a multiple of 32.
        padded = np.pad(chunk, ((0, (32 - ch % 32) % 32), (0, (32 - cw % 32) % 32), (0, 0)))
        probs[y:y_end] = np.maximum(probs[y:y_end], _predict_probs(padded)[:ch, :cw])
        if y_end == wh:
            break

    if (wh, ww) != (h, w):
        probs = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (probs > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, probs


def apply_line_corrections(mask: np.ndarray, lines: list[list[tuple[int, int]]]) -> None:
    """User-drawn crack lines (image coordinates) are burned into the mask, in place."""
    for pts in lines:
        if len(pts) > 1:
            cv2.polylines(mask, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, 255, thickness=6)


def apply_box_erasures(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    """User-drawn "intact rock" boxes (x, y, w, h) erase the cracks under them, in place."""
    for x, y, bw, bh in boxes:
        cv2.rectangle(mask, (int(x), int(y)), (int(x + bw), int(y + bh)), 0, -1)
