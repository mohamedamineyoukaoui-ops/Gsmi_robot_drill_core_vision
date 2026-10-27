"""
Step 4 — label each stretch of a column as intact or crushed.

The zone classifier looks at one 128x128 window at a time (intact / rubble /
sand / empty). A window slides down the column with 50 % overlap; every row
takes the class of the nearest window centre; runs shorter than 5 cm are
absorbed by their neighbour. Each zone keeps its class and the mean class
probabilities of its windows; its binary "type" is intact or crushed.

Correction with the crack mask: a clean linear crack can only exist in solid
rock. If a crack crosses a crushed zone AND continues beyond it, that zone is
relabelled intact.

Role in RQD (pieces.py):
  - rows the classifier sees as rock (intact / rubble / sand) help define the
    core span, i.e. the RQD denominator;
  - a confident rubble, sand or empty zone vetoes the pieces lying in it;
  - without crack mask or detector, RQD comes from the intact zones alone.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline import weights

WINDOW = 128          # the classifier was trained on 128x128 patches

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        if not weights.exists(weights.ZONES):
            raise FileNotFoundError(f"Zone model not found: {weights.path(weights.ZONES)}")
        _model = YOLO(str(weights.path(weights.ZONES)), task="classify")
    return _model


def classify_zones(column_bgr: np.ndarray, px_per_cm: float, min_zone_cm: float = 5.0,
                   overlap: float = 0.5) -> dict:
    """{'segments': [{type, zone_class, probs, y_start, y_end, length_px}], 'intact_ratio': float}"""
    h, w = column_bgr.shape[:2]
    model = _get_model()

    # Resize so the column is exactly one window wide (as in training).
    scale = WINDOW / w
    scaled_h = int(h * scale)
    if scaled_h == 0:
        return {"segments": [], "intact_ratio": 1.0}
    col = cv2.resize(column_bgr, (WINDOW, scaled_h))

    centers, classes, probs = [], [], []
    for y in range(0, scaled_h - WINDOW // 2, max(1, int(WINDOW * (1 - overlap)))):
        chunk = col[y:min(y + WINDOW, scaled_h)]
        if chunk.shape[0] < WINDOW // 2:
            continue
        if chunk.shape[0] < WINDOW:                      # pad the last window with black
            padded = np.zeros((WINDOW, WINDOW, 3), np.uint8)
            padded[:chunk.shape[0]] = chunk
            chunk = padded
        r = model(chunk, verbose=False)[0]
        centers.append(int((y + WINDOW // 2) / scale))
        classes.append(r.names[r.probs.top1])
        probs.append({r.names[i]: float(p) for i, p in enumerate(r.probs.data.cpu().numpy().ravel())})

    if not centers:
        return {"segments": [], "intact_ratio": 1.0}

    # Every row takes the class of its nearest window.
    centers_arr = np.array(centers)
    nearest = np.abs(np.arange(h)[:, None] - centers_arr[None, :]).argmin(axis=1)
    names = sorted(set(classes) | {"intact"})
    codes = np.array([names.index(classes[k]) for k in nearest])

    # Minimum zone length in cm, at this image's scale (at most half the column).
    min_frac = min(0.5, min_zone_cm * px_per_cm / h)
    codes = _absorb_short_runs(codes, max(10, int(h * min_frac)))

    segments, i = [], 0
    while i < h:
        j = i
        while j < h and codes[j] == codes[i]:
            j += 1
        zone_class = names[codes[i]]
        inside = [k for k, c in enumerate(centers) if i <= c < j] or [int(nearest[(i + j) // 2])]
        mean_probs = {c: round(float(np.mean([probs[k].get(c, 0.0) for k in inside])), 3) for c in probs[0]}
        segments.append({"type": "intact" if zone_class == "intact" else "crushed", "zone_class": zone_class,
                         "probs": mean_probs, "y_start": i, "y_end": j, "length_px": j - i})
        i = j

    return {"segments": segments, "intact_ratio": float((codes == names.index("intact")).mean())}


def _absorb_short_runs(codes: np.ndarray, min_len: int) -> np.ndarray:
    """A run shorter than min_len takes the label of the run before it (or after, at the top)."""
    out = codes.copy()
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_len:
            if i > 0:
                out[i:j] = out[i - 1]
            elif j < len(out):
                out[i:j] = out[j]
        i = j
    return out


# ── crack-continuity correction ─────────────────────────────────────────────

def extract_crack_blobs(mask: np.ndarray, min_area: int = 20, min_aspect: float = 0.25) -> list[dict]:
    """Connected crack regions, kept only if wide enough to be a crack across the core."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or w < 5 or (h > 0 and w / h < min_aspect):
            continue
        blobs.append({"y_top": y, "y_bottom": y + h - 1, "x_left": x, "x_right": x + w - 1})
    return sorted(blobs, key=lambda b: b["y_top"])


def blobs_in_column(blobs: list[dict], col: dict, pad_px: int = 10, merge_px: int = 10) -> list[dict]:
    """Blobs overlapping the column, in column coordinates, with vertically close ones merged."""
    x1, x2 = col["x_start"] - pad_px, col["x_end"] + pad_px
    y1, y2 = col["y_start"], col["y_end"]
    inside = [
        {"y_top": max(0, b["y_top"] - y1), "y_bottom": max(0, b["y_bottom"] - y1)}
        for b in blobs
        if b["x_right"] >= x1 and b["x_left"] <= x2 and b["y_bottom"] >= y1 and b["y_top"] <= y2
    ]
    merged: list[dict] = []
    for b in inside:
        if merged and b["y_top"] <= merged[-1]["y_bottom"] + merge_px:
            merged[-1]["y_bottom"] = max(merged[-1]["y_bottom"], b["y_bottom"])
        else:
            merged.append(dict(b))
    return merged


def reclassify_with_cracks(segments: list[dict], blobs: list[dict]) -> list[dict]:
    """A crushed zone crossed by a crack that continues outside it becomes intact."""
    zones = [dict(z) for z in segments]
    for z in zones:
        if z["type"] != "crushed":
            continue
        for b in blobs:
            overlaps = min(b["y_bottom"], z["y_end"]) - max(b["y_top"], z["y_start"]) > 0
            extends = b["y_top"] < z["y_start"] - 2 or b["y_bottom"] > z["y_end"] + 2
            if overlaps and extends:
                z["type"] = "intact"
                if z.get("zone_class") in ("rubble", "sand"):
                    z["zone_class"], z["reclassified"] = "intact", True
                break
    return zones
