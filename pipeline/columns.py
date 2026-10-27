"""
Step 2 — find the core columns in the tray.

A core tray holds 3 to 5 parallel columns of rock. One detector
(models/detection/columns_uni, YOLO11s) boxes each column from where the core
starts to where it ends, so the box gives both the x-range of the column and
the y-range of the core, without the label zones above and below it.

Fallback without AI: column detector missing or no box -> dark-valley
projection (dividers are dark) and a brightness test for the core ends.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline import weights

IMGSZ = 1024                     # the detector was trained at 1024 px
_column_model = None


def _get_column_model():
    global _column_model
    if _column_model is None:
        from ultralytics import YOLO
        if not weights.exists(weights.COLUMNS):
            raise FileNotFoundError(f"Column model not found: {weights.path(weights.COLUMNS)}")
        _column_model = YOLO(str(weights.path(weights.COLUMNS)), task="detect")
    return _column_model


def detect_columns(image: np.ndarray, conf: float = 0.6) -> list[dict]:
    """Columns of a BGR tray image, left to right.

    Each column: {image, x_start, x_end, y_start, y_end, column_id, confidence}.
    """
    try:
        return _detect_with_yolo(image, conf)
    except Exception:
        return _detect_by_projection(image, num_columns=5)


def _detect_with_yolo(image: np.ndarray, conf: float) -> list[dict]:
    h, w = image.shape[:2]
    boxes = _get_column_model()(image, verbose=False, conf=conf, imgsz=IMGSZ)[0].boxes
    if len(boxes) == 0:
        raise RuntimeError("No columns detected")

    detections = sorted(
        (tuple(int(v) for v in b.xyxy[0].cpu().numpy()) + (float(b.conf[0]),) for b in boxes),
        key=lambda d: d[0],
    )
    columns = []
    for x1, y1, x2, y2, c in detections:
        x_start, x_end, y_start, y_end = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
        if x_end <= x_start or y_end <= y_start:
            continue
        columns.append(dict(image=image[y_start:y_end, x_start:x_end], x_start=x_start, x_end=x_end,
                            y_start=y_start, y_end=y_end, column_id=len(columns) + 1, confidence=c))
    return columns


def _detect_by_projection(image: np.ndarray, num_columns: int) -> list[dict]:
    """No model: the dividers between columns are the darkest vertical lines."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    projection = gray.mean(axis=0)
    bright = np.where(projection > projection.max() * 0.25)[0]
    left, right = (int(bright[0]), int(bright[-1])) if len(bright) > 10 else (int(w * 0.05), int(w * 0.95))
    tray_w = right - left

    profile = cv2.GaussianBlur(gray[:, left:right], (5, 51), 0).mean(axis=0)
    k = max(5, int(tray_w * 0.02)) | 1
    profile = cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (k, 1), 0).flatten()

    # Local minima, strongest first, at least 75 % of the expected spacing apart.
    margin = int(tray_w * 0.05)
    window = max(5, int(tray_w * 0.02))
    region = profile[margin:len(profile) - margin]
    valleys = sorted(
        ((-region[x], x + margin) for x in range(window, len(region) - window)
         if region[x] == region[x - window:x + window + 1].min()),
        reverse=True,
    )
    min_spacing = tray_w / num_columns * 0.75
    dividers: list[int] = []
    for _, x in valleys:
        if all(abs(x - d) >= min_spacing for d in dividers):
            dividers.append(x)
        if len(dividers) == num_columns - 1:
            break
    if len(dividers) < num_columns - 1:
        dividers = [int(i * tray_w / num_columns) for i in range(1, num_columns)]

    # Refine each divider to the darkest column of the raw projection nearby.
    refined = []
    for d in sorted(dividers):
        lo, hi = max(0, left + d - int(tray_w * 0.04)), min(w, left + d + int(tray_w * 0.04))
        refined.append(lo + int(np.argmin(projection[lo:hi])) if hi > lo else left + d)

    edges = [left] + refined + [right]
    columns = []
    for x_start, x_end in zip(edges[:-1], edges[1:]):
        if x_end <= x_start:
            continue
        rows = np.where(gray[:, x_start:x_end].mean(axis=1) > gray[:, x_start:x_end].mean() * 0.3)[0]
        y_start, y_end = (int(rows[0]), int(rows[-1]) + 1) if len(rows) > 10 else (0, h)
        columns.append(dict(image=image[y_start:y_end, x_start:x_end], x_start=int(x_start), x_end=int(x_end),
                            y_start=y_start, y_end=y_end, column_id=len(columns) + 1, confidence=None))
    return columns
