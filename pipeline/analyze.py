"""
The full pipeline: one tray photo in, RQD + rock type out.

It is split in two stages so that a user can correct the result without
running any neural network again:

  model_stage(image)  runs every model ONCE and returns a cached "state"
      1. crack mask      U-Net on the whole image, at 10 px/cm        (cracks.py)
      2. columns         YOLO column boxes, core start to end         (columns.py)
      3. piece boxes     YOLO11, only for method "piece_detect"       (pieces.py)
      4. zones           YOLO classifier per column, optional          (zones.py)
      5. rock type       DINOv2 on tiles inside the columns, optional  (lithology_image.py)

  rqd_stage(state, corrections)  pure geometry, no model, milliseconds
      - merge the user's corrections into a copy of the cached mask
      - per column: core span, pieces (contour / connected_components / piece_detect),
        user boxes, zone vetoes
      - merge pieces cut by a column boundary
      - RQD per column and for the tray, fracture frequency, overlay

analyze() = model_stage + rqd_stage; recompute() = rqd_stage only.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline import columns as columns_step
from pipeline import cracks, overlay, pieces as pieces_step, rqd, zones as zones_step

METHODS = ("contour", "connected_components", "piece_detect")
TRAINED_COLUMN_WIDTH_PX = (50, 260)    # column widths in the 201 training tray photos


def analyze(
    rgb_image: np.ndarray,
    pixels_per_cm: float,
    method: str = "contour",
    run_zones: bool = False,
    run_lithology: bool = False,
    xrf_elements: dict[str, float] | None = None,
    manual_lines: list[list[tuple[int, int]]] | None = None,
    manual_boxes: list[tuple[int, int, int, int]] | None = None,
    depth_from_m: float = 0.0,
    depth_to_m: float = 0.0,
    lithology_rgb: np.ndarray | None = None,
    crack_threshold: float = 0.45,
    column_conf: float = 0.60,
    close_gap_cm: float = pieces_step.CLOSE_GAP_CM,
    min_zone_cm: float = 5.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (result, state). Keep `state` to call recompute() after corrections.

    depth_from_m / depth_to_m: drilled run written on the tray; when given,
    RQD uses the run length as denominator.
    lithology_rgb: optional higher-resolution copy of the same image, used
    only to cut the lithology tiles.
    """
    state = model_stage(rgb_image, pixels_per_cm, method, run_zones, run_lithology, lithology_rgb,
                        crack_threshold, column_conf, min_zone_cm)
    state["run_length_cm"] = rqd.run_length_cm(depth_from_m, depth_to_m)
    if run_lithology and state["lithology_image"]:
        from pipeline.lithology_fusion import decide
        state["lithology_decision"] = decide(state["lithology_image"], xrf_elements)
    return rqd_stage(state, manual_lines, manual_boxes, method, close_gap_cm), state


def recompute(
    state: dict[str, Any],
    manual_lines: list[list[tuple[int, int]]] | None = None,
    manual_boxes: list[tuple[int, int, int, int]] | None = None,
    method: str | None = None,
    close_gap_cm: float = pieces_step.CLOSE_GAP_CM,
) -> dict[str, Any]:
    """Apply new corrections to a previous analysis. No model is run."""
    return rqd_stage(state, manual_lines, manual_boxes, method or state["method"], close_gap_cm)


# ── stage 1: models ─────────────────────────────────────────────────────────

def model_stage(rgb_image, pixels_per_cm, method="contour", run_zones=False, run_lithology=False,
                lithology_rgb=None, crack_threshold=0.45, column_conf=0.60, min_zone_cm=5.0) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    rgb = validate_rgb(rgb_image)
    px_per_cm = validate_scale(pixels_per_cm)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    warnings: list[str] = []

    mask = None
    if method != "piece_detect":
        try:
            mask, _ = cracks.segment_cracks(bgr, px_per_cm, threshold=crack_threshold)
        except Exception as exc:
            warnings.append(f"Crack segmentation failed: {exc}")

    detected = columns_step.detect_columns(bgr, conf=column_conf)
    if not detected:
        warnings.append("No columns detected")

    piece_boxes = None
    if method == "piece_detect":
        if pieces_step.piece_detector_available():
            piece_boxes = pieces_step.detect_pieces(bgr)
            if not piece_boxes:
                warnings.append("Piece detector found no pieces")
        else:
            warnings.append("Piece detector not found")

    # The models only saw columns 50-260 px wide: far outside, every model sees the core at an unknown zoom.
    widths = [c["x_end"] - c["x_start"] for c in detected if c["x_end"] > c["x_start"]]
    if widths and not TRAINED_COLUMN_WIDTH_PX[0] <= np.median(widths) <= TRAINED_COLUMN_WIDTH_PX[1]:
        warnings.append(f"Columns are {np.median(widths):.0f} px wide, outside the training range "
                        f"{TRAINED_COLUMN_WIDTH_PX}: resize the image for reliable results")

    blobs = zones_step.extract_crack_blobs(mask) if mask is not None else []
    cols = []
    for c in detected:
        height_px = c["y_end"] - c["y_start"]
        col = {k: c[k] for k in ("column_id", "x_start", "x_end", "y_start", "y_end", "confidence")}
        col.update(height_px=height_px, height_cm=height_px / px_per_cm, intact_ratio=None, zones=[])
        if run_zones and c["image"].size:
            try:
                z = zones_step.classify_zones(c["image"], px_per_cm, min_zone_cm)
                col["zones"] = zones_step.reclassify_with_cracks(z["segments"], zones_step.blobs_in_column(blobs, c))
                col["intact_ratio"] = z["intact_ratio"]
            except Exception as exc:
                warnings.append(f"Column {c['column_id']}: zone classification failed: {exc}")
        cols.append(col)

    lithology_image = None
    if run_lithology:
        from pipeline import lithology_image as litho
        if litho.available():
            hires = None if lithology_rgb is None else cv2.cvtColor(validate_rgb(lithology_rgb), cv2.COLOR_RGB2BGR)
            lithology_image = litho.classify(bgr, detected, hires)
        else:
            warnings.append("Lithology weights not found")

    return {"bgr": bgr, "px_per_cm": px_per_cm, "method": method, "mask": mask, "piece_boxes": piece_boxes,
            "columns": cols, "lithology_image": lithology_image, "lithology_decision": None,
            "run_length_cm": None, "warnings": warnings}


# ── stage 2: geometry ───────────────────────────────────────────────────────

def rqd_stage(state, manual_lines=None, manual_boxes=None, method="contour",
              close_gap_cm=pieces_step.CLOSE_GAP_CM) -> dict[str, Any]:
    px = state["px_per_cm"]
    bgr = state["bgr"]
    warnings = list(state["warnings"])
    boxes = list(manual_boxes or [])

    mask = state["mask"]
    if mask is not None:
        mask = mask.copy()
        cracks.apply_line_corrections(mask, list(manual_lines or []))
        cracks.apply_box_erasures(mask, boxes)
    if method == "piece_detect" and state["piece_boxes"] is None:
        method = "connected_components" if mask is not None else "zones"
        warnings.append(f"Piece detection unavailable: using {method}")
    elif method != "piece_detect" and mask is None:
        method = "zones"

    annotated = bgr.copy()
    if mask is not None and method != "piece_detect":
        overlay.tint_mask(annotated, mask)

    results = []
    for col in state["columns"]:
        out = {k: v for k, v in col.items() if k != "zones"}
        out["segments"] = [dict(s, length_cm=s["length_px"] / px) for s in col["zones"]]
        out["pieces"] = []

        if method == "zones":
            out["rqd_percent"] = rqd.rqd_from_zones(out["segments"], col["height_cm"])
            out["rqd_class"] = rqd.quality_class(out["rqd_percent"])
            results.append(out)
            continue

        y0, y1, x0, x1 = col["y_start"], col["y_end"], col["x_start"], col["x_end"]
        col_bgr = bgr[y0:y1, x0:x1]
        crack = mask[y0:y1, x0:x1] if mask is not None else None
        box_pieces = pieces_step.pieces_from_boxes(state["piece_boxes"], col, px) if method == "piece_detect" else []

        # One denominator for every method: the core span seen by any model.
        span = pieces_step.core_span(col["height_px"], out["segments"], None if crack is None else crack > 0, box_pieces)

        if method == "piece_detect":
            found = box_pieces
        elif method == "contour":
            added: list = []
            found = pieces_step.extract_pieces_contour(crack, col_bgr, span, px, close_gap_cm, added_out=added)
            out["contour_added"] = added
        else:
            found, breaks = pieces_step.extract_pieces_cc_bridged(crack, col_bgr, px, close_gap_cm)
            if breaks:
                out["fracture_breaks"] = breaks

        core_cm = round((span[1] - span[0]) / px, 2) if span else 0.0
        if boxes:
            found = pieces_step.apply_intact_boxes(found, boxes, col, px)
            if found:
                core_cm = max(core_cm, round((found[-1]["y_end"] - found[0]["y_start"]) / px, 2))

        for v in pieces_step.apply_zone_rules(found, out["segments"]):
            warnings.append(f"Column {col['column_id']}: {v['length_cm']:.0f} cm piece rejected, "
                            f"{v['class']} zone (p={v['prob']:.2f})")
        out.update(pieces=found, has_pieces=True, core_length_cm=core_cm)
        results.append(out)

    if method != "zones":
        pieces_step.merge_across_columns(results)
        for out in results:
            out.update(rqd.rqd_from_pieces(out["pieces"], out["core_length_cm"], px))

    for out in results:
        overlay.draw_column(annotated, out)

    overall = rqd.overall_rqd(results, state.get("run_length_cm"))
    if overall.get("denominator_warning"):
        warnings.append(overall["denominator_warning"])

    result = {"status": "partial" if warnings else "ok", "warnings": warnings, "method": method,
              "pixels_per_cm": px, "image_shape": list(bgr.shape[:2]), "columns": results,
              **overall, **rqd.fracture_frequency(results),
              "lithology_image": state["lithology_image"],
              "annotated_image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)}
    decision = state.get("lithology_decision")
    if decision is not None:
        result["lithology"] = {"target": decision.target, "confidence": decision.confidence,
                               "source": decision.source, "conflict": decision.conflict}
    return result


# ── input checks ────────────────────────────────────────────────────────────

def validate_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Input must be a non-empty RGB array of shape (H, W, 3)")
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def validate_scale(pixels_per_cm: float) -> float:
    value = float(pixels_per_cm)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("pixels_per_cm must be finite and positive")
    return value
