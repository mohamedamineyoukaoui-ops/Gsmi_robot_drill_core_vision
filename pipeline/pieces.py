"""
Step 5 — turn a column into a list of rock pieces with a length in cm.

A piece is a depth interval {y_start, y_end, length_cm, counts_for_rqd}.
It counts for RQD when it is at least 10 cm long (ASTM D6032).

Three interchangeable methods:

  contour               (default) regions enclosed by the U-Net outline, kept
                        only if their interior is solid rock.
  connected_components  rock = core minus crack pixels; fractures that cross
                        the core are first extended over its full width.
  piece_detect          no crack mask: the YOLO11 detector returns pieces as boxes,
                        stacked along the core without overlap (resolve_overlaps).

The same rules then apply to every method, so their results are comparable:
  - one denominator: the core span seen by any model (core_span);
  - the zone classifier can veto pieces lying in confident rubble/sand/empty (apply_zone_rules);
  - a piece cut by a column boundary is counted whole (merge_across_columns).

Sizes that matter physically are given in cm and converted with the image's
own scale, so the method behaves the same at any resolution.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline import weights

MIN_INTACT_CM = 10.0
CLOSE_GAP_CM = 0.6        # mask gap bridged (best of 0.15-0.8 cm on 528 annotated columns)
CONTOUR_MAX_DARK = 0.20   # max share of a piece's interior that may be dark tray
BRIDGE_ALONG_CM = 1.0     # gap bridged along a fracture trace
FULL_CROSSING = 0.70      # share of the core width a fracture must cross
# A piece is overruled when > 50 % of it lies in a zone of this class whose
# mean probability reaches the threshold. "empty" is the strictest: a piece
# means another model saw rock there.
ZONE_VETO_PROB = {"empty": 0.85, "rubble": 0.70, "sand": 0.70}

_detector = None


def cm_to_px(cm: float, px_per_cm: float, minimum: int = 1) -> int:
    return max(minimum, int(round(float(cm) * float(px_per_cm))))


def _piece(y0: int, y1: int, px_per_cm: float, **extra: Any) -> dict:
    length_px = int(y1 - y0)
    length_cm = round(length_px / px_per_cm, 2) if px_per_cm > 0 else 0.0
    return {"y_start": int(y0), "y_end": int(y1), "length_px": length_px, "length_cm": length_cm,
            "is_rubble": False, "is_void": False, "counts_for_rqd": length_cm >= MIN_INTACT_CM, **extra}


def _binary(mask: np.ndarray) -> np.ndarray:
    m = mask > 0
    return m.any(axis=2) if m.ndim == 3 else m


# ── geometry of the core inside a column ────────────────────────────────────

def core_band(col_bgr: np.ndarray | None, col_w: int) -> tuple[int, int]:
    """x-range of the core cylinder: the widest bright band between the dark
    tray walls (median brightness of every image column, smoothed, threshold
    halfway between min and max). Falls back to the central 70 %."""
    fallback = (int(0.15 * col_w), max(int(0.15 * col_w) + 1, int(0.85 * col_w)))
    if col_bgr is None or col_bgr.size == 0 or col_w < 8:
        return fallback
    gray = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2GRAY) if col_bgr.ndim == 3 else col_bgr
    prof = np.median(gray, axis=0).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (0, 1), sigmaX=max(1.0, col_w / 40)).ravel()
    bright = prof > 0.5 * (float(prof.min()) + float(prof.max()))

    best, start, best_len = None, None, 0
    for x, b in enumerate(list(bright) + [False]):
        if b and start is None:
            start = x
        elif not b and start is not None:
            if x - start > best_len:
                best, best_len = (start, x), x - start
            start = None
    return best if best is not None and best_len >= 0.3 * col_w else fallback


def core_span(col_h: int, segments: list[dict], crack: np.ndarray | None,
              box_pieces: list[dict] | None = None) -> tuple[int, int] | None:
    """Rows of the column that hold core: the RQD denominator, same for every method.

    A row holds material when any model sees something there: the zone
    classifier says intact/rubble/sand, the crack mask is present, or a
    detector piece covers it. Empty tray before/after the core is excluded;
    empty stretches inside it (core loss) stay, as the RQD definition requires.
    With detector pieces alone the span would be their own extent and RQD
    would drift to 100 % by construction, so the full column is used instead.
    """
    if col_h <= 0:
        return None
    zoned = [s for s in segments or [] if s.get("zone_class")]
    has_mask = crack is not None and crack.size > 0
    if not zoned and not has_mask:
        return 0, col_h

    material = np.zeros(col_h, bool)
    for s in zoned:
        if s["zone_class"] in ("intact", "rubble", "sand"):
            material[max(0, int(s["y_start"])):min(col_h, int(s["y_end"]))] = True
    if has_mask:
        rows = crack.any(axis=1)
        material[:min(col_h, rows.size)] |= rows[:col_h]
    for p in box_pieces or []:
        material[max(0, int(p["y_start"])):min(col_h, int(p["y_end"]))] = True

    idx = np.flatnonzero(material)
    return (int(idx[0]), int(idx[-1]) + 1) if idx.size else None


def bridge_mask(mask: np.ndarray, px_per_cm: float, close_gap_cm: float) -> np.ndarray:
    """Close small holes in the crack trace at the real scale: a round closing
    (close_gap_cm) plus a horizontal closing along the trace (1 cm), so a
    fracture with a hole still separates the rock on both sides."""
    u8 = _binary(mask).astype(np.uint8) * 255
    k = cm_to_px(close_gap_cm, px_per_cm)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    along = cm_to_px(BRIDGE_ALONG_CM, px_per_cm, minimum=3)
    return cv2.morphologyEx(u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (along, 1)))


def crossing_fractures(mask: np.ndarray, col_bgr: np.ndarray | None, px_per_cm: float,
                       close_gap_cm: float = CLOSE_GAP_CM) -> list[tuple[int, int]]:
    """Fractures that cross the core, as depth intervals (y0, y1).

    1. Keep the core band minus 12 % on each side: the U-Net also traces the
       core's long edges, and those are not fractures.
    2. Bridge small holes (bridge_mask).
    3. A trace is a fracture when it spans >= 70 % of that width. This is a
       projection test, so holes inside the trace do not matter.
    4. Its depth is where it crosses the core's centre line (ASTM measures
       piece length along the axis), not its oblique ends.
    """
    if mask is None or mask.size == 0:
        return []
    h, w = _binary(mask).shape
    xl, xr = core_band(col_bgr, w)
    bw = max(1, xr - xl)
    il, ir = xl + int(0.12 * bw), xr - int(0.12 * bw)
    if ir - il < 4:
        return []

    inner = np.ascontiguousarray(bridge_mask(mask, px_per_cm, close_gap_cm)[:, il:ir])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
    cx0 = (xl + xr) // 2 - int(0.10 * bw) - il
    cx1 = (xl + xr) // 2 + int(0.10 * bw) - il

    breaks = []
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if ww < FULL_CROSSING * (ir - il):
            continue
        ys = np.flatnonzero((labels[y:y + hh, max(0, cx0):max(0, cx1)] == i).any(axis=1))
        if ys.size:
            breaks.append((int(y + ys[0]), int(y + ys[-1]) + 1))
        else:
            mid = int(y + hh // 2)
            breaks.append((mid, mid + 1))

    merged: list[tuple[int, int]] = []
    for b0, b1 in sorted(breaks):
        if merged and b0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b1))
        else:
            merged.append((b0, b1))
    return merged


# ── method 1: contour (default) ─────────────────────────────────────────────

def extract_pieces_contour(mask: np.ndarray, col_bgr: np.ndarray | None, span: tuple[int, int] | None,
                           px_per_cm: float, close_gap_cm: float = CLOSE_GAP_CM,
                           added_out: list | None = None) -> list[dict]:
    """Pieces = regions enclosed by the U-Net outline whose interior is solid rock.

    1. Draw the core's outer outline (its two long edges and the span ends)
       into the mask. Otherwise a piece whose outline is open (touching the
       column end, or with an untraced edge) would be flood-filled away.
    2. Close gaps up to close_gap_cm, flood-fill from the border to remove the
       region connected to it (empty tray); each enclosed region is a candidate.
    3. Keep a candidate only if at most 20 % of its eroded interior is darker
       than the rock/tray threshold. The added outline also encloses rubble
       (bright fragments with dark gaps) and empty tray, rejected here.
    Benchmarked on 528 hand-annotated columns: RQD mean absolute error 15.1 -> 11.5.

    added_out, if given, receives the outline polygons the method ADDED to the
    U-Net trace, for display.
    """
    m = _binary(mask)
    h, w = m.shape

    # 1. seal the core outline
    sealed = m.astype(np.uint8) * 255
    band = core_band(col_bgr, w)
    top, bottom = span if span else (0, h)
    t = max(2, w // 60)
    cv2.rectangle(sealed, (max(0, band[0] - t), max(0, top - t)),
                  (min(w - 1, band[1] + t), min(h - 1, bottom + t)), 255, t)

    # 2. close gaps, thicken, flood-fill the outside away
    k = max(3, cm_to_px(close_gap_cm, px_per_cm) * 2 + 1)
    closed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    if added_out is not None:
        added = ((closed > 0) & ~m).astype(np.uint8)
        polys, _ = cv2.findContours(added, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        added_out.extend(p.reshape(-1, 2).tolist() for p in polys if len(p) >= 3)

    padded = cv2.copyMakeBorder(cv2.bitwise_not(closed), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    cv2.floodFill(padded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 0)
    rock = cv2.morphologyEx(np.ascontiguousarray(padded[1:-1, 1:-1]), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rock, connectivity=8)
    min_area = max(20, int(0.002 * h * w))

    # 3. solid interior test: threshold halfway between the darkest tray and typical rock
    gray = None
    if col_bgr is not None and col_bgr.size:
        gray = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2GRAY) if col_bgr.ndim == 3 else col_bgr
        core = gray[:, band[0]:band[1]]
        threshold = 0.5 * (float(np.percentile(gray, 5)) + float(np.percentile(core if core.size else gray, 75)))
    shrink = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    pieces = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        if gray is not None:
            interior = cv2.erode((labels[y:y + hh, x:x + ww] == i).astype(np.uint8), shrink) > 0
            values = gray[y:y + hh, x:x + ww][interior]
            if not values.size or float((values < threshold).mean()) > CONTOUR_MAX_DARK:
                continue
        pieces.append(_piece(y, y + hh, px_per_cm))
    return sorted(pieces, key=lambda p: p["y_start"])


# ── method 2: connected components ──────────────────────────────────────────

def extract_pieces_cc(mask: np.ndarray, px_per_cm: float) -> list[dict]:
    """Pieces = connected regions of (core span minus crack pixels).

    The core span runs from the first to the last row containing mask pixels.
    Overlapping pieces (rock enclosed by a crack loop and the rock around it)
    are not merged here; rqd.rqd_from_pieces takes the union of their depths.
    """
    crack = _binary(mask)
    h, w = crack.shape
    rows = np.where(crack.any(axis=1))[0]
    if rows.size == 0:
        return []                          # no mask = no core: never invent a piece
    top, bottom = int(rows[0]), int(rows[-1]) + 1

    rock = np.ones((bottom - top, w), np.uint8)
    rock[crack[top:bottom]] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(rock, connectivity=8)
    min_area = max(20, int(0.002 * (bottom - top) * w))
    return sorted((_piece(top + y, top + y + hh, px_per_cm) for _, y, _, hh, area in stats[1:n]
                   if area >= min_area), key=lambda p: p["y_start"])


def extract_pieces_cc_bridged(mask: np.ndarray, col_bgr: np.ndarray | None, px_per_cm: float,
                              close_gap_cm: float = CLOSE_GAP_CM) -> tuple[list[dict], list[tuple[int, int]]]:
    """connected_components after closing holes and extending every crossing
    fracture over the full width. Without that, the rock stays connected
    around the end of a fracture and the whole column fuses into one piece."""
    breaks = crossing_fractures(mask, col_bgr, px_per_cm, close_gap_cm)
    bridged = bridge_mask(mask, px_per_cm, close_gap_cm)
    for b0, b1 in breaks:
        bridged[b0:max(b0 + 1, b1), :] = 255
    return extract_pieces_cc(bridged, px_per_cm), breaks


# ── method 3: direct piece detection ────────────────────────────────────────

def piece_detector_available() -> bool:
    return weights.exists(weights.PIECES)


def detect_pieces(bgr_image: np.ndarray, conf: float = 0.2, imgsz: int = 1024) -> list[dict]:
    """Boxes {x1, y1, x2, y2, confidence} of intact pieces in the full image, top to bottom."""
    global _detector
    if _detector is None:
        from ultralytics import YOLO
        _detector = YOLO(str(weights.path(weights.PIECES)), task="detect")

    result = _detector.predict(bgr_image, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = []
    if result.boxes is not None:
        for (x1, y1, x2, y2), c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            boxes.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "confidence": float(c)})
    return sorted(boxes, key=lambda b: b["y1"])


def pieces_from_boxes(boxes: list[dict], col: dict, px_per_cm: float) -> list[dict]:
    """Detector boxes → this column's pieces: assign to the column, then make them non-overlapping."""
    return sorted(resolve_overlaps(boxes_in_column(boxes, col, px_per_cm), px_per_cm), key=lambda p: p["y_start"])


def boxes_in_column(boxes: list[dict], col: dict, px_per_cm: float, min_overlap: float = 0.5) -> list[dict]:
    """Detector boxes that belong to this column, as column-local (possibly overlapping) pieces.

    A box belongs to a column when most of its width lies inside the column,
    so a box is never counted in two neighbouring columns.
    """
    x0, x1, y0 = col["x_start"], col["x_end"], col["y_start"]
    col_h, col_w = col["y_end"] - y0, max(1, x1 - x0)
    pieces = []
    for b in boxes:
        overlap = min(b["x2"], x1) - max(b["x1"], x0)
        if overlap <= 0 or overlap / min(max(1, b["x2"] - b["x1"]), col_w) < min_overlap:
            continue
        top = max(0, min(col_h, b["y1"] - y0))
        bottom = max(0, min(col_h, b["y2"] - y0))
        if bottom - top > 1:
            pieces.append(_piece(top, bottom, px_per_cm, confidence=round(b["confidence"], 3)))
    return sorted(pieces, key=lambda p: p["y_start"])


def resolve_overlaps(pieces: list[dict], px_per_cm: float) -> list[dict]:
    """Make one column's detector boxes non-overlapping: pieces stack along the core.

    The training labels were made with the same rule, so the output is resolved
    the way the labels were. Top to bottom, each box is compared with the piece above it:
      - an overlap of at least half the shorter one is the same depth reported
        twice (e.g. the two halves of a core split lengthwise): keep the more confident;
      - a smaller overlap is the boundary between two pieces: cut it at its middle.
    """
    out: list[dict] = []
    for p in sorted((dict(q) for q in pieces), key=lambda q: q["y_start"]):
        if out and p["y_start"] < out[-1]["y_end"]:
            above = out[-1]
            overlap = above["y_end"] - p["y_start"]
            if overlap >= 0.5 * min(above["y_end"] - above["y_start"], p["y_end"] - p["y_start"]):
                if p.get("confidence", 0.0) > above.get("confidence", 0.0):
                    out[-1] = p
                continue
            above["y_end"] = p["y_start"] = (above["y_end"] + p["y_start"]) // 2
        out.append(p)

    result = []
    for p in out:
        length_px = p["y_end"] - p["y_start"]
        if length_px <= 1:
            continue
        p["length_px"] = int(length_px)
        p["length_cm"] = round(length_px / px_per_cm, 2) if px_per_cm > 0 else 0.0
        p["counts_for_rqd"] = p["length_cm"] >= MIN_INTACT_CM
        result.append(p)
    return result


# ── rules shared by every method ────────────────────────────────────────────

def apply_intact_boxes(pieces: list[dict], boxes: list[tuple[int, int, int, int]], col: dict,
                       px_per_cm: float) -> list[dict]:
    """A user box (x, y, w, h) declares its depth range to be ONE intact piece.

    Pieces whose midpoint falls inside the box are replaced by it; its length
    is the box height, whatever the crack mask says inside.
    """
    col_h = col["y_end"] - col["y_start"]
    spans = []
    for bx, by, bw, bh in boxes:
        if bx + bw <= col["x_start"] or bx >= col["x_end"]:
            continue
        y0 = max(0, min(col_h, int(by) - col["y_start"]))
        y1 = max(0, min(col_h, int(by + bh) - col["y_start"]))
        if y1 - y0 > 1:
            spans.append((y0, y1))
    if not spans:
        return pieces

    def inside(p: dict) -> bool:
        mid = 0.5 * (p["y_start"] + p["y_end"])
        return any(y0 <= mid < y1 for y0, y1 in spans)

    kept = [p for p in pieces if not inside(p)]
    kept += [_piece(y0, y1, px_per_cm, manual=True) for y0, y1 in spans]
    return sorted(kept, key=lambda p: p["y_start"])


def apply_zone_rules(pieces: list[dict], segments: list[dict]) -> list[dict]:
    """The zone classifier overrules pieces it is confident about:
    empty -> void (not rock), rubble / sand -> never counts for RQD.
    User boxes are never overruled. Returns the vetoed pieces that would
    otherwise have counted, so they can be reported."""
    vetoes = []
    zoned = [s for s in segments or [] if s.get("zone_class")]
    if not zoned:
        return vetoes
    for p in pieces:
        if p.get("manual"):
            continue
        y0, y1 = int(p["y_start"]), int(p["y_end"])
        length = max(1, y1 - y0)
        share: dict[str, float] = {}
        prob: dict[str, float] = {}
        for s in zoned:
            cls = s["zone_class"]
            pr = float((s.get("probs") or {}).get(cls, 1.0))
            if cls not in ZONE_VETO_PROB or pr < ZONE_VETO_PROB[cls]:
                continue
            ov = min(y1, int(s["y_end"])) - max(y0, int(s["y_start"]))
            if ov > 0:
                share[cls] = share.get(cls, 0.0) + ov / length
                prob[cls] = max(prob.get(cls, 0.0), pr)
        if not share:
            continue
        cls = max(share, key=share.get)
        if share[cls] <= 0.5:
            continue
        would_count = bool(p["counts_for_rqd"]) or p["length_cm"] >= MIN_INTACT_CM
        p["counts_for_rqd"] = False
        p["zone_type"], p["zone_prob"] = cls, round(prob[cls], 2)
        p["is_void"], p["is_rubble"] = cls == "empty", cls != "empty"
        if would_count and cls != "empty":
            vetoes.append({"class": cls, "length_cm": p["length_cm"], "prob": round(prob[cls], 2)})
    return vetoes


def merge_across_columns(columns: list[dict], edge_tol_px: int = 3) -> None:
    """Column N's bottom is column N+1's top (same core, cut to fit the tray).

    A piece touching the bottom of one column and a piece touching the top of
    the next are one physical piece split in two. Each half may be < 10 cm
    and be wrongly rejected, so the halves are chained, their lengths added,
    and the 10 cm rule applied to the total. Chained pieces share a chain_id.
    A piece overruled by the zones never joins a chain (it would be re-enabled).
    Modifies the pieces in place.
    """
    chain: list[dict] = []
    chain_cm = 0.0
    next_id = 0

    def close() -> None:
        nonlocal chain, chain_cm, next_id
        if len(chain) >= 2:
            for p in chain:
                p["counts_for_rqd"] = chain_cm >= MIN_INTACT_CM
                p["merged_length_cm"] = round(chain_cm, 2)
                p["chain_id"] = next_id
            next_id += 1
        chain, chain_cm = [], 0.0

    def overruled(p: dict) -> bool:
        return bool(p.get("is_rubble") or p.get("is_void"))

    for col in columns:
        pieces = col.get("pieces") or []
        if not pieces:
            close()
            continue
        top, bottom = pieces[0], pieces[-1]
        touches_top = top["y_start"] <= edge_tol_px and not overruled(top)
        touches_bottom = bottom["y_end"] >= col["height_px"] - edge_tol_px and not overruled(bottom)

        if touches_top and chain:
            chain.append(top)
            chain_cm += top["length_cm"]
            if top is not bottom:
                close()
        else:
            close()

        if top is bottom and touches_top and touches_bottom:
            if not chain:                       # one piece spanning the whole column
                chain, chain_cm = [top], top["length_cm"]
        elif touches_bottom:
            close()
            chain, chain_cm = [bottom], bottom["length_cm"]
        else:
            close()

    close()
