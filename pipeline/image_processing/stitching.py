"""
Stitch the photos taken by the scanning robot into one tray image.

Three strategies, tried in this order:

1. Robot positions (scan_manifest.json). The robot records the X/Y of every
   photo. Core trays are repetitive (dividers, walls, look-alike core), so
   image matching alone regularly locks onto the wrong place. Here the robot
   decides the layout and image matching only measures the scale and nudges
   each photo by a few pixels. The scale also gives pixels per cm directly.
   See assemble_from_manifest.

2. Known grid. The file names carry the grid position ("capture_r2c4.png" =
   row 2, column 4). Neighbours are known; only their overlap is measured,
   with normalised template matching (no keypoints needed, so it works on
   low-texture rock).

3. Blind. OpenCV's Stitcher in SCANS mode (made for flat documents), then
   SIFT + RANSAC homography if that fails. Scale from a 5 cm ArUco marker.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

# "..._r<row>c<col>_..." anywhere in the file name, delimited by _ or -
_GRID_POS_RE = re.compile(r"(?:^|[_\-])r(\d+)c(\d+)(?:[_\-]|$)", re.IGNORECASE)

FALLBACK_PIXELS_PER_CM = 50.0
ARUCO_SIZE_CM = 5.0

MANIFEST_NAME = "scan_manifest.json"
WORK_SCALE = 0.5        # matching resolution, fraction of full size
REFINE_PX = 16          # max correction around the robot position (full-res px)
MIN_STEP_SCORE = 0.25   # mean NCC needed to trust the measured scale
MIN_SEAM_SCORE = 0.3    # NCC needed to nudge one seam
BLEND_POWER = 8.0       # higher = sharper hand-over between photos (less ghosting)
# The vision models were trained on portrait photos with core in vertical
# columns ~100-220 px wide. Each tray row of the panorama is turned into such
# a column at this width. Lithology gets the same layout 4x larger: tiles cut
# from 140 px columns cost ~10 points of accuracy.
MODEL_COLUMN_PX = 140
LITHOLOGY_COLUMN_PX = 560


def parse_grid_position(filename: str) -> tuple[int, int] | None:
    """(row, col) encoded in a file name, or None."""
    m = _GRID_POS_RE.search(Path(filename).stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def stitch_folder(folder: str | Path) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Load every image of a folder and stitch them.

    Returns (tray image, pixels_per_cm, high-resolution copy for lithology or None).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    frames, positions, by_name = [], [], {}
    for f in sorted(p for p in folder.iterdir() if p.suffix.lower() in exts):
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
            positions.append(parse_grid_position(f.name))
            by_name[f.name] = img

    if len(frames) < 2:
        raise ValueError(f"Need at least 2 readable images in {folder}, found {len(frames)}")

    placed = assemble_from_manifest(folder, by_name)
    if placed is not None:
        panorama, px_per_cm, centres, pitch = placed
        hires, _ = to_model_layout(panorama, px_per_cm, centres, pitch, LITHOLOGY_COLUMN_PX)
        tray, px_per_cm = to_model_layout(panorama, px_per_cm, centres, pitch, MODEL_COLUMN_PX)
        return tray, px_per_cm, hires

    if all(p is not None for p in positions):
        stitched = assemble_grid(frames, positions)
    else:
        stitched = stitch_blind(frames)
    stitched = crop_black_borders(stitched)
    return stitched, detect_scale_marker(stitched), None


def stitch_blind(frames: list[np.ndarray]) -> np.ndarray:
    """OpenCV Stitcher (SCANS mode), SIFT + RANSAC if it fails."""
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 images to stitch, got {len(frames)}")
    status, stitched = cv2.Stitcher_create(cv2.Stitcher_SCANS).stitch(frames)
    if status != cv2.Stitcher_OK:
        stitched = sift_stitch(frames)
    return stitched


# ── known-grid assembly ─────────────────────────────────────────────────────

def assemble_grid(frames: list[np.ndarray], positions: list[tuple[int, int]]) -> np.ndarray:
    """Build each row left to right, then stack the rows top to bottom."""
    by_row: dict[int, dict[int, np.ndarray]] = {}
    for frame, (r, c) in zip(frames, positions):
        by_row.setdefault(r, {})[c] = frame

    strips = []
    for r in sorted(by_row):
        strip, prev = None, None
        for c in sorted(by_row[r]):
            tile = by_row[r][c]
            strip = tile if strip is None else stitch_pair(strip, tile, axis=1, prev=prev)
            prev = tile
        strips.append(strip)

    panorama = strips[0]
    for prev, strip in zip(strips, strips[1:]):
        panorama = stitch_pair(panorama, strip, axis=0, prev=prev)
    return panorama


def stitch_pair(a: np.ndarray, b: np.ndarray, axis: int, fallback_overlap_frac: float = 0.15,
                prev: np.ndarray | None = None) -> np.ndarray:
    """Append b after a along axis (1 = horizontal, 0 = vertical), removing the overlap.

    prev is the tile (or row) b was captured next to. The overlap is searched
    in it only: searching the whole accumulated panorama let a repetitive
    tray match one tile further back and cut whole tiles out.
    """
    ref = a if prev is None else prev
    if axis == 1:
        h = min(a.shape[0], b.shape[0], ref.shape[0])
        a, b, ref = a[:h], b[:h], ref[:h]
    else:
        w = min(a.shape[1], b.shape[1], ref.shape[1])
        a, b, ref = a[:, :w], b[:, :w], ref[:, :w]

    found = find_overlap(ref, b, axis)
    if found is not None and found[1] >= 0.35:      # confident match
        overlap = found[0]
    else:                                            # assume the planned overlap
        overlap = int(b.shape[axis] * fallback_overlap_frac)

    if overlap > 0:
        a = a[:-overlap] if axis == 0 else a[:, :-overlap]
    return np.concatenate([a, b], axis=axis)


def find_overlap(a: np.ndarray, b: np.ndarray, axis: int) -> tuple[int, float] | None:
    """Slide b's leading strip (30 % of b) over a's trailing region and return
    (overlap_px, match_score) at the best normalised correlation."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    size_a = ga.shape[axis]
    template_size = max(8, int(gb.shape[axis] * 0.3))
    search_start = max(0, size_a - int(size_a * 0.65) - template_size)
    if axis == 1:
        template, search = gb[:, :template_size], ga[:, search_start:]
    else:
        template, search = gb[:template_size, :], ga[search_start:, :]

    if template.shape[axis] >= search.shape[axis] or min(template.shape) < 4:
        return None

    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    match_pos = search_start + (loc[1] if axis == 0 else loc[0])
    overlap = size_a - match_pos
    if overlap <= 0 or overlap >= size_a:
        return None
    return overlap, float(score)


# ── robot positions (scan manifest) ─────────────────────────────────────────
#
#   1. the px/mm scale and the photo orientation are measured once per scan by
#      summing the match scores of ALL neighbour pairs: one consensus peak, so
#      individual wrong matches are outvoted;
#   2. every photo is placed at scale x its robot position; image matching may
#      only nudge it by a few px (least squares over all photos at once);
#   3. brightness is equalised between photos and seams are blended,
#      favouring the centre of each photo (least lens distortion).

# The 8 ways a camera image can be mirrored/rotated relative to the machine axes.
ORIENTATIONS = ("identity", "flip_h", "flip_v", "rot180", "transpose", "rot90cw", "rot90ccw", "transpose_rot180")


def apply_orientation(img: np.ndarray, op: str) -> np.ndarray:
    return {
        "identity": lambda i: i, "flip_h": lambda i: i[:, ::-1], "flip_v": lambda i: i[::-1],
        "rot180": lambda i: i[::-1, ::-1], "transpose": lambda i: i.swapaxes(0, 1),
        "rot90cw": lambda i: np.rot90(i, -1), "rot90ccw": lambda i: np.rot90(i, 1),
        "transpose_rot180": lambda i: i.swapaxes(0, 1)[::-1, ::-1],
    }[op](img)


def _find_manifest(folder: Path) -> Path | None:
    """The manifest sits in the photo folder or one/two levels above it."""
    for d in (folder, *list(folder.parents)[:2]):
        if (d / MANIFEST_NAME).is_file():
            return d / MANIFEST_NAME
    return None


def manifest_tiles(folder: Path, by_name: dict[str, np.ndarray]) -> list[dict] | None:
    """[{row, col, a, b, img}]: a/b = robot position (mm) along the sample axis /
    across rows, signed so that a larger index means a larger position."""
    path = _find_manifest(folder)
    if path is None:
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    camera = (doc.get("passes") or {}).get("camera")
    points = camera.get("points") if isinstance(camera, dict) else None
    if not isinstance(points, list):
        return None
    primary = str(((doc.get("grid") or {}).get("definition") or {}).get("primary_axis", "x")).lower()

    tiles: dict[tuple[int, int], dict] = {}
    for p in points:
        if not isinstance(p, dict) or p.get("status", "ok") != "ok":
            continue
        img = by_name.get(Path(str(p.get("filepath") or "")).name)
        if img is None:
            continue
        try:
            row = int(p.get("row_index", p.get("row")))
            col = int(p.get("sample_index", p.get("sample")))
            x, y = float(p["x"]), float(p["y"])
        except (KeyError, TypeError, ValueError):
            continue
        a, b = (x, y) if primary == "x" else (y, x)
        tiles[(row, col)] = {"row": row, "col": col, "a": a, "b": b, "img": img}

    out = list(tiles.values())
    if len(out) < 2 or len({t["img"].shape for t in out}) != 1:
        return None
    for key, idx in (("a", "col"), ("b", "row")):
        idxs = np.array([t[idx] for t in out], float)
        vals = np.array([t[key] for t in out], float)
        if float(((idxs - idxs.mean()) * (vals - vals.mean())).sum()) < 0:
            for t in out:
                t[key] = -t[key]
    return out


def _match_prep(img: np.ndarray) -> np.ndarray:
    """Half-size, high-passed grayscale: texture only, insensitive to exposure differences."""
    small = cv2.resize(img, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 8)


def _neighbour_pairs(tiles: list[dict], along_row: bool) -> list[tuple[int, int]]:
    index = {(t["row"], t["col"]): i for i, t in enumerate(tiles)}
    pairs = []
    for (r, c), i in index.items():
        j = index.get((r, c + 1) if along_row else (r + 1, c))
        if j is not None:
            pairs.append((i, j))
    return pairs


def _dominant_pairs(tiles, pairs, key) -> tuple[list[tuple[int, int]], float]:
    """Pairs sharing the most common robot step (mm), and that step."""
    groups: dict[float, list] = {}
    for i, j in pairs:
        step = round(tiles[j][key] - tiles[i][key], 1)
        if step > 0:
            groups.setdefault(step, []).append((i, j))
    if not groups:
        return [], 0.0
    step = max(groups, key=lambda s: len(groups[s]))
    return groups[step], step


def _consensus_shift(work, pairs, along_row: bool, window: tuple[float, float] | None = None):
    """Shift (dx, dy) in work px of the second photo of every pair relative to
    the first that maximises the NCC summed over ALL pairs. (dx, dy, score) or None."""
    if not pairs:
        return None
    h, w = work[pairs[0][0]].shape
    total = None
    if along_row:
        m, tw = max(2, int(0.04 * h)), max(16, int(0.12 * w))
        for i, j in pairs:
            res = cv2.matchTemplate(work[i], work[j][m:h - m, :tw], cv2.TM_CCOEFF_NORMED)
            total = res if total is None else total + res
        total = np.nan_to_num(total / len(pairs), nan=-1.0)
        total[:, :max(1, int(0.05 * w))] = -1.0          # same spot is not a step
        y, x = np.unravel_index(int(np.argmax(total)), total.shape)
        return float(x), float(y - m), float(total[y, x])

    th, m = max(16, int(0.08 * h)), max(2, int(0.04 * w))
    if window is None:
        return None
    lo, hi = max(1, int(window[0])), min(h - th, int(np.ceil(window[1])))
    if hi < lo:
        return None                                       # rows don't overlap
    for i, j in pairs:
        res = cv2.matchTemplate(work[i], work[j][:th, m:w - m], cv2.TM_CCOEFF_NORMED)
        total = res if total is None else total + res
    total = np.nan_to_num(total / len(pairs), nan=-1.0)
    allowed = np.full_like(total, -1.0)
    allowed[lo:hi + 1] = total[lo:hi + 1]
    y, x = np.unravel_index(int(np.argmax(allowed)), allowed.shape)
    return float(x - m), float(y), float(allowed[y, x])


def _overlap_windows(shape, rel, r):
    """Search window in the first photo and template in the second for an
    offset rel (work px) +- r. None when they barely overlap."""
    h, w = shape
    rx, ry = int(round(rel[0])), int(round(rel[1]))
    x0, x1, y0, y1 = max(0, rx), min(w, rx + w), max(0, ry), min(h, ry + h)
    tx0, tx1, ty0, ty1 = x0 - rx + r, x1 - rx - r, y0 - ry + r, y1 - ry - r
    if tx1 - tx0 < 24 or ty1 - ty0 < 24:
        return None
    return (rx, ry), (slice(y0, y1), slice(x0, x1)), (slice(ty0, ty1), slice(tx0, tx1))


def _group_refine(work, pairs, rel, r=40):
    """Refine a shared neighbour offset by matching the WHOLE overlap of every
    pair at once. The thin strip used for the coarse search is biased by lens
    distortion; the full overlap is not."""
    win = _overlap_windows(work[pairs[0][0]].shape, rel, r)
    if win is None:
        return None
    (rx, ry), sa, sb = win
    total = None
    for i, j in pairs:
        res = cv2.matchTemplate(work[i][sa], work[j][sb], cv2.TM_CCOEFF_NORMED)
        total = res if total is None else total + res
    total = np.nan_to_num(total / len(pairs), nan=-1.0)
    y, x = np.unravel_index(int(np.argmax(total)), total.shape)
    if x in (0, total.shape[1] - 1) or y in (0, total.shape[0] - 1):
        return None
    return float(rx + x - r), float(ry + y - r), float(total[y, x])


def measure_layout(tiles: list[dict]):
    """Photo orientation and full-res px/mm vectors along the sample axis (h)
    and across rows (v): (orientation, h_vec, v_vec, score), or None."""
    h_pairs, h_step = _dominant_pairs(tiles, _neighbour_pairs(tiles, True), "a")
    v_pairs, v_step = _dominant_pairs(tiles, _neighbour_pairs(tiles, False), "b")
    if not h_pairs:
        return None
    base = [_match_prep(t["img"]) for t in tiles]

    def evaluate(op, hp, vp):
        work = [np.ascontiguousarray(apply_orientation(g, op)) for g in base]
        hs = _consensus_shift(work, hp, along_row=True)
        if hs is None:
            return None
        h_vec = np.array(hs[:2]) / h_step
        score = hs[2]
        v_vec = np.array([-h_vec[1], h_vec[0]])          # square pixels, same tilt
        if vp:
            expected = float(np.linalg.norm(h_vec)) * v_step
            vs = _consensus_shift(work, vp, along_row=False, window=(0.8 * expected, 1.2 * expected))
            if vs is not None:
                v_vec = np.array(vs[:2]) / v_step
                score += vs[2]
        return h_vec / WORK_SCALE, v_vec / WORK_SCALE, score, hs[2]

    # Orientation chosen on a spread sample of pairs, then measured on all of them.
    sample = lambda p: p[::max(1, len(p) // 12)]
    results = {op: r for op in ORIENTATIONS if (r := evaluate(op, sample(h_pairs), sample(v_pairs))) is not None}
    if "identity" not in results:
        return None
    best = max(results, key=lambda o: results[o][2])
    if results[best][2] < results["identity"][2] + 0.05:   # keep photos as saved unless clearly better
        best = "identity"
    full = evaluate(best, h_pairs, v_pairs)
    if full is None:
        return None
    h_vec, v_vec, score, h_score = full
    if np.linalg.norm(h_vec) <= 0 or h_score < MIN_STEP_SCORE:
        return None

    work = [np.ascontiguousarray(apply_orientation(g, best)) for g in base]
    fine = _group_refine(work, h_pairs, h_vec * WORK_SCALE * h_step)
    if fine is not None:
        h_vec = np.array(fine[:2]) / (WORK_SCALE * h_step)
    if v_pairs:
        fine_v = _group_refine(work, v_pairs, v_vec * WORK_SCALE * v_step)
        if fine_v is not None and fine_v[2] >= 0.15:
            v_vec = np.array(fine_v[:2]) / (WORK_SCALE * v_step)
        else:
            v_vec = np.array([-h_vec[1], h_vec[0]])
    return best, h_vec, v_vec, score


def _refine_offset(work_a, work_b, rel_full):
    """Best offset of b relative to a within +-REFINE_PX of rel_full: (offset, score) or None."""
    r = max(2, int(REFINE_PX * WORK_SCALE))
    win = _overlap_windows(work_a.shape, np.asarray(rel_full) * WORK_SCALE, r)
    if win is None:
        return None
    (rx, ry), sa, sb = win
    res = cv2.matchTemplate(work_a[sa], work_b[sb], cv2.TM_CCOEFF_NORMED)
    _, score, _, (px, py) = cv2.minMaxLoc(np.nan_to_num(res, nan=-1.0))
    if px in (0, res.shape[1] - 1) or py in (0, res.shape[0] - 1):
        return None                                       # optimum outside the window
    return np.array([rx + px - r, ry + py - r], float) / WORK_SCALE, float(score)


def _overlap_means(img_a, img_b, rel):
    h, w = img_a.shape[:2]
    rx, ry = int(round(rel[0])), int(round(rel[1]))
    x0, x1, y0, y1 = max(0, rx), min(w, rx + w), max(0, ry), min(h, ry + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    ma = img_a[y0:y1:4, x0:x1:4].reshape(-1, 3).mean(axis=0)
    mb = img_b[y0 - ry:y1 - ry:4, x0 - rx:x1 - rx:4].reshape(-1, 3).mean(axis=0)
    return ma, mb


def assemble_from_manifest(folder: Path, by_name: dict[str, np.ndarray]):
    """(panorama, px_per_cm, row centres, row pitch), or None to fall back."""
    tiles = manifest_tiles(folder, by_name)
    if tiles is None:
        return None
    layout = measure_layout(tiles)
    if layout is None:
        return None
    op, h_vec, v_vec, _ = layout

    imgs = [np.ascontiguousarray(apply_orientation(t["img"], op)) for t in tiles]
    work = [_match_prep(img) for img in imgs]
    n = len(tiles)
    pred = np.array([h_vec * t["a"] + v_vec * t["b"] for t in tiles])   # where the robot says

    # Joint fine-tuning: measured neighbour offsets (strong weight) + robot
    # positions (weak anchor), least squares, clamped near the robot position.
    edges = []
    for i, j in _neighbour_pairs(tiles, True) + _neighbour_pairs(tiles, False):
        found = _refine_offset(work[i], work[j], pred[j] - pred[i])
        if found is not None and found[1] >= MIN_SEAM_SCORE:
            edges.append((i, j, found[0], found[1]))
    rows, rhs = [], []
    for i, j, rel, wgt in edges:
        row = np.zeros(n)
        row[j], row[i] = wgt, -wgt
        rows.append(row)
        rhs.append(wgt * rel)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 0.05
        rows.append(row)
        rhs.append(0.05 * pred[i])
    pos = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    pos = np.clip(pos, pred - 3 * REFINE_PX, pred + 3 * REFINE_PX)

    # Brightness / colour equalisation: one log-gain per photo and channel.
    rows, rhs = [], []
    for i, j, _, _ in edges:
        means = _overlap_means(imgs[i], imgs[j], pos[j] - pos[i])
        if means is None:
            continue
        row = np.zeros(n)
        row[j], row[i] = 1.0, -1.0
        rows.append(row)
        rhs.append(np.log(means[0] + 1) - np.log(means[1] + 1))
    for i in range(n):
        row = np.zeros(n)
        row[i] = 0.3
        rows.append(row)
        rhs.append(np.zeros(3))
    gains = np.clip(np.exp(np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]), 0.75, 1.33)

    # Compose in horizontal bands (bounded memory), centre-weighted blend.
    pos = np.round(pos - pos.min(axis=0)).astype(int)
    th, tw = imgs[0].shape[:2]
    ch, cw = int(pos[:, 1].max()) + th, int(pos[:, 0].max()) + tw
    wy = 1.0 - np.abs(np.arange(th) - (th - 1) / 2) / (th / 2)
    wx = 1.0 - np.abs(np.arange(tw) - (tw - 1) / 2) / (tw / 2)
    weight = (np.clip(np.outer(wy, wx), 1e-4, None) ** BLEND_POWER).astype(np.float32)
    canvas = np.zeros((ch, cw, 3), np.uint8)
    for y0 in range(0, ch, 256):
        y1 = min(ch, y0 + 256)
        acc = np.zeros((y1 - y0, cw, 3), np.float32)
        wsum = np.zeros((y1 - y0, cw), np.float32)
        for k, (px, py) in enumerate(pos):
            a, b = max(y0, py), min(y1, py + th)
            if a >= b:
                continue
            wk = weight[a - py:b - py]
            acc[a - y0:b - y0, px:px + tw] += (imgs[k][a - py:b - py].astype(np.float32)
                                               * gains[k].astype(np.float32)) * wk[..., None]
            wsum[a - y0:b - y0, px:px + tw] += wk
        covered = wsum > 0
        out = np.zeros_like(acc)
        out[covered] = acc[covered] / wsum[covered][:, None]
        canvas[y0:y1] = np.clip(out, 0, 255).astype(np.uint8)

    # A camera slightly rotated against the machine X axis makes the tray slope: level it.
    tilt = float(np.degrees(np.arctan2(h_vec[1], h_vec[0])))
    rot = cv2.getRotationMatrix2D((cw / 2, ch / 2), tilt, 1.0)
    if 0.1 < abs(tilt) < 10:
        canvas = cv2.warpAffine(canvas, rot, (cw, ch), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else:
        rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Centre line of every tray row in the levelled panorama, and the row pitch.
    centres = []
    for r in sorted({t["row"] for t in tiles}):
        idx = [k for k, t in enumerate(tiles) if t["row"] == r]
        cx = float(np.median(pos[idx, 0])) + tw / 2
        cy = float(np.median(pos[idx, 1])) + th / 2
        centres.append(float(rot[1, 0] * cx + rot[1, 1] * cy + rot[1, 2]))
    _, v_step = _dominant_pairs(tiles, _neighbour_pairs(tiles, False), "b")
    pitch = float(np.linalg.norm(v_vec)) * v_step if v_step > 0 else float(th)

    px_per_cm = float(np.linalg.norm(h_vec)) * 10.0       # px/mm -> px/cm
    return canvas, px_per_cm, centres, pitch


def to_model_layout(panorama: np.ndarray, px_per_cm: float, centres: list[float], pitch: float,
                    column_px: int = MODEL_COLUMN_PX) -> tuple[np.ndarray, float]:
    """Lay the tray out like the training photos: each tray row becomes a
    vertical column (row start at the top, first row on the left), scaled to
    ~column_px wide. px/cm is scaled by the same factor."""
    h = panorama.shape[0]
    half = int(round(pitch / 2))
    columns = []
    for cy in centres:
        y0, y1 = int(round(cy)) - half, int(round(cy)) + half
        band = np.zeros((y1 - y0, panorama.shape[1], 3), np.uint8)
        a, b = max(0, y0), min(h, y1)
        if a < b:
            band[a - y0:b - y0] = panorama[a:b]
        columns.append(cv2.rotate(band, cv2.ROTATE_90_CLOCKWISE))
    f = column_px / max(1.0, pitch)
    tray = cv2.resize(np.hstack(columns), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return tray, px_per_cm * f


# ── blind fallback ──────────────────────────────────────────────────────────

def sift_stitch(frames: list[np.ndarray]) -> np.ndarray:
    """SIFT keypoints, Lowe's ratio test, RANSAC homography, warp onto a growing canvas."""
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    result = frames[0]

    for frame in frames[1:]:
        kp1, des1 = sift.detectAndCompute(result, None)
        kp2, des2 = sift.detectAndCompute(frame, None)
        if des1 is None or des2 is None:
            continue
        good = [m for m, n in matcher.knnMatch(des1, des2, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            continue

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)

        h1, w1 = result.shape[:2]
        h2, w2 = frame.shape[:2]
        warped = cv2.warpPerspective(frame, H, (w1 + w2, max(h1, h2)))
        warped[0:h1, 0:w1] = result
        result = warped

    return result


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    """Remove the black margins left by warping."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return image[y:y + h, x:x + w]


def detect_scale_marker(image: np.ndarray) -> float:
    """Pixels per cm from a 5 cm ArUco 4x4 marker; 50 px/cm if none is found."""
    try:
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters(),
        )
        corners, ids, _ = detector.detectMarkers(image)
        if ids is not None and len(ids) > 0:
            c = corners[0][0]
            return float(np.linalg.norm(c[0] - c[1])) / ARUCO_SIZE_CM
    except Exception:
        pass
    return FALLBACK_PIXELS_PER_CM
