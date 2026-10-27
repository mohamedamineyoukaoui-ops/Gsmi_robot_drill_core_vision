"""Draw the result on the tray image so it can be checked by eye.

    yellow  crack mask (after manual corrections) and column boundaries
    cyan    outline the contour method ADDED to the mask to enclose pieces
    red     fracture crossing the core (connected_components method)
    green   piece >= 10 cm      -> counts for RQD
    orange  piece <  10 cm      -> does not count
    red     rubble / sand zone  -> does not count
    blue    empty tray          -> not rock
"""

import cv2
import numpy as np

# BGR
YELLOW, CYAN, RED_LINE = (0, 255, 255), (255, 255, 0), (0, 0, 230)
FILLS = {"counts": (60, 160, 60), "short": (40, 130, 200), "rubble": (60, 60, 190), "void": (170, 110, 40)}
LINES = {"counts": (0, 230, 0), "short": (0, 165, 255), "rubble": (0, 0, 200), "void": (200, 130, 50)}


def _role(p: dict) -> str:
    if p.get("is_void"):
        return "void"
    if p.get("is_rubble"):
        return "rubble"
    return "counts" if p["counts_for_rqd"] else "short"


def tint_mask(image: np.ndarray, mask: np.ndarray) -> None:
    overlay = image.copy()
    overlay[mask > 0] = YELLOW
    image[:] = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)


def draw_column(image: np.ndarray, col: dict) -> None:
    x1, x2, y1, y2 = col["x_start"], col["x_end"], col["y_start"], col["y_end"]
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), YELLOW, 2)

    pieces = col.get("pieces") or []
    overlay = image.copy()
    for p in pieces:
        cv2.rectangle(overlay, (x1, y1 + p["y_start"]), (x2, y1 + p["y_end"]), FILLS[_role(p)], -1)
    if not pieces and not col.get("has_pieces"):
        for s in col.get("segments", []):          # RQD from zones only
            fill = FILLS["counts"] if s["type"] == "intact" else FILLS["rubble"]
            cv2.rectangle(overlay, (x1, y1 + s["y_start"]), (x2, y1 + s["y_end"]), fill, -1)
    cv2.addWeighted(overlay, 0.28, image, 0.72, 0, image)

    for p in pieces:
        for y in (y1 + p["y_start"], y1 + p["y_end"]):
            cv2.line(image, (x1, y), (x2, y), LINES[_role(p)], 1)

    added = col.get("contour_added")
    if added:
        layer = image.copy()
        cv2.fillPoly(layer, [np.array(p, np.int32) + np.array([x1, y1], np.int32) for p in added], CYAN)
        cv2.addWeighted(layer, 0.7, image, 0.3, 0, image)

    thick = max(2, (x2 - x1) // 60)
    for b0, b1 in col.get("fracture_breaks") or []:
        y = y1 + (int(b0) + int(b1)) // 2
        cv2.line(image, (x1, y), (x2, y), RED_LINE, thick)
