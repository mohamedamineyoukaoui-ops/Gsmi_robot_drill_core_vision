"""
Evaluate the trained piece detector on the validation split.

Writes results/metrics.json:
  - box metrics: precision, recall and mAP, and the precision / recall / F1
    curves against the confidence threshold;
  - piece metrics, which is what RQD uses: the columns are found by the column
    detector, the boxes are assigned to them and made non-overlapping
    (pipeline/pieces.py), and each predicted piece is matched to an annotated
    one when their depth intervals have IoU >= 0.5. Also per column: the
    error in the number of pieces, and the share of depth counted twice.

    python evaluate.py --weights runs/pieces_uni/weights/best.pt --data data/yolo/dataset.yaml
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
IMGSZ = 1024                            # training image size
PIPELINE_CONF = 0.2                     # confidence threshold used in pipeline/pieces.py


def box_metrics(model: YOLO, data: Path, conf_used: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        m = model.val(data=str(data), split="val", imgsz=IMGSZ, batch=8, conf=0.001, iou=0.7,
                      plots=False, project=tmp, name="val", verbose=False, workers=0)
    box = m.box
    grid = np.asarray(box.px)
    k = int(np.argmin(np.abs(grid - conf_used)))
    pick = np.linspace(0, len(grid) - 1, 101).round().astype(int)
    per_class = {}
    for j, c in enumerate(box.ap_class_index):
        per_class[m.names[int(c)]] = {
            "instances": int(m.nt_per_class[int(c)]),
            "precision": round(float(box.p[j]), 4), "recall": round(float(box.r[j]), 4),
            "mAP50": round(float(box.all_ap[j, 0]), 4), "mAP50_95": round(float(box.all_ap[j].mean()), 4),
            "at_pipeline_conf": {"conf": conf_used, "precision": round(float(box.p_curve[j][k]), 4),
                                 "recall": round(float(box.r_curve[j][k]), 4), "f1": round(float(box.f1_curve[j][k]), 4)},
            "curves": {"x": grid[pick].round(3).tolist(),
                       "precision_vs_conf": box.p_curve[j][pick].round(4).tolist(),
                       "recall_vs_conf": box.r_curve[j][pick].round(4).tolist(),
                       "f1_vs_conf": box.f1_curve[j][pick].round(4).tolist(),
                       "precision_vs_recall": np.asarray(box.prec_values)[j][pick].round(4).tolist()},
        }
    return {"overall": {"precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
                        "mAP50": round(float(box.map50), 4), "mAP50_95": round(float(box.map), 4)},
            "per_class": per_class}


def read(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def truth_boxes(label: Path, w: int, h: int) -> list[dict]:
    out = []
    for line in label.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append({"x1": int((cx - bw / 2) * w), "x2": int((cx + bw / 2) * w),
                        "y1": int((cy - bh / 2) * h), "y2": int((cy + bh / 2) * h), "confidence": 1.0})
    return out


def interval_iou(a: dict, b: dict) -> float:
    inter = max(0, min(a["y_end"], b["y_end"]) - max(a["y_start"], b["y_start"]))
    union = (a["y_end"] - a["y_start"]) + (b["y_end"] - b["y_start"]) - inter
    return inter / max(1, union)


def match(pred: list[dict], true: list[dict]) -> int:
    """Greedy one-to-one matching by interval IoU >= 0.5, best pairs first."""
    pairs = sorted(((interval_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)), reverse=True)
    used_p, used_t, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); tp += 1
    return tp


def double_counted(pieces: list[dict], h: int) -> tuple[int, int]:
    cover = np.zeros(h + 1, int)
    for p in pieces:
        cover[p["y_start"]:p["y_end"]] += 1
    return int((cover >= 2).sum()), int(sum(p["y_end"] - p["y_start"] for p in pieces))


def piece_metrics(data: Path, weights_dir: Path, confs: list[float]) -> dict:
    os.environ["GSMI_MODELS_DIR"] = str(weights_dir)
    from pipeline import pieces as P
    from pipeline.columns import detect_columns

    images = sorted((data.parent / "images" / "val").iterdir())
    cache = []
    for img_path in images:
        img = read(img_path)
        h, w = img.shape[:2]
        truth = truth_boxes(data.parent / "labels" / "val" / (img_path.stem + ".txt"), w, h)
        boxes = P.detect_pieces(img, conf=min(confs), imgsz=IMGSZ)
        cache.append((detect_columns(img), truth, boxes))

    out = {}
    for conf in confs:
        s = {"tp": 0, "pred": 0, "true": 0, "double_px": 0, "length_px": 0, "count_err": []}
        for cols, truth, boxes in cache:
            kept = [b for b in boxes if b["confidence"] >= conf]
            for col in cols:
                col_h = col["y_end"] - col["y_start"]
                t = P.boxes_in_column(truth, col, 1.0)
                pred = P.pieces_from_boxes(kept, col, 1.0)
                s["tp"] += match(pred, t); s["pred"] += len(pred); s["true"] += len(t)
                d, length = double_counted(pred, col_h)
                s["double_px"] += d; s["length_px"] += length
                s["count_err"].append(abs(len(pred) - len(t)))
        p, r = s["tp"] / max(1, s["pred"]), s["tp"] / max(1, s["true"])
        out[str(conf)] = {"predicted_pieces": s["pred"], "annotated_pieces": s["true"], "matched": s["tp"],
                          "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(1e-9, p + r), 4),
                          "double_counted_depth_share": round(s["double_px"] / max(1, s["length_px"]), 4),
                          "mean_piece_count_error_per_column": round(float(np.mean(s["count_err"])), 3)}
    return {"match": "interval IoU >= 0.5", "columns": "column detector", "by_conf": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HERE / "runs" / "pieces_uni" / "weights" / "best.pt")
    ap.add_argument("--data", type=Path, default=HERE / "data" / "yolo" / "dataset.yaml")
    ap.add_argument("--models-dir", type=Path, required=True,
                    help="folder with piece_yolo11s.pt and column_yolo11s.pt, as the pipeline expects")
    args = ap.parse_args()
    model = YOLO(str(args.weights))
    out = {"split": "val", **box_metrics(model, args.data, PIPELINE_CONF),
           "pieces": piece_metrics(args.data, args.models_dir, [0.1, 0.2, 0.35])}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"overall": out["overall"], "pieces": out["pieces"]["by_conf"]}, indent=1))
