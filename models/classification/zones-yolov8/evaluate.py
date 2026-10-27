"""
Evaluate the zone classifier on the validation patches (real images only).

Writes results/metrics.json: accuracy, macro-F1, precision / recall / F1 per
class, the confusion matrix, and the binary "intact vs broken" scores that
matter for RQD.

    python evaluate.py --weights runs/zone_cls_v4/weights/best.pt --data data/zone_cls_v4/val
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
CLASSES = ["intact", "rubble", "sand", "empty"]


def scores(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def main(weights: Path, data: Path) -> dict:
    model = YOLO(str(weights), task="classify")
    y_true, y_pred = [], []
    for i, c in enumerate(CLASSES):
        files = sorted(p for p in (data / c).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        images = [cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR) for f in files]
        for r in model.predict(images, imgsz=128, verbose=False):
            y_true.append(i)
            y_pred.append(CLASSES.index(r.names[r.probs.top1]))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    cm = np.zeros((4, 4), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    s = scores(cm)
    # What RQD cares about: is the patch intact or not?
    bt, bp = y_true == 0, y_pred == 0
    return {
        "split": "val (real patches only)",
        "n": int(len(y_true)),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(float(s["f1"].mean()), 4),
        "per_class": {c: {"support": int(cm[i].sum()), "precision": round(float(s["precision"][i]), 4),
                          "recall": round(float(s["recall"][i]), 4), "f1": round(float(s["f1"][i]), 4)}
                      for i, c in enumerate(CLASSES)},
        "confusion_rows_true_cols_predicted": cm.tolist(),
        "intact_vs_broken": {
            "broken_called_intact": int((~bt & bp).sum()), "intact_called_broken": int((bt & ~bp).sum()),
            "intact_precision": round(float((bt & bp).sum() / max(bp.sum(), 1)), 4),
            "intact_recall": round(float((bt & bp).sum() / max(bt.sum(), 1)), 4),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HERE / "runs" / "zone_cls_v4" / "weights" / "best.pt")
    ap.add_argument("--data", type=Path, default=HERE / "data" / "zone_cls_v4" / "val")
    args = ap.parse_args()
    out = main(args.weights, args.data)
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
