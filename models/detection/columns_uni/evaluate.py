"""
Evaluate the trained column detector on the validation split.

Writes results/metrics.json: precision, recall and mAP, the precision / recall /
F1 curves against the confidence threshold, the precision-recall curve, the
operating point the pipeline uses, and how far the predicted core ends (top and
bottom edges of each box) are from the annotated ones, in pixels.

    python evaluate.py --weights runs/columns_uni/weights/best.pt --data data/yolo/dataset.yaml
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
PIPELINE_CONF = 0.6                     # confidence threshold used in pipeline/columns.py
IMGSZ = 1024                            # training image size


def box_metrics(model: YOLO, data: Path, conf_used: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        m = model.val(data=str(data), split="val", imgsz=IMGSZ, batch=8, conf=0.001, iou=0.7,
                      plots=False, project=tmp, name="val", verbose=False, workers=0)
    box = m.box
    grid = np.asarray(box.px)                   # 1,000 points in [0, 1]: confidence, or recall for the PR curve
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


def core_end_errors(model: YOLO, data: Path, conf_used: float) -> dict:
    """Each annotated column is matched to the predicted box with the largest IoU;
    the top and bottom edges of the pair are compared in pixels and in % of the image height."""
    images = sorted((data.parent / "images" / "val").glob("*"))
    top, bottom, top_pct, bottom_pct, unmatched = [], [], [], [], 0
    for img in images:
        result = model(str(img), imgsz=IMGSZ, conf=conf_used, verbose=False)[0]
        h, w = result.orig_shape
        pred = result.boxes.xyxy.cpu().numpy()
        label = data.parent / "labels" / "val" / (img.stem + ".txt")
        for line in label.read_text().split("\n"):
            if not line.strip():
                continue
            _, xc, yc, bw, bh = map(float, line.split())
            t = np.array([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
            if len(pred) == 0:
                unmatched += 1
                continue
            ix = np.clip(np.minimum(pred[:, 2], t[2]) - np.maximum(pred[:, 0], t[0]), 0, None)
            iy = np.clip(np.minimum(pred[:, 3], t[3]) - np.maximum(pred[:, 1], t[1]), 0, None)
            inter = ix * iy
            union = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) + (t[2] - t[0]) * (t[3] - t[1]) - inter
            iou = inter / union
            if iou.max() < 0.5:
                unmatched += 1
                continue
            p = pred[int(iou.argmax())]
            top.append(abs(p[1] - t[1]))
            bottom.append(abs(p[3] - t[3]))
            top_pct.append(abs(p[1] - t[1]) / h * 100)
            bottom_pct.append(abs(p[3] - t[3]) / h * 100)

    def summary(values: list[float]) -> dict:
        v = np.asarray(values)
        return {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
                "p90": round(float(np.percentile(v, 90)), 2), "max": round(float(v.max()), 2)}

    return {"conf": conf_used, "columns_matched": len(top), "columns_unmatched": unmatched,
            "top_edge_px": summary(top), "bottom_edge_px": summary(bottom),
            "top_edge_pct_of_height": summary(top_pct), "bottom_edge_pct_of_height": summary(bottom_pct)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HERE / "runs" / "columns_uni" / "weights" / "best.pt")
    ap.add_argument("--data", type=Path, default=HERE / "data" / "yolo" / "dataset.yaml")
    args = ap.parse_args()
    model = YOLO(str(args.weights))
    out = {"split": "val", **box_metrics(model, args.data, PIPELINE_CONF),
           "core_ends": core_end_errors(model, args.data, PIPELINE_CONF)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"overall": out["overall"], "core_ends": out["core_ends"]}, indent=1))
