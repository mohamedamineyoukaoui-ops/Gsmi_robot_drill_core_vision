"""
Evaluate the exported model (frozen DINOv2 + logistic head) on the test set.

Writes results/metrics.json: accuracy, macro-F1, per-class precision / recall /
F1 / average precision, one-vs-rest ROC-AUC, the confusion matrix, accuracy by
data source, calibration (ECE and the reliability diagram), and the
precision-recall curve of every class.

    python evaluate.py --weights runs        (folder with dinov2_vits14_224.pt and dinov2_head_A.npz)
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import train
from features import IMG, MODEL_NAME
from train import CLASSES, read, split_files

HERE = Path(__file__).resolve().parent
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def probabilities(files: list[Path], weights: Path) -> np.ndarray:
    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, img_size=IMG)
    bb.load_state_dict(torch.load(weights / "dinov2_vits14_224.pt", map_location="cpu"))
    bb = bb.eval().to(device)
    head = np.load(weights / "dinov2_head_A.npz")
    feats = []
    for i in range(0, len(files), 64):
        x = np.stack([cv2.resize(read(f), (IMG, IMG), interpolation=cv2.INTER_AREA)[:, :, ::-1] for f in files[i:i + 64]])
        t = (torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD
        feats.append(torch.nn.functional.normalize(bb(t.to(device)), dim=1).cpu().numpy())
    logits = np.concatenate(feats) @ head["coef"].T + head["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def reliability(P: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error and the per-bin table (confidence vs accuracy)."""
    conf, pred = P.max(1), P.argmax(1)
    ece, table = 0.0, []
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            acc = float(np.mean(pred[m] == y[m]))
            ece += m.mean() * abs(acc - conf[m].mean())
            table.append({"bin": f"{lo:.1f}-{lo + 0.1:.1f}", "n": int(m.sum()),
                          "confidence": round(float(conf[m].mean()), 3), "accuracy": round(acc, 3)})
    return float(ece), table


def main(weights: Path) -> dict:
    files, y = split_files("test")
    P = probabilities(files, weights)
    pred = P.argmax(1)
    k = len(CLASSES)

    cm = np.zeros((k, k), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    precision, recall = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    onehot = np.eye(k)[y]
    ece, table = reliability(P, y)

    per_class, pr_curves = {}, {}
    for i, c in enumerate(CLASSES):
        per_class[c] = {"support": int(cm[i].sum()), "precision": round(float(precision[i]), 4),
                        "recall": round(float(recall[i]), 4), "f1": round(float(f1[i]), 4),
                        "average_precision": round(float(average_precision_score(onehot[:, i], P[:, i])), 4)}
        pr, rc, _ = precision_recall_curve(onehot[:, i], P[:, i])
        keep = np.linspace(0, len(rc) - 1, min(120, len(rc))).round().astype(int)
        pr_curves[c] = {"recall": rc[keep].round(4).tolist(), "precision": pr[keep].round(4).tolist()}

    sources = np.array([re.sub(r"_\d+$", "", f.stem) for f in files])
    by_source = {s: {"n": int((sources == s).sum()), "accuracy": round(float((pred[sources == s] == y[sources == s]).mean()), 4)}
                 for s in sorted(set(sources))}

    return {"split": "test", "n": int(len(y)), "classes": CLASSES,
            "accuracy": round(float((pred == y).mean()), 4), "macro_f1": round(float(f1.mean()), 4),
            "roc_auc_macro_ovr": round(float(roc_auc_score(onehot, P, average="macro")), 4),
            "ece": round(ece, 4), "reliability": table,
            "per_class": per_class, "confusion_rows_true_cols_predicted": cm.tolist(),
            "accuracy_by_source": by_source, "pr_curves": pr_curves}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HERE / "runs")
    ap.add_argument("--data", type=Path, default=train.DATA, help="dataset folder with train/val/test")
    args = ap.parse_args()
    train.DATA = args.data
    out = main(args.weights)
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("pr_curves", "reliability")}, indent=1))
