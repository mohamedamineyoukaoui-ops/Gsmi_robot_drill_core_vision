"""
Train the lithology head on frozen DINOv2 features, evaluate it, export it.

    1. embed every train / val image with DINOv2 (frozen, no gradient)
    2. fit a multinomial logistic regression for C in {0.1, 0.3, 1, 3, 10},
       class_weight="balanced"; keep the C with the best validation macro-F1
    3. report accuracy and macro-F1 on the held-out test set
    4. export the head as plain arrays (coef, intercept) + the backbone weights,
       so the pipeline needs no scikit-learn at inference

    python train.py      (expects data/dataset_A/{train,val,test}/<class>/*.jpg)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from features import MODEL_NAME, embed_bgr, get_backbone

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset_A"
RUNS = ROOT / "runs"
# Variant A: 6 classes. Phosphate is left to the XRF (P2O5 is unambiguous chemically).
CLASSES = ["limestone", "dolomitic_limestone", "marl", "flint", "out_of_series", "no_rock"]


def split_files(split: str) -> tuple[list[Path], np.ndarray]:
    files, y = [], []
    for i, c in enumerate(CLASSES):
        for f in sorted((DATA / split / c).glob("*.jpg")):
            files.append(f)
            y.append(i)
    return files, np.array(y)


def read(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Model-selection criterion: every class counts the same, whatever its size."""
    f = []
    for i in range(len(CLASSES)):
        tp = np.sum((p == i) & (y == i))
        f.append(2 * tp / max(1, 2 * tp + np.sum((p == i) & (y != i)) + np.sum((p != i) & (y == i))))
    return float(np.mean(f))


def main() -> None:
    ftr, ytr = split_files("train")
    fva, yva = split_files("val")
    fte, yte = split_files("test")
    Xtr = embed_bgr([read(f) for f in ftr])
    Xva = embed_bgr([read(f) for f in fva])

    best, sweep = None, []
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        clf = LogisticRegression(max_iter=5000, C=C, class_weight="balanced").fit(Xtr, ytr)
        score = macro_f1(yva, clf.predict(Xva))
        sweep.append({"C": C, "val_macro_f1": round(score, 4)})
        print(f"  C={C}: val macro-F1 {score * 100:.1f}%")
        if best is None or score > best[0]:
            best = (score, C, clf)
    score, C, clf = best

    Xte = embed_bgr([read(f) for f in fte])
    pred = clf.predict(Xte)
    test = {"accuracy": float((pred == yte).mean()), "macro_f1": macro_f1(yte, pred)}
    print(f"best C={C}  val macro-F1 {score * 100:.1f}%  test accuracy {test['accuracy'] * 100:.1f}%  "
          f"test macro-F1 {test['macro_f1'] * 100:.1f}%")

    RUNS.mkdir(exist_ok=True)
    joblib.dump(clf, RUNS / "model.joblib")
    (RUNS / "info.json").write_text(json.dumps({"C": C, "val_macro_f1": score, "c_sweep": sweep, "test": test}),
                                    encoding="utf-8")

    # Export for the pipeline: plain arrays, no scikit-learn needed at inference.
    np.savez(RUNS / "dinov2_head_A.npz", coef=clf.coef_.astype(np.float32),
             intercept=clf.intercept_.astype(np.float32), classes=np.array([CLASSES[int(i)] for i in clf.classes_]))
    torch.save(get_backbone().state_dict(), RUNS / "dinov2_vits14_224.pt")
    print(f"exported {RUNS / 'dinov2_head_A.npz'} and the {MODEL_NAME} backbone")


if __name__ == "__main__":
    main()
