"""
Train the XRF random forest ("A4") on real rock chemistry.

Data
  WORLD      data/ngdb_sedimentary.csv: 4,802 lab analyses from the USGS National
             Geochemical Database (public domain), labelled by rock name:
             limestone, dolomite, chert (flint), phosphorite, and 2,000 rocks
             foreign to the deposit (granite, basalt, sandstone, shale...).
  BENGUERIR  the published Benguerir lab-XRF mean +- SD per rock type
             (pipeline/lithology_fusion.py cites the papers), sampled 300 times per
             class with the noise of a handheld Niton analyser. The only source for
             marl, siliceous marl, phosphate flint and the phosphatic mixtures.

Features (identical to pipeline.lithology_fusion.a4_features)
  7 oxides -> closed composition (sums to 1, undetected = 65 % of the
  detection limit) -> all 21 pairwise log-ratios + a "MgO not measured" flag.
  Log-ratios because compositions are relative: only ratios between parts
  carry information (compositional data analysis, Aitchison).

Model
  RandomForestClassifier(200 trees, min_samples_leaf=2, class_weight="balanced")

Validation
  5-fold cross-validation that holds out whole US STATES, so near-identical
  analyses from one area never sit on both sides.

    python train.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from pipeline.lithology_fusion import A4_OXIDES, DETECTION_LIMIT_PCT, a4_features  # noqa: E402

WORLD = ROOT / "data" / "ngdb_sedimentary.csv"
OUT = ROOT / "runs"
N_BENGUERIR = 300
NITON_REL_ERR, NITON_MG_REL_ERR = 0.06, 0.15      # handheld XRF relative error, majors / MgO

REF_OXIDES = ("P2O5", "CaO", "SiO2", "Al2O3", "Fe2O3", "MgO")
# Published Benguerir lab XRF, mean and SD (wt %). Minerals 2026 16:619 (n = 49, 35, 23, 20)
# and Buildings 2024 14:2375 (3 samples each: SD taken as 25 % of the mean).
LITERATURE = {
    "marl": (dict(P2O5=5.2, CaO=32.4, SiO2=21.7, Al2O3=3.4, Fe2O3=2.1, MgO=1.5),
             dict(P2O5=3.9, CaO=8.6, SiO2=11.4, Al2O3=1.8, Fe2O3=1.0, MgO=0.6)),
    "limestone": (dict(P2O5=2.4, CaO=47.2, SiO2=9.8, Al2O3=1.4, Fe2O3=0.9, MgO=0.8),
                  dict(P2O5=1.8, CaO=5.3, SiO2=6.1, Al2O3=0.8, Fe2O3=0.4, MgO=0.3)),
    "phosphate": (dict(P2O5=18.6, CaO=44.8, SiO2=13.2, Al2O3=2.0, Fe2O3=1.3, MgO=0.6),
                  dict(P2O5=4.3, CaO=4.7, SiO2=7.0, Al2O3=1.0, Fe2O3=0.6, MgO=0.2)),
    "siliceous_marl": (dict(P2O5=4.7, CaO=16.5, SiO2=48.7, Al2O3=5.1, Fe2O3=2.6, MgO=1.0),
                       dict(P2O5=3.2, CaO=8.2, SiO2=12.3, Al2O3=2.0, Fe2O3=1.2, MgO=0.5)),
    "flint": (dict(P2O5=3.57, CaO=6.97, SiO2=81.8, Al2O3=0.27, Fe2O3=1.07, MgO=1.77), {}),
    "phosphate_flint": (dict(P2O5=13.7, CaO=25.7, SiO2=50.23, Al2O3=0.23, Fe2O3=0.43, MgO=1.7), {}),
    "dolomitic_limestone": (dict(P2O5=2.6, CaO=32.8, SiO2=6.97, Al2O3=0.4, Fe2O3=0.2, MgO=18.3), {}),
}
DERIVED = {"phosphatic_limestone": ("phosphate", "limestone"), "phosphatic_marl": ("phosphate", "marl")}


def literature_table() -> dict[str, tuple[dict, dict]]:
    table = {}
    for cls, (mean, sd) in LITERATURE.items():
        table[cls] = (mean, {ox: sd.get(ox, 0.25 * mean[ox]) for ox in REF_OXIDES})
    for cls, (a, b) in DERIVED.items():           # no published analysis: 50/50 of the end-members
        table[cls] = ({ox: 0.5 * (table[a][0][ox] + table[b][0][ox]) for ox in REF_OXIDES},
                      {ox: 0.5 * (table[a][1][ox] + table[b][1][ox]) for ox in REF_OXIDES})
    return table


def draw(rng, mean: dict, sd: dict, n: int) -> list[dict]:
    """Log-normal samples with the published mean and SD (concentrations stay positive)."""
    rows = []
    for _ in range(n):
        r = {}
        for ox in REF_OXIDES:
            m, s = mean[ox], sd[ox]
            sig = math.sqrt(math.log(1 + (s / m) ** 2))
            r[ox] = m * math.exp(rng.normal(-0.5 * sig * sig, sig))
        total = sum(r.values())
        if total > 95:                                 # leave room for LOI / F / CO2
            r = {k: v * 95 / total for k, v in r.items()}
        rows.append(r)
    return rows


def niton(rng, true: dict) -> dict:
    """What a handheld analyser would report: relative noise, then detection limits."""
    out = {}
    for ox, v in true.items():
        m = max(0.0, v * (1 + rng.normal(0, NITON_MG_REL_ERR if ox == "MgO" else NITON_REL_ERR)))
        out[ox] = m if m >= DETECTION_LIMIT_PCT.get(ox, 0.0) else float("nan")
    return out


def benguerir_samples(seed: int = 21) -> tuple[list[dict], list[str]]:
    rng = np.random.default_rng(seed)
    comps, labels = [], []
    for cls, (mean, sd) in literature_table().items():
        for r in draw(rng, mean, sd, N_BENGUERIR):
            comps.append(niton(rng, r))
            labels.append(cls)
    return comps, labels


def new_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                  n_jobs=-1, random_state=0)


def main() -> None:
    world = pd.read_csv(WORLD)
    world = world[world.target != "siliceous_marl"].reset_index(drop=True)
    w_comps = [{ox: float(r[ox]) for ox in A4_OXIDES if not pd.isna(r[ox])} for _, r in world.iterrows()]
    w_y = list(world.target)
    b_comps, b_y = benguerir_samples()
    Xw = np.stack([a4_features(c) for c in w_comps])
    Xb = np.stack([a4_features(c) for c in b_comps])
    print(f"WORLD {len(w_y)} analyses, BENGUERIR {len(b_y)} sampled readings")

    # 5-fold CV holding out whole US states (Benguerir samples split by index)
    states = world.state.fillna("?").astype(str).values
    uniq = np.array(sorted(set(states)))
    np.random.default_rng(0).shuffle(uniq)
    b_fold = np.arange(len(b_y)) % 5
    pred_w = np.empty(len(w_y), dtype=object)
    pred_b = np.empty(len(b_y), dtype=object)
    for k, fold_states in enumerate(np.array_split(uniq, 5)):
        w_test, b_test = np.isin(states, fold_states), b_fold == k
        rf = new_forest().fit(np.vstack([Xw[~w_test], Xb[~b_test]]),
                              np.array([y for y, t in zip(w_y, w_test) if not t] + [y for y, t in zip(b_y, b_test) if not t]))
        pred_w[w_test] = rf.predict(Xw[w_test])
        pred_b[b_test] = rf.predict(Xb[b_test])
    print(f"held-out accuracy: WORLD {np.mean(pred_w == np.array(w_y)) * 100:.1f} %, "
          f"BENGUERIR {np.mean(pred_b == np.array(b_y)) * 100:.1f} %")

    # Final model on everything
    final = new_forest().fit(np.vstack([Xw, Xb]), np.array(w_y + b_y))
    OUT.mkdir(exist_ok=True)
    joblib.dump(dict(model=final, oxides=A4_OXIDES, use_traces=False), OUT / "xrf_a4_world.joblib")
    print(f"saved {OUT / 'xrf_a4_world.joblib'}")

    names = [f"ln({A4_OXIDES[i]}/{A4_OXIDES[j]})" for i in range(len(A4_OXIDES)) for j in range(i + 1, len(A4_OXIDES))]
    report = {"cross_validation": {"world (held-out US states)": cv_report(w_y, pred_w),
                                   "benguerir (held-out samples)": cv_report(b_y, pred_b)},
              "feature_importance": dict(sorted(zip(names + ["MgO not measured"], final.feature_importances_.round(4).tolist()),
                                                key=lambda kv: -kv[1]))}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def cv_report(y_true: list[str], y_pred) -> dict:
    """Accuracy, macro-F1, per-class precision / recall / F1 and the confusion matrix."""
    y_true, y_pred = np.array(y_true), np.array(y_pred, dtype=str)
    classes = sorted(set(y_true) | set(y_pred))
    cm = np.array([[int(((y_true == t) & (y_pred == p)).sum()) for p in classes] for t in classes])
    tp = np.diag(cm).astype(float)
    precision, recall = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    present = cm.sum(1) > 0                                   # classes that exist in this subset
    return {"n": int(len(y_true)), "accuracy": round(float((y_true == y_pred).mean()), 4),
            "macro_f1": round(float(f1[present].mean()), 4), "classes": classes,
            "per_class": {c: {"support": int(cm[i].sum()), "precision": round(float(precision[i]), 4),
                              "recall": round(float(recall[i]), 4), "f1": round(float(f1[i]), 4)}
                          for i, c in enumerate(classes) if present[i]},
            "confusion_rows_true_cols_predicted": cm.tolist()}


if __name__ == "__main__":
    main()
