"""
Draw every figure of the model READMEs from the logged training histories
(results/results.csv) and the evaluation outputs (results/metrics.json).

    python tools/make_figures.py

Values are plotted raw, without smoothing. On training curves, the dashed line
marks the epoch whose weights were kept; on evaluation curves, the threshold
the pipeline uses.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1] / "models"

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#2a78d6", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load(folder: Path, name: str):
    p = folder / "results" / name
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p).rename(columns=str.strip)


# ── training curves ─────────────────────────────────────────────────────────

def yolo_best_epoch(df: pd.DataFrame, task: str) -> int:
    """The epoch Ultralytics keeps as best.pt: max fitness."""
    if task == "detect":
        fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    else:
        fitness = df["metrics/accuracy_top1"]
    return int(df.loc[fitness.idxmax(), "epoch"])


def training_curves(folder: Path, title: str, losses, metrics, best: int, metric_label: str) -> None:
    df = load(folder, "results.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for (col, label), color in zip(losses, SERIES):
        ax1.plot(df["epoch"], df[col], color=color, label=label)
    ax1.set(title="Loss", xlabel="epoch")
    for (col, label), color in zip(metrics, [SERIES[0], SERIES[2]]):
        ax2.plot(df["epoch"], df[col], color=color, label=label)
        y = df.loc[df.epoch == best, col].iloc[0]
        ax2.annotate(f"{y:.3f}", (best, y), xytext=(6, -12), textcoords="offset points", color=INK_2, fontsize=9)
    ax2.set(title=metric_label, xlabel="epoch", ylim=(0, 1.02))
    for ax in (ax1, ax2):
        ax.axvline(best, color=MUTED, linestyle="--", linewidth=1)
        ax.legend(loc="best")
    ax2.text(best, 0.03, f" kept: epoch {best}", color=MUTED, fontsize=9)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12)
    save(fig, folder / "results" / "training_curves.png")


# ── evaluation figures ──────────────────────────────────────────────────────

def detector_curves(folder: Path, title: str) -> None:
    m = load(folder, "metrics.json")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    titles = ["Precision vs confidence", "Recall vs confidence", "F1 vs confidence", "Precision vs recall"]
    keys = ["precision_vs_conf", "recall_vs_conf", "f1_vs_conf", "precision_vs_recall"]
    for ax, t, key in zip(axes, titles, keys):
        for (name, c), color in zip(m["per_class"].items(), SERIES):
            x = c["curves"]["x"]
            ax.plot(x, c["curves"][key], color=color, label=f"{name} (AP@0.5 {c['mAP50']:.3f})" if key == keys[-1] else name)
        ax.set(title=t, xlim=(0, 1), ylim=(0, 1.02), xlabel="recall" if key == keys[-1] else "confidence")
        if key != keys[-1]:
            conf = next(iter(m["per_class"].values()))["at_pipeline_conf"]["conf"]
            ax.axvline(conf, color=MUTED, linestyle="--", linewidth=1)
            ax.text(conf, 0.04, f" pipeline: {conf}", color=MUTED, fontsize=8)
    axes[-1].legend(loc="lower left", fontsize=8)
    if len(m["per_class"]) > 1:
        axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{title}: validation set", color=INK, x=0.01, ha="left", fontsize=12)
    save(fig, folder / "results" / "validation_curves.png")


def confusion(cm, labels, title: str, path: Path, size=(6.0, 5.0), fontsize=10) -> None:
    cm = np.array(cm, float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=size)
    ax.grid(False)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=fontsize,
                    color="#ffffff" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=8.5, rotation=45 if len(labels) > 6 else 0,
                  ha="right" if len(labels) > 6 else "center")
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.set(xlabel="predicted", ylabel="true")
    ax.set_title(title, loc="left")
    save(fig, path)


def zones_figures(folder: Path) -> None:
    m = load(folder, "metrics.json")
    confusion(m["confusion_rows_true_cols_predicted"], list(m["per_class"]),
              f"Zone classifier, validation (n={m['n']})\naccuracy {m['accuracy'] * 100:.1f} %, "
              f"macro-F1 {m['macro_f1'] * 100:.1f} %", folder / "results" / "confusion_matrix.png", size=(5.2, 4.6))


def cracks_figures(folder: Path) -> None:
    """Both runs: training loss and the piece score every 5 epochs; then piece F1 and
    the share of the outline the contour method closes, against the threshold."""
    m = load(folder, "metrics.json")
    runs = {"run 1 (150-epoch schedule, stopped at 65)": "results_run1.csv", "run 2 (80 epochs)": "results_run2.csv"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.6))
    for (label, name), color in zip(runs.items(), SERIES):
        df = load(folder, name)
        ax1.plot(df["epoch"], df["train/loss"], color=color, label=label)
        ev = df.dropna(subset=["metrics/piece_f1"])
        ax2.plot(ev["epoch"], ev["metrics/piece_f1"], color=color, marker="o", markersize=3, label=label)
        ax3.plot(ev["epoch"], ev["metrics/closing_share"] * 100, color=color, marker="o", markersize=3, label=label)
        best = ev.loc[ev["metrics/piece_f1"].idxmax()]
        ax2.annotate(f"{best['metrics/piece_f1']:.3f}", (best["epoch"], best["metrics/piece_f1"]), xytext=(4, 6),
                     textcoords="offset points", color=INK_2, fontsize=8)
    ax1.set(title="Training loss (BCE + Dice + clDice)", xlabel="epoch")
    ax2.set(title="Validation pieces, F1 (depth IoU ≥ 0.5)", xlabel="epoch", ylim=(0.6, 0.8))
    ax3.set(title="Outline closed by the method (%)", xlabel="epoch", ylim=(0, 45))
    for ax in (ax1, ax2, ax3):
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Crack U-Net (ResNet34 encoder)", color=INK, x=0.01, ha="left", fontsize=12)
    save(fig, folder / "results" / "training_curves.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for (name, run), color in zip(m["runs"].items(), SERIES):
        t = [float(k) for k in run["by_threshold"]]
        v = list(run["by_threshold"].values())
        ax1.plot(t, [x["piece_f1"] for x in v], color=color, marker="o", markersize=3, label=name.replace("run", "run "))
        ax2.plot(t, [x["closing_share"] * 100 for x in v], color=color, marker="o", markersize=3, label=name.replace("run", "run "))
    thr = m["pipeline"]["threshold"]
    for ax in (ax1, ax2):
        ax.axvline(thr, color=MUTED, linestyle="--", linewidth=1)
        ax.legend(loc="best", fontsize=8)
    ax1.text(thr, 0.605, f" pipeline: {thr}", color=MUTED, fontsize=8)
    ax1.set(title="Piece F1 vs probability threshold", xlabel="threshold", ylim=(0.6, 0.8))
    ax2.set(title="Outline closed by the method (%) vs threshold", xlabel="threshold", ylim=(0, 45))
    fig.suptitle(f"Crack U-Net, whole validation trays ({m['trays']} trays, {m['pieces']} pieces)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    save(fig, folder / "results" / "threshold_sweep.png")


def lithology_figures(folder: Path) -> None:
    m, info = load(folder, "metrics.json"), load(folder, "info.json")
    names = [c.replace("dolomitic_limestone", "dolomitic\nlimestone").replace("out_of_series", "out of\nseries")
             .replace("no_rock", "no rock") for c in m["classes"]]
    confusion(m["confusion_rows_true_cols_predicted"], names,
              f"DINOv2 + logistic head, test set (n={m['n']})\naccuracy {m['accuracy'] * 100:.1f} %, "
              f"macro-F1 {m['macro_f1'] * 100:.1f} %", folder / "results" / "confusion_matrix.png", size=(6.4, 5.4))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.9))
    sweep = pd.DataFrame(info["c_sweep"])
    ax1.plot(sweep.C, sweep.val_macro_f1, color=SERIES[0], marker="o")
    ax1.set_xscale("log")
    ax1.set(title="Training: regularisation sweep", xlabel="C (inverse L2 strength)", ylabel="validation macro-F1")
    ax1.annotate(f"chosen C={info['C']:g}", (sweep.C.iloc[-1], sweep.val_macro_f1.iloc[-1]), xytext=(-70, -18),
                 textcoords="offset points", color=INK_2, fontsize=9)

    rel = pd.DataFrame(m["reliability"])
    ax2.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="perfect calibration")
    ax2.plot(rel.confidence, rel.accuracy, color=SERIES[0], marker="o", label=f"model (ECE {m['ece'] * 100:.1f} %)")
    for _, r in rel.iterrows():
        ax2.annotate(str(int(r.n)), (r.confidence, r.accuracy), xytext=(4, -10), textcoords="offset points",
                     color=MUTED, fontsize=7)
    ax2.set(title="Calibration (test)", xlabel="mean confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1.02))
    ax2.legend(loc="upper left", fontsize=8)

    for (c, curve), color in zip(m["pr_curves"].items(), SERIES):
        ax3.plot(curve["recall"], curve["precision"], color=color,
                 label=f"{c.replace('_', ' ')} (AP {m['per_class'][c]['average_precision']:.2f})")
    ax3.set(title="Precision vs recall per class (test)", xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
    ax3.legend(loc="lower left", fontsize=7.5)
    save(fig, folder / "results" / "evaluation_curves.png")


def xrf_figures(folder: Path) -> None:
    m = load(folder, "metrics.json")
    imp = pd.Series(m["feature_importance"]).iloc[:12][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    ax.set(title="Random forest: 12 most important features", xlabel="mean decrease in impurity")
    ax.grid(axis="y", visible=False)
    save(fig, folder / "results" / "feature_importance.png")

    cv = m["cross_validation"]["benguerir (held-out samples)"]
    labels = [c.replace("_", " ") for c in cv["classes"]]
    confusion(cv["confusion_rows_true_cols_predicted"], labels,
              f"Held-out Benguerir samples (n={cv['n']})\naccuracy {cv['accuracy'] * 100:.1f} %, "
              f"macro-F1 {cv['macro_f1'] * 100:.1f} %", folder / "results" / "confusion_matrix.png",
              size=(7.2, 6.2), fontsize=8.5)


if __name__ == "__main__":
    det, seg, cls = ROOT / "detection", ROOT / "segmentation", ROOT / "classification"
    for name, title in (("columns_uni", "Column detector (YOLO11s)"), ("pieces_uni", "Piece detector (YOLO11s)")):
        df = load(det / name, "results.csv")
        training_curves(det / name, title, [("train/box_loss", "train box loss"), ("val/box_loss", "val box loss")],
                        [("metrics/mAP50(B)", "mAP@0.5"), ("metrics/mAP50-95(B)", "mAP@0.5:0.95")],
                        yolo_best_epoch(df, "detect"), "Validation mAP")
        detector_curves(det / name, title)

    zones = cls / "zones-yolov8"
    training_curves(zones, "Zone classifier (YOLOv8n-cls)", [("train/loss", "train"), ("val/loss", "validation")],
                    [("metrics/accuracy_top1", "top-1 accuracy")], yolo_best_epoch(load(zones, "results.csv"), "classify"),
                    "Validation accuracy")
    zones_figures(zones)

    cracks_figures(seg / "cracks_uni")

    lithology_figures(cls / "lithology-dinov2")
    xrf_figures(cls / "xrf-randomforest")
    print("figures written")
