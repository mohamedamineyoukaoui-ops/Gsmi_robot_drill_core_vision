"""
Evaluate the trained crack U-Nets on the validation trays.

For each run and each probability threshold, the pipeline's contour method is
run on the predicted masks of the whole validation trays (at 10 px/cm) and its
pieces are matched to the pieces of the checked masks (depth IoU >= 0.5).
Also measured: the share of the final outlines the method had to close itself
(the lower, the more of each outline the U-Net drew), and pixel F1.

Writes results/metrics.json.

    python evaluate.py --runs run1=runs/cracks_uni_run1/best_unet.pth run2=runs/cracks_uni/best_unet.pth
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
PIPELINE = {"run": "run2", "threshold": 0.45}     # what pipeline/cracks.py uses


def load(weights: Path, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[f"run1={HERE / 'runs' / 'cracks_uni_run1' / 'best_unet.pth'}",
                                                 f"run2={HERE / 'runs' / 'cracks_uni' / 'best_unet.pth'}"])
    args = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        import train as T                          # the same split, pieces and scoring as during training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_trays = []
    for r in T.val_records:
        rgb, mask, factor = T.load_tray(r, T.path, T.WORK_PX_PER_CM)
        h, w = mask.shape
        columns = [[max(0, round(c[0] * factor)), max(0, round(c[1] * factor)),
                    min(w, round(c[2] * factor)), min(h, round(c[3] * factor))] for c in r["auto"]["columns"]]
        val_trays.append((rgb, mask, columns, T.true_pieces(mask, columns)))

    out = {"split": "val", "trays": len(val_trays), "pieces": sum(len(p) for v in val_trays for p in v[3]),
           "work_px_per_cm": T.WORK_PX_PER_CM, "match": "depth-interval IoU >= 0.5", "pipeline": PIPELINE, "runs": {}}
    for spec in args.runs:
        name, weights = spec.split("=", 1)
        model = load(Path(weights), device)
        sweep = {}
        for t in THRESHOLDS:
            T.THRESHOLD = t
            sweep[str(t)] = {k: round(v, 4) for k, v in T.validate(model, val_trays, device).items()}
            print(name, t, sweep[str(t)])
        out["runs"][name] = {"weights": Path(weights).name, "by_threshold": sweep}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
