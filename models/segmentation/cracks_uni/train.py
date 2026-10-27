"""
Train the crack U-Net (ResNet34 encoder) on the reviewed crack masks.

1. Read the reviewed masks (one JSON per tray: image, mask, columns) and the scale of each tray.
2. Split 80 / 20 by physical tray; your own masks and the auto ones are mixed in both sets.
3. Train on 512 x 512 crops at 10 px/cm (±25 %), loss BCE + Dice + clDice (dataset.py, model.py).
4. Every 5 epochs, run the pipeline's contour method on the whole validation trays and score its pieces
   against the pieces of the checked masks (depth IoU >= 0.5); keep the epoch with the best piece F1,
   and log the share of the outline the method had to close. 80 epochs, one learning-rate cycle.

    python train.py      (expects completing annotation/data/unified/*.json)
"""

from pathlib import Path
import json
import random
import re


WORK_PX_PER_CM = 10.0          # every tray is brought to this resolution: the median of the training trays


# data importation
path = Path(__file__).resolve().parent / "completing annotation" / "data" / "unified"

# a loop to select only reviewed , usable data
usable = []
for file in sorted(path.glob("*.json")):
    record = json.loads(file.read_text(encoding="utf-8"))
    if record["review"] in ("accepted", "corrected") and record["auto"]["columns"]:
        usable.append(record)

# the scale of each tray : core length in pixels / drilled run in cm (written in the name)

for record in usable :
    name = Path(record["image"]).stem
    m = re.search(r"_(\d+)m(\d+)_(\d+)m(\d+)", name)
    run_cm = (int(m.group(3)) + int(m.group(4)) / 100 - int(m.group(1)) - int(m.group(2)) / 100) * 100
    record["px_per_cm"] = sum(c[3] - c[1] for c in record["auto"]["columns"]) / run_cm

# giving id's so the split wont leak data

for record in usable :
    parts = Path(record["image"]).stem.split("_")
    record["id"] = "_".join(parts[:2])

# splitting to train val , using the ids : 20 % of the physical trays, your masks and the auto ones mixed
# (one tray can have a band in each source, so the ids are drawn from the whole pool)

all_ids = sorted({r["id"] for r in usable})
random.seed(42)
random.shuffle(all_ids)
val_ids = set(all_ids[:round(0.2 * len(all_ids))])

val_records = [r for r in usable if r["id"] in val_ids]
train_records = [r for r in usable if r["id"] not in val_ids]

# verify everything goes as planned

print(len(usable), "trays")
for source in ("manual", "auto_contour") :
    print(source, "train", sum(r["source"] == source for r in train_records), "val", sum(r["source"] == source for r in val_records))
print({r["id"] for r in train_records} & {r["id"] for r in val_records})


# ── the pieces of a validation tray, as RQD sees them ────────────────────────

import csv
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline import pieces as P                       # noqa: E402  (the pipeline's own contour method)
from dataset import CrackCrops, MEAN, STD, load_tray     # noqa: E402
from model import CrackLoss, create_model                # noqa: E402

THRESHOLD = 0.45                # as in the pipeline; re-chosen on validation in evaluate.py
MIN_AREA = 0.0002               # an enclosed area smaller than 0.02 % of the tray is a gap in a stroke


def stack(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """One list of pieces along the core: the same depth twice becomes one piece, a small overlap is cut."""
    out = []
    for a, b in sorted(intervals):
        if out and a < out[-1][1]:
            pa, pb = out[-1]
            if pb - a >= 0.5 * min(pb - pa, b - a):
                out[-1] = (pa, max(pb, b))
                continue
            out[-1], a = (pa, (pb + a) // 2), (pb + a) // 2
        out.append((a, b))
    return out


def true_pieces(mask: np.ndarray, columns: list[list[int]]) -> list[list[tuple[int, int]]]:
    """Pieces of each column from a checked mask: every area closed by an outline (as for the piece labels)."""
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), connectivity=4)
    per_column = [[] for _ in columns]
    for x, y, bw, bh, area in stats[1:]:
        if x == 0 or y == 0 or x + bw == w or y + bh == h or area < MIN_AREA * h * w:
            continue
        overlaps = [min(x + bw, c[2]) - max(x, c[0]) for c in columns]
        if max(overlaps) > 0:
            k = overlaps.index(max(overlaps))
            per_column[k].append((max(0, y - columns[k][1]), min(columns[k][3], y + bh) - columns[k][1]))
    return [stack(p) for p in per_column]


def predicted_pieces(pred: np.ndarray, rgb: np.ndarray, columns: list[list[int]]) -> tuple[list, int, int]:
    """Pieces of each column from the predicted mask with the pipeline's contour method,
    and how many outline pixels the method had to add to close them (its cyan)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out, added_px, outline_px = [], 0, 0
    for x1, y1, x2, y2 in columns:
        crack, col = pred[y1:y2, x1:x2], bgr[y1:y2, x1:x2]
        span = P.core_span(y2 - y1, [], crack > 0)
        found = P.extract_pieces_contour(crack, col, span, WORK_PX_PER_CM)
        out.append(stack([(p["y_start"], p["y_end"]) for p in found]))
        # the closing the method adds (same steps as extract_pieces_contour)
        m = crack > 0
        sealed = m.astype(np.uint8) * 255
        band = P.core_band(col, x2 - x1)
        top, bottom = span if span else (0, y2 - y1)
        t = max(2, (x2 - x1) // 60)
        cv2.rectangle(sealed, (max(0, band[0] - t), max(0, top - t)),
                      (min(x2 - x1 - 1, band[1] + t), min(y2 - y1 - 1, bottom + t)), 255, t)
        k = max(3, P.cm_to_px(P.CLOSE_GAP_CM, WORK_PX_PER_CM) * 2 + 1)
        closed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        added_px += int(((closed > 0) & ~m).sum())
        outline_px += int((closed > 0).sum())
    return out, added_px, outline_px


def match(pred: list[tuple[int, int]], true: list[tuple[int, int]]) -> int:
    """Pieces matched one to one by depth-interval IoU >= 0.5, best pairs first."""
    def iou(a, b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        return inter / max(1, (a[1] - a[0]) + (b[1] - b[0]) - inter)
    pairs = sorted(((iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)), reverse=True)
    used_p, used_t, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); tp += 1
    return tp


@torch.no_grad()
def predict(model, rgb: np.ndarray, device) -> np.ndarray:
    """Crack probability of a whole tray (sides padded to a multiple of 32)."""
    h, w = rgb.shape[:2]
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    x = np.pad(x, ((0, (32 - h % 32) % 32), (0, (32 - w % 32) % 32), (0, 0)))
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(device))
    return torch.sigmoid(logits.float())[0, 0, :h, :w].cpu().numpy()


def validate(model, val_trays, device) -> dict:
    """Piece precision / recall / F1 against the checked masks, the share of the outline
    the contour method had to close, and pixel F1."""
    model.eval()
    tp = n_pred = n_true = added = outline = 0
    inter = pred_sum = true_sum = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for rgb, mask, columns, truth in val_trays:
        prob = predict(model, rgb, device)
        pred = (prob > THRESHOLD).astype(np.uint8) * 255
        pred = cv2.morphologyEx(cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
        found, a, o = predicted_pieces(pred, rgb, columns)
        added += a; outline += o
        for p, t in zip(found, truth):
            tp += match(p, t); n_pred += len(p); n_true += len(t)
        inter += int(((pred > 0) & (mask > 0)).sum()); pred_sum += int((pred > 0).sum()); true_sum += int((mask > 0).sum())
    precision, recall = tp / max(1, n_pred), tp / max(1, n_true)
    return {"piece_precision": precision, "piece_recall": recall,
            "piece_f1": 2 * precision * recall / max(1e-9, precision + recall),
            "closing_share": added / max(1, outline), "pixel_f1": 2 * inter / max(1, pred_sum + true_sum)}


# ── training ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    EPOCHS, BATCH_SIZE, LR = 80, 12, 3e-4
    EVAL_EVERY = 5                                     # score every 5 epochs; no early stopping: the learning rate
                                                       # must go through its whole cycle, the best epochs come as it decays
    SAVE_DIR = Path(__file__).resolve().parent / "runs" / "cracks_uni"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader = DataLoader(CrackCrops(train_records, path, WORK_PX_PER_CM), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)
    val_trays = []
    for r in val_records:                              # whole trays at exactly the working resolution
        rgb, mask, factor = load_tray(r, path, WORK_PX_PER_CM)
        h, w = mask.shape
        columns = [[max(0, round(c[0] * factor)), max(0, round(c[1] * factor)),
                    min(w, round(c[2] * factor)), min(h, round(c[3] * factor))] for c in r["auto"]["columns"]]
        val_trays.append((rgb, mask, columns, true_pieces(mask, columns)))
    print(len(train_loader.dataset), "training crops per epoch,", len(val_trays), "validation trays,",
          sum(len(p) for v in val_trays for p in v[3]), "validation pieces")

    model = create_model().to(device)
    criterion = CrackLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    # OneCycle: warm up to LR, then anneal to ~0 over the whole run (stepped every batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log = SAVE_DIR / "results.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train/loss", "lr", "metrics/piece_precision", "metrics/piece_recall",
                                "metrics/piece_f1", "metrics/closing_share", "metrics/pixel_f1"])
    best_f1, best_epoch = -1.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
            loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                     # clip the real gradients, not the scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        lr = optimizer.param_groups[0]["lr"]

        row = [epoch, f"{train_loss:.4f}", f"{lr:.6f}", "", "", "", "", ""]
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            v = validate(model, val_trays, device)
            row[3:] = [f"{v[k]:.4f}" for k in ("piece_precision", "piece_recall", "piece_f1", "closing_share", "pixel_f1")]
            print(f"epoch {epoch}: loss {train_loss:.4f} | pieces P {v['piece_precision']:.3f} R {v['piece_recall']:.3f} "
                  f"F1 {v['piece_f1']:.3f} | closing {v['closing_share'] * 100:.1f} % | pixel F1 {v['pixel_f1']:.3f}")
            if v["piece_f1"] > best_f1:
                best_f1, best_epoch = v["piece_f1"], epoch
                torch.save(model.state_dict(), SAVE_DIR / "best_unet.pth")
        else:
            print(f"epoch {epoch}: loss {train_loss:.4f} | lr {lr:.6f}")
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow(row)

    torch.save(model.state_dict(), SAVE_DIR / "last_unet.pth")
    print(f"best piece F1 {best_f1:.3f} at epoch {best_epoch} -> {SAVE_DIR / 'best_unet.pth'}")
