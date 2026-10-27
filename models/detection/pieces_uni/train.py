"""
Train the YOLO11 piece detector: one box per intact rock piece, pieces never overlapping.

1. Read the reviewed annotations (one JSON per tray: image path, size, boxes [x1, y1, x2, y2] in pixels).
2. Split 80 / 20 by physical tray, so the scans of one tray never sit on both sides.
3. Write the YOLO dataset: images/, labels/ (class xc yc w h, normalised) and dataset.yaml.
4. Train from COCO weights.

    python train.py      (expects completing annotation/data/unified/*.json)
"""

from ultralytics import YOLO
from pathlib import Path
import json
import random
import shutil


# data importation
path = Path(__file__).resolve().parent / "completing annotation" / "data" / "unified"

# a loop to select only reviewed , usable data
usable = []
for file in sorted(path.glob("*.json")):
    record = json.loads(file.read_text(encoding= "utf-8"))
    if record["review"] in ("accepted", "corrected") :
        usable.append(record)

# check for the amount of usable datra

print(len(usable))
print(sum(len(record["boxes"]) for record in usable))

# giving id's so the split wont leak data

for record in usable :
    name = record["image"]
    parts = name.split("_")
    the_id = "_".join(parts[:2])
    record["id"] = the_id



# splitting to train val , using the ids

all_ids = sorted({r["id"] for r in usable})
random.seed(42)
random.shuffle(all_ids)
n_val = round(0.2 * len(all_ids))
val_ids = set(all_ids[:n_val])

#making a list of dictionnaries , for each train and validation

val_records = []
train_records = []
for r in usable :
    if r["id"] in val_ids :
        val_records.append(r)
    else :
        train_records.append(r)

# verify everything goes as planned

print(len(all_ids), len(val_ids))
print(len(val_records), len(train_records))
print({r["id"] for r in train_records} & {r["id"] for r in val_records})

# building the yolo folders : images/{train,val} + labels/{train,val} , to make it as suitable by YOLO

root = Path(__file__).resolve().parent / "data" / "yolo"
if root.exists() :
    shutil.rmtree(root)
for kind in ("images", "labels") :
    for split in ("train", "val") :
        (root / kind / split).mkdir(parents=True, exist_ok=True)

for split, records in [("train", train_records), ("val", val_records)] :
    for r in records :
        shutil.copy(r["image_path"], root / "images" / split / r["image"])
        label_file = root / "labels" / split / (Path(r["image"]).stem + ".txt")
        W = r["width"]
        H = r["height"]
        lines = []
        for box in r["boxes"] :
            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2 / W
            yc = (y1 + y2) / 2 / H
            w  = (x2 - x1) / W
            h  = (y2 - y1) / H
            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        label_file.write_text("\n".join(lines), encoding="utf-8")

# the dataset.yaml : tells yolo where the data is and what class 0 is called

lines = [
    f"path: {root.as_posix()}",
    "train: images/train",
    "val: images/val",
    "names: {0: piece}",
]
(root / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")


# 1. MODEL SELECTION
model = YOLO("yolo11s.pt")

#2. TRAINING
results = model.train(
    data= str(root / "dataset.yaml" ) ,
    optimizer = "AdamW" ,
    lr0 = 0.001 ,
    seed = 42 ,
    workers = 0,
    epochs=200,
    imgsz=1024,
    batch=8,
    patience=50,
    device=0,
    flipud = 0.5,
    degrees = 0  ,
    close_mosaic = 20    ,
    project= str( Path(__file__).resolve().parent / "runs") , name="pieces_uni",   # where results are saved
)
