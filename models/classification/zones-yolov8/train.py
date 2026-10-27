"""
Train the YOLOv8n-cls zone classifier: intact / rubble / sand / empty.

Domain rule behind the settings:
  - sand <-> rubble confusion is tolerable (both are broken material);
  - sand or rubble <-> intact confusion is critical (it breaks RQD).

Strategy:
  1. Offline augmentation of the minority classes up to 60 % of the majority
     class (sand had very few real patches). Validation keeps real images only.
  2. 150 epochs, no early stopping.
  3. Label smoothing + mixup against over-confidence.
  4. Strong brightness/contrast augmentation: sand <-> intact confusions come
     from similar smooth textures under different lighting.

    python train.py      (expects data/zones/{intact,rubble,sand,empty}/*.jpg, 128x128 patches)
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
LABELED_DIR = SCRIPT_DIR / "data" / "zones"
DATASET_DIR = SCRIPT_DIR / "data" / "zone_cls_v4"
MODEL_SAVE = SCRIPT_DIR / "runs" / "zone_cls_v4"

TRAIN_RATIO = 0.85
CLASSES = ["intact", "rubble", "sand", "empty"]


# ── offline augmentation ────────────────────────────────────────────────────

def augment_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_contrast(img, alpha):
    mean = np.mean(img)
    return np.clip(alpha * (img.astype(np.float32) - mean) + mean, 0, 255).astype(np.uint8)


def augment_gaussian_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def augment_color_jitter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_sharpen(img):
    return cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


def augment_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


TRANSFORMS = [
    ("bright_up", lambda i: augment_brightness(i, 1.35)),
    ("bright_down", lambda i: augment_brightness(i, 0.65)),
    ("bright_mid_u", lambda i: augment_brightness(i, 1.15)),
    ("bright_mid_d", lambda i: augment_brightness(i, 0.85)),
    ("contrast_up", lambda i: augment_contrast(i, 1.4)),
    ("contrast_lo", lambda i: augment_contrast(i, 0.7)),
    ("noise", lambda i: augment_gaussian_noise(i, 20)),
    ("noise_light", lambda i: augment_gaussian_noise(i, 10)),
    ("blur", lambda i: augment_blur(i, 5)),
    ("color_jit", augment_color_jitter),
    ("sharpen", augment_sharpen),
    ("clahe", augment_clahe),
    ("fliph", lambda i: cv2.flip(i, 1)),
    ("flipv", lambda i: cv2.flip(i, 0)),
    ("rot90", lambda i: cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE)),
    ("rot180", lambda i: cv2.rotate(i, cv2.ROTATE_180)),
    ("rot270", lambda i: cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    # compound transforms, most useful at the sand <-> intact boundary
    ("bright_noise", lambda i: augment_gaussian_noise(augment_brightness(i, 1.2), 12)),
    ("dark_blur", lambda i: augment_blur(augment_brightness(i, 0.75), 3)),
    ("flip_jit", lambda i: augment_color_jitter(cv2.flip(i, 1))),
    ("clahe_noise", lambda i: augment_gaussian_noise(augment_clahe(i), 8)),
    ("sharp_bright", lambda i: augment_brightness(augment_sharpen(i), 1.1)),
    ("dark_contrast", lambda i: augment_contrast(augment_brightness(i, 0.7), 1.3)),
    ("flip_clahe", lambda i: augment_clahe(cv2.flip(i, 0))),
]


def generate_augmented_copies(src_path, dst_dir, num_copies):
    img = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 0
    transforms = TRANSFORMS[:]
    random.shuffle(transforms)
    count = 0
    for name, fn in transforms[:min(num_copies, len(transforms))]:
        cv2.imencode(".jpg", fn(img))[1].tofile(str(dst_dir / f"aug_{name}_{src_path.stem}.jpg"))
        count += 1
    return count


# ── dataset ─────────────────────────────────────────────────────────────────

def prepare_dataset():
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ["train", "val"]:
        for cls in CLASSES:
            (DATASET_DIR / split / cls).mkdir(parents=True)

    stats = {"train": {c: 0 for c in CLASSES}, "val": {c: 0 for c in CLASSES}}
    for cls in CLASSES:
        images = sorted(f for f in (LABELED_DIR / cls).iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"})
        random.seed(42)
        random.shuffle(images)
        ratio = 0.70 if cls == "sand" else TRAIN_RATIO     # sand: keep more for training
        split_idx = min(int(len(images) * ratio), max(len(images) - 1, 0))   # at least 1 val image
        for i, img in enumerate(images):
            split = "train" if i < split_idx else "val"
            shutil.copy2(str(img), str(DATASET_DIR / split / cls / img.name))
            stats[split][cls] += 1
    print(f"Raw split  train {stats['train']}  val {stats['val']}")

    # Minority classes up to 60 % of the majority class, training split only.
    target_min = int(max(stats["train"].values()) * 0.6)
    for cls in CLASSES:
        current = stats["train"][cls]
        if current >= target_min or current == 0:
            continue
        train_dir = DATASET_DIR / "train" / cls
        originals = list(train_dir.glob("*.jpg"))
        needed, added, cycle = target_min - current, 0, 0
        while added < needed and cycle <= 5:
            for img_path in originals:
                added += generate_augmented_copies(img_path, train_dir, min(needed - added, 24))
                if added >= needed:
                    break
            cycle += 1
        stats["train"][cls] = len(list(train_dir.glob("*.jpg")))
        print(f"Augmented {cls}: {current} -> {stats['train'][cls]}")
    return DATASET_DIR


def train_model(dataset_path):
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=str(dataset_path),
        epochs=150,
        imgsz=128,
        batch=16,
        patience=0,            # run all 150 epochs
        project=str(MODEL_SAVE.parent),
        name=MODEL_SAVE.name,
        exist_ok=True,
        device=0,
        verbose=True,
        # regularisation
        dropout=0.25,
        weight_decay=0.0005,
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=5,
        # heavy augmentation, especially brightness/contrast (sand <-> intact)
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.3,
        flipud=0.3,
        fliplr=0.5,
        erasing=0.15,
        mixup=0.15,
        # against over-confidence
        label_smoothing=0.1,
    )
    print(f"Best model: {MODEL_SAVE / 'weights' / 'best.pt'}  (deploy as zone_classifier_v8.pt)")


if __name__ == "__main__":
    train_model(prepare_dataset())
