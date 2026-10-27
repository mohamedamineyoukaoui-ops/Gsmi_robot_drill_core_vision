"""
Crack masks at one physical resolution.

Every tray is first resized to WORK_PX_PER_CM (its own px/cm is known), so a
1 cm crack has the same width in pixels in every image, and the pipeline runs
the model at that same resolution.

Training sample:
  1. pick a tray, pick a scale s between 0.75 and 1.25 (the resolution varies by ±25 %)
  2. cut a random window of 512 / s pixels from the tray at the working resolution
  3. resize it to 512 x 512 (so the window ends up at s x the working resolution)
  4. flips, brightness / contrast, noise; no 90° rotation (trays are always vertical)
Validation: whole trays at exactly the working resolution (see train.py).
"""

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def read(path: str, flag: int) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, np.uint8), flag)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


def to_resolution(image: np.ndarray, mask: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Resize image and mask by factor. The mask is resized as a float and kept where
    it covers a quarter of a pixel, so thin strokes do not vanish when shrinking."""
    h, w = image.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    m = cv2.resize((mask > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    return image, (m > 0.25).astype(np.uint8)


def load_tray(record: dict, folder, work_px_per_cm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(RGB image, 0/1 mask, factor) of one tray at the working resolution."""
    image = cv2.cvtColor(read(record["image_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = read(str(folder / record["mask"]), cv2.IMREAD_GRAYSCALE)
    factor = work_px_per_cm / record["px_per_cm"]
    image, mask = to_resolution(image, mask, factor)
    return image, mask, factor


class CrackCrops(Dataset):
    def __init__(self, records: list[dict], folder, work_px_per_cm: float, size: int = 512,
                 scale=(0.75, 1.25), crops_per_tray: int = 4):
        self.size, self.scale, self.crops_per_tray = size, scale, crops_per_tray
        self.trays = [load_tray(r, folder, work_px_per_cm)[:2] for r in records]      # kept in memory
        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.trays) * self.crops_per_tray

    def __getitem__(self, idx):
        image, mask = self.trays[idx % len(self.trays)]
        s = np.random.uniform(*self.scale)
        win = int(round(self.size / s))
        h, w = mask.shape
        if h < win or w < win:                                # small tray: pad up to the window
            ph, pw = max(0, win - h), max(0, win - w)
            image = np.pad(image, ((0, ph), (0, pw), (0, 0)))
            mask = np.pad(mask, ((0, ph), (0, pw)))
            h, w = mask.shape
        y, x = np.random.randint(0, h - win + 1), np.random.randint(0, w - win + 1)
        crop_img = cv2.resize(image[y:y + win, x:x + win], (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        crop_mask = cv2.resize(mask[y:y + win, x:x + win].astype(np.float32), (self.size, self.size),
                               interpolation=cv2.INTER_LINEAR)
        out = self.augment(image=crop_img, mask=(crop_mask > 0.25).astype(np.float32))
        return out["image"], out["mask"].unsqueeze(0)
