"""Where the trained model files live.

Weights are not part of the repository. Point GSMI_MODELS_DIR at a folder that
contains them (default: ./weights). Expected files, and the folder that trains each:

    column_yolo11s.pt          models/detection/columns_uni
    piece_yolo11s.pt           models/detection/pieces_uni
    crack_unet_resnet34.pth    models/segmentation/cracks_uni (run 2; run 1 = crack_unet_resnet34_run1.pth)
    zone_classifier_v8.pt      models/classification/zones-yolov8
    lithology/                 models/classification/lithology-dinov2 + xrf-randomforest
        dinov2_vits14_224.pt   frozen DINOv2 backbone
        dinov2_head_A.npz      trained logistic-regression head
        xrf_a4_world.joblib    XRF random forest
"""
import os
from pathlib import Path

COLUMNS = "column_yolo11s.pt"
PIECES = "piece_yolo11s.pt"
CRACKS = "crack_unet_resnet34.pth"
ZONES = "zone_classifier_v8.pt"
LITHOLOGY_DIR = "lithology"


def models_dir() -> Path:
    return Path(os.environ.get("GSMI_MODELS_DIR", "weights")).expanduser().resolve()


def path(name: str) -> Path:
    return models_dir() / name


def exists(name: str) -> bool:
    return path(name).exists()
