"""
Command line.

    gsmi-vision demo                         no weights needed
    gsmi-vision stitch FOLDER --output tray.png
    gsmi-vision analyze TRAY.png --pixels-per-cm 50 --output runs/tray [--method ...] [--zones] [--lithology]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items() if k != "annotated_image"}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(data), indent=2), encoding="utf-8")


def parse_elements(text: str) -> dict[str, float]:
    """"Ca=33.7,Si=4.6,P=1.0" -> {"Ca": 33.7, "Si": 4.6, "P": 1.0}"""
    out = {}
    for part in text.split(","):
        name, value = part.split("=")
        out[name.strip()] = float(value)
    return out


def demo(output: Path) -> dict:
    """The geometry of the pipeline on synthetic inputs: stitching, piece
    extraction, RQD and fusion. Checks the plumbing; says nothing about model accuracy."""
    from pipeline import pieces, rqd
    from pipeline.image_processing.stitching import assemble_grid
    from pipeline.lithology_fusion import decide

    # 1. two overlapping tiles of a random scene must stitch back exactly
    scene = np.random.default_rng(42).integers(30, 220, (160, 300, 3), dtype=np.uint8)
    stitched = assemble_grid([scene[:, :190], scene[:, 110:]], [(0, 0), (0, 1)])
    write_image(output / "stitched.png", stitched)

    # 2. a 60 cm column (10 px/cm) outlined by the mask and cut by three cracks
    px_per_cm = 10.0
    mask = np.zeros((600, 60), np.uint8)
    cv2.rectangle(mask, (0, 0), (59, 599), 255, 2)
    for y in (150, 190, 420):
        cv2.line(mask, (0, y), (59, y), 255, 3)
    found = pieces.extract_pieces_contour(mask, None, (0, 600), px_per_cm)
    column_rqd = rqd.rqd_from_pieces(found, 600 / px_per_cm, px_per_cm)

    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for p in found:
        color = (60, 160, 60) if p["counts_for_rqd"] else (40, 130, 200)
        cv2.rectangle(view, (4, p["y_start"]), (55, p["y_end"]), color, -1)
    write_image(output / "pieces.png", view)

    # 3. image says limestone, XRF of a typical Benguerir limestone agrees
    decision = decide({"limestone": 0.8, "marl": 0.15, "no_rock": 0.05},
                      {"Ca": 33.7, "Si": 4.6, "P": 1.05, "Al": 0.74, "Fe": 0.63})

    data = {
        "kind": "synthetic demonstration, not a model evaluation",
        "stitching_exact": bool(np.array_equal(stitched, scene)),
        "pieces_cm": [p["length_cm"] for p in found],
        "rqd": column_rqd,
        "lithology": {"target": decision.target, "confidence": round(decision.confidence, 3),
                      "source": decision.source, "conflict": decision.conflict},
    }
    save_json(output / "results.json", data)
    print(json.dumps(data, indent=2))
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gsmi-vision", description="GSMI core vision pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="synthetic demonstration, no weights needed")
    p_demo.add_argument("--output", type=Path, default=Path("demo-output"))

    p_stitch = sub.add_parser("stitch", help="stitch a folder of overlapping photos")
    p_stitch.add_argument("folder", type=Path)
    p_stitch.add_argument("--output", type=Path, required=True)

    p_an = sub.add_parser("analyze", help="full analysis of one tray image (needs weights)")
    p_an.add_argument("image", type=Path)
    p_an.add_argument("--pixels-per-cm", type=float, required=True)
    p_an.add_argument("--output", type=Path, required=True)
    p_an.add_argument("--weights", type=Path, default=None, help="folder with the model files (default ./weights)")
    p_an.add_argument("--method", choices=["contour", "connected_components", "piece_detect"], default="contour")
    p_an.add_argument("--zones", action="store_true", help="also run the zone classifier")
    p_an.add_argument("--lithology", action="store_true", help="also classify the rock type")
    p_an.add_argument("--xrf", type=parse_elements, default=None, help='e.g. "Ca=33.7,Si=4.6,P=1.0"')
    p_an.add_argument("--depth-from", type=float, default=0.0, help="run start depth (m); RQD then uses the run length")
    p_an.add_argument("--depth-to", type=float, default=0.0, help="run end depth (m)")
    p_an.add_argument("--lithology-image", type=Path, default=None,
                      help="higher-resolution copy of the same tray, written by 'stitch'")

    args = parser.parse_args(argv)

    if args.command == "demo":
        demo(args.output)
        return 0

    if args.command == "stitch":
        from pipeline.image_processing.stitching import stitch_folder
        image, px_per_cm, hires = stitch_folder(args.folder)
        write_image(args.output, image)
        info = {"image": str(args.output), "pixels_per_cm": px_per_cm}
        if hires is not None:
            info["lithology_image"] = str(args.output.with_name(args.output.stem + "_lithology.jpg"))
            write_image(Path(info["lithology_image"]), hires)
        print(json.dumps(info))
        return 0

    if not np.isfinite(args.pixels_per_cm) or args.pixels_per_cm <= 0:
        parser.error("--pixels-per-cm must be finite and positive")
    if args.weights is not None:
        os.environ["GSMI_MODELS_DIR"] = str(args.weights.resolve())
    os.environ["YOLO_AUTOINSTALL"] = "false"

    from pipeline import weights
    required = [weights.COLUMNS]
    required.append(weights.PIECES if args.method == "piece_detect" else weights.CRACKS)
    if args.zones:
        required.append(weights.ZONES)
    if args.lithology:
        required.append(weights.LITHOLOGY_DIR)
    missing = [name for name in required if not weights.exists(name)]
    if missing:
        parser.error(f"Missing local model assets in {weights.models_dir()}: " + ", ".join(missing))

    from pipeline.analyze import analyze
    rgb = cv2.cvtColor(read_image(args.image), cv2.COLOR_BGR2RGB)
    hires = None
    if args.lithology_image is not None:
        hires = cv2.cvtColor(read_image(args.lithology_image), cv2.COLOR_BGR2RGB)
    result, _ = analyze(rgb, args.pixels_per_cm, method=args.method, run_zones=args.zones,
                        run_lithology=args.lithology, xrf_elements=args.xrf, lithology_rgb=hires,
                        depth_from_m=args.depth_from, depth_to_m=args.depth_to)

    write_image(args.output / "annotated.png", cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR))
    save_json(args.output / "result.json", result)
    print(f"RQD {result['rqd_percent']} % ({result['rqd_class']}), status {result['status']}", result["warnings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
