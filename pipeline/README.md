# Pipeline

[Project overview](../README.md) · [Setup and model files](../README.md#quick-start) · [Stitching](image_processing/README.md)

The code that runs the trained models and turns their outputs into measurements. Entry point: [analyze.py](analyze.py), or the `gsmi-vision` command ([cli.py](cli.py)).

## Python API

Install the inference dependencies and provide the [required model files](#model-files) before running this example from the project root. The public analysis API expects an **RGB** image and a finite, positive scale in **pixels per centimetre**.

```python
import cv2
from pipeline.analyze import analyze, recompute
from pipeline.cli import read_image
from pathlib import Path

rgb = cv2.cvtColor(read_image(Path("tray.png")), cv2.COLOR_BGR2RGB)
result, state = analyze(rgb, pixels_per_cm=12.7, method="contour")

# Supply the complete set of corrections on each call, in original-image pixels.
corrected = recompute(
    state,
    manual_lines=[[(50, 200), (150, 200)]],  # polyline: (x, y) points
    manual_boxes=[(50, 300, 100, 150)],      # intact box: (x, y, width, height)
)
```

Replace 12.7 with the measured scale for the input image. `result` includes an RGB `annotated_image` array; the CLI writes that separately to `annotated.png` and serializes the remaining data to `result.json`.

## Command line

Run from the project root.

In the commands below, `python` means your virtual environment's interpreter: activate it or use its full path as above. The installed `gsmi-vision` command is equivalent to `python -m pipeline.cli`.

```bash
python -m pip install -e ".[inference]"
python -m pipeline.cli stitch scans/tray_01/ --output tray.png
```

The stitch command prints JSON containing `pixels_per_cm` and, when available, a `lithology_image` path. Use the scale for **your scan**. The following example assumes the returned scale is 12.7:

```bash
python -m pipeline.cli analyze tray.png --pixels-per-cm 12.7 --weights weights --output runs/tray_01
```

| Optional argument | Purpose |
| --- | --- |
| `--method contour` | Default: pieces reconstructed from crack outlines |
| `--method connected_components` | Pieces separated by cracks extended across the core |
| `--method piece_detect` | Direct piece detection; uses the piece model instead of the U-Net |
| `--zones` | Enable zone classification and piece vetoes |
| `--lithology` | Enable image-based rock-type prediction |
| `--xrf "Ca=33.7,Si=4.6,P=1.0"` | Element concentrations in weight percent; requires `--lithology` in this CLI |
| `--lithology-image tray_lithology.jpg` | Use the higher-resolution copy, if the stitcher produced one |
| `--depth-from 70 --depth-to 73` | Use a 3 m drilled run as the tray RQD denominator |

Analysis writes `annotated.png` and `result.json`. The JSON includes pieces, RQD, denominator source, fracture frequency, optional lithology, and any warnings. Review `status` and `warnings` alongside the measurements.

### Model files

Weights are not published with this project. Place the required files in `weights/`, pass `--weights PATH`, or set `GSMI_MODELS_DIR`. Names are defined in [pipeline/weights.py](weights.py).

| Path relative to the weights folder | Required for |
| --- | --- |
| `column_yolo11s.pt` | Every CLI analysis |
| `crack_unet_resnet34.pth` | `contour` and `connected_components` (U-Net run 2) |
| `piece_yolo11s.pt` | `piece_detect` |
| `zone_classifier_v8.pt` | `--zones` |
| `lithology/dinov2_vits14_224.pt` | `--lithology`: frozen backbone |
| `lithology/dinov2_head_A.npz` | `--lithology`: trained head |
| `lithology/xrf_a4_world.joblib` | Optional `xrf_method="a3xa4"` in the fusion API |

## Two stages

![Stage 1 runs the models once; stage 2 recomputes the geometry without any model](../docs/diagrams/pipeline_two_stages.png)

`analyze()` runs both stages and returns `(result, state)`. When a geologist draws a missed crack or marks a block as intact, `recompute(state, ...)` replays only stage 2. Corrections are applied to a copy of the cached mask, so each call starts from the original model output.

The state contains only the models requested by the original analysis. You can switch between `contour` and `connected_components` using a cached mask. To switch to or from `piece_detect`, run `analyze()` again to obtain the required model output. Manual crack lines affect mask-based methods; intact boxes also work with detected pieces.

## From pixels to pieces

A **piece** is a depth interval of one column, with a length in cm. Three methods produce them; the default is `contour`.

| Method | How | Weak point |
|---|---|---|
| `contour` | Draw the core outline into the crack mask, close gaps up to 0.6 cm, flood-fill the outside away. Each enclosed region is a candidate piece, kept only if at most 20 % of its interior is dark tray. | an outline gap wider than 0.6 cm merges two pieces |
| `connected_components` | Find the fractures that cross ≥ 70 % of the core width, extend them across it, then rock = core minus cracks. | depends on the crossing test |
| `piece_detect` | The YOLO11 detector returns pieces as boxes. In each column they are stacked along the core: a box reporting the same depth as the one above keeps only the more confident of the two, and a small overlap is cut in the middle. | pieces the detector never boxed cannot be recovered; depth labels can be boxed |

The recorded development experiment on 528 annotated columns reduced mean absolute RQD error from 15.1 to 11.5 percentage points with the interior test, and selected a 0.6 cm closing gap. This tuning result is not an independent validation against field logs; its detailed evaluation report is not bundled here.

Rules shared by all three methods, so their results are comparable:

1. **Shared denominator rule:** the core span runs from the first to the last row where the available model outputs indicate rock (zones, mask or detector). Gaps inside that span remain in the denominator. Different methods can produce different detected spans; supplying drilled depths fixes the tray denominator across methods.
2. **Zone vetoes:** a piece lying more than 50 % in a confident rubble or sand zone (p ≥ 0.70) never counts, and in an empty zone (p ≥ 0.85) it is not rock at all.
3. **User boxes win:** an "intact" box replaces the pieces under it, and its height is the piece length.
4. **Column boundaries:** pieces within 3 pixels of the bottom of one column and the top of the next can be chained before the 10 cm rule is applied to their total. This assumes left-to-right column order and top-to-bottom core progression; the geometry alone cannot prove that the two ends belong to one physical piece.

## From pieces to RQD

```text
RQD (%) = 100 × length of the union of qualifying piece intervals / denominator length
```

- **Union, not sum:** overlapping pieces are counted once, so RQD can never exceed 100 %.
- **Denominator:** the core length detected in the image, or the drilled run length when the start and end depths are given. With the run length, lost core counts against RQD, as in Deere's definition, and core recovery (TCR) is reported too.
- **Fracture frequency** (fractures per metre) is reported next to RQD. RQD discards everything below 10 cm: a core broken every 9 cm and a crushed core both score 0 %. Frequency does not, and it only needs to know that two pieces are separate, not their exact lengths. A chained boundary between columns is not counted as a fracture.

The output quality classes are: **Very Poor** below 25%, **Poor** from 25% to below 50%, **Fair** from 50% to below 75%, **Good** from 75% to below 90%, and **Excellent** from 90%. `rqd_from_jv` is a separate heuristic derived from fracture frequency; the primary measurement is `rqd_percent`.

## Outputs and fallbacks

- The result includes `columns`, `rqd_percent`, `rqd_class`, `rqd_denominator_cm`, `rqd_denominator_source`, `fracture_frequency_per_m`, `status` and `warnings`. Lithology fields are populated only when the requested classification produces a result.
- Valid `depth_from_m` and `depth_to_m` values with end greater than start select the drilled-run denominator and enable `core_recovery_percent`. Otherwise the detected span is used.
- The CLI checks for required model assets before analysis. The Python column detector can fall back to a brightness-based five-column layout if detection fails or finds no boxes; this is a heuristic, not a replacement for the trained model.
- Failed crack segmentation can trigger a zone-based fallback. It is useful only if zone classification succeeded; inspect the reported `method`, overlay and warnings before using its measurements.
- Warnings set `status` to `partial`. A result with `status="ok"` is not a guarantee of geological accuracy.

## Files

| File | Step |
|---|---|
| [image_processing/](image_processing) | stitching the robot's photos, pixel scale (no AI) |
| [cracks.py](cracks.py) | U-Net on the whole image at 10 px/cm, in overlapping 1,024-row bands |
| [columns.py](columns.py) | column boxes from core start to core end, with a non-AI fallback |
| [zones.py](zones.py) | sliding-window zone classification, crack-continuity correction |
| [pieces.py](pieces.py) | the three piece methods and the shared rules |
| [rqd.py](rqd.py) | RQD, quality class, fracture frequency |
| [lithology_image.py](lithology_image.py) | tiles inside the core band → DINOv2 → averaged probabilities |
| [lithology_fusion.py](lithology_fusion.py) | XRF chemistry, image + XRF fusion, conflict flag |
| [overlay.py](overlay.py) | annotated image |
| [analyze.py](analyze.py) | the two stages |
| [cli.py](cli.py) | `gsmi-vision demo / stitch / analyze` |
| [weights.py](weights.py) | where the model files are expected |

## Rock type: image + XRF

```text
P(rock) ∝ P_image(rock)^w_image × P_XRF(rock)^w_xrf
```

The image classifier does not have separate labels for phosphatic mixtures. Each image class spreads its probability over compatible chemical classes, and the two distributions are combined using a weighted product. A **conflict flag** marks cases where the chemistry rules out the rock family preferred by the image.

The default chemistry method is **A3**, a reference-composition scorer with no trained weights. **A3 × A4** additionally uses the [XRF random forest](../models/classification/xrf-randomforest/README.md):

```python
from pipeline.lithology_fusion import decide

decision = decide(
    {"limestone": 0.8, "marl": 0.15, "no_rock": 0.05},
    {"Ca": 33.7, "Si": 4.6, "P": 1.0},  # elemental weight percentages
    xrf_method="a3xa4",                  # requires lithology/xrf_a4_world.joblib
)
```

`analyze()` and the CLI use A3; they do not expose a chemistry-method selector. In the CLI, `--xrf` is used only with `--lithology` and successful image classification. For chemistry-only decisions, call `decide(None, elements)` directly. A historical 11/12 image-plus-XRF result used an earlier image model and is not a benchmark for the current pipeline.
