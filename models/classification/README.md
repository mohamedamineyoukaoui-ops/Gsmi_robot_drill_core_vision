# Classification

[Project overview](../../README.md) · [Pipeline guide](../../pipeline/README.md)

These models classify either an image patch or a chemical composition.

| Model | Input | Output | Pipeline role |
| --- | --- | --- | --- |
| [zones-yolov8](zones-yolov8/README.md) | 128 × 128 column patch | Intact, rubble, sand, empty | Optional `--zones`; helps define core span and veto pieces |
| [lithology-dinov2](lithology-dinov2/README.md) | Square rock-surface tile | Six image classes, including `no_rock` | Optional `--lithology`; averages tile probabilities |
| [xrf-randomforest](xrf-randomforest/README.md) | Seven oxides transformed into 22 features | Nine Benguerir rock types plus `out_of_series` | Optional A4 chemistry scorer through the Python API |

The image classes are limestone, dolomitic limestone, marl, flint, out of series and no rock. [Lithology fusion](../../pipeline/README.md#rock-type-image--xrf) maps them into the richer chemical class set and flags disagreement. The default chemistry scorer is the untrained reference model **A3**; the random forest is used only when `xrf_method="a3xa4"` is requested.

## Why three classifiers?

**Zones describe the physical state of the core.** The useful distinction is whether a stretch contains intact rock, fragments, loose material or no rock. Its output affects the RQD geometry, particularly when a mask or detector encloses rubble as if it were one long piece.

**Image lithology describes appearance.** Frozen DINOv2 features encode the surface's texture and colour; a trained linear head maps them to the six visual classes. Tile probabilities are aggregated so a tray can be described without treating its walls and labels as rock. The model comparison examines transfer to local photographs and probability calibration alongside test accuracy.

**XRF classification describes composition.** A random forest uses oxide log-ratios and a missing-MgO flag. Its label space includes chemical mixtures that the image classifier does not distinguish. Fusion first maps the visual classes into that richer space, then combines the two distributions and reports family-level disagreement.

## Reading the results

- **Accuracy:** the share of predictions that are correct. Read it alongside class support when the dataset is imbalanced.
- **Macro-F1:** the unweighted average of per-class F1 scores. DINOv2's regularisation is selected by validation macro-F1; the zone model uses its classification checkpoint criterion.
- **Confusion matrix:** inspect the axis labels. The generated report matrices use true classes as rows and predictions as columns.
- **Expected calibration error (ECE):** compares confidence with observed correctness. It is relevant when probabilities are combined across models.

Each model README links to its recorded metrics and describes its split. The zone results use real validation patches; image lithology uses a held-out test set; XRF reports cross-validation separately for USGS analyses and synthetic Benguerir readings.
