# Segmentation

[Project overview](../../README.md) · [Pipeline guide](../../pipeline/README.md)

The crack model assigns a probability to every pixel, distinguishing crack or piece-outline pixels from the background.

| Model | Output | Pipeline role |
| --- | --- | --- |
| [cracks_uni](cracks_uni/README.md) | Crack probability map and binary mask | Supplies `contour` and `connected_components` piece extraction |

The U-Net uses a ResNet34 encoder and runs at **10 px/cm** when the image scale is supplied. Its probabilities are resized back to the input image and thresholded at **p > 0.45**, followed by morphological cleanup. `piece_detect` uses the separate piece detector and does not require this mask.

## Why pixel-level boundaries?

A box around a crack would locate it approximately, but reconstructing intact pieces requires knowing which pixels separate one region of rock from the next. The mask represents both cross-core fractures and the outlines along the sides of pieces. The geometry closes small gaps, identifies enclosed regions and converts each region's vertical extent into a length.

This makes continuity important: a small missing section can join two pieces even when most predicted pixels are correct. The training loss therefore combines binary cross-entropy, Dice overlap and clDice, which compares thin centre-line structures. Evaluation then measures the pieces produced by the geometry as well as the segmentation pixels.

## Reading the results

- **Pixel precision / recall:** the share of predicted crack pixels that match the annotation / the share of annotated crack pixels found.
- **Pixel F1 (Dice):** `2 * TP / (2 * TP + FP + FN)`.
- **IoU:** `TP / (TP + FP + FN)`, no greater than Dice for the same non-empty prediction and truth.

Thin cracks make pixel scores sensitive to small shifts. The measurement task is therefore also evaluated on the **pieces reconstructed from the mask**. Whole validation trays pass through the contour method, and predicted pieces are matched one to one to annotated pieces at depth-interval IoU ≥ 0.5.

The best checkpoint is selected by reconstructed-piece F1. The [model README](cracks_uni/README.md#results) reports both runs, threshold sweeps and the share of outlines that the geometry had to close. Run 2 at threshold 0.45 is the pipeline default.
