# Detection

[Project overview](../../README.md) · [Pipeline guide](../../pipeline/README.md)

Both detectors use **YOLO11s**, pretrained on COCO and fine-tuned end to end. They return bounding boxes with a class and confidence score.

| Model | Purpose | Pipeline confidence | Validation result |
| --- | --- | --- | --- |
| [columns_uni](columns_uni/README.md) | Locate each column from core start to core end | 0.60 | mAP@0.5 0.995 |
| [pieces_uni](pieces_uni/README.md) | Detect intact pieces for `piece_detect` | 0.20 | Piece F1 0.863 after interval processing |

## Architecture

![YOLO11 architecture](../../docs/architectures/yolo11.png)

Source and licence: [architecture credits](../../docs/architectures/CREDITS.md).

The backbone extracts features, the neck combines resolutions, and the detection heads predict boxes at multiple scales. Column boxes define the regions processed by later steps. Piece boxes are assigned to columns and resolved into depth intervals before RQD is computed.

The architecture has three roles:

- **Backbone:** build progressively smaller feature maps that describe edges, texture and object structure. YOLO11 uses C3k2 blocks and a C2PSA attention block after spatial pyramid pooling.
- **Neck:** combine information across resolutions, so fine spatial detail can be used together with a larger view of the tray.
- **Detection heads:** predict boxes and class scores at three scales, with strides of 8, 16 and 32 pixels. Fine maps help locate thin pieces; coarser maps cover larger structures such as columns.

Both tasks use a single object class, but their measurements differ. For columns, the top and bottom edges establish the recovered-core span. For pieces, each box's vertical interval becomes a candidate physical length after scaling and overlap resolution. The model-specific evaluations therefore include core-end error for columns and reconstructed-piece metrics for pieces.

## Reading the results

- **IoU:** intersection over union between a predicted box and its annotation.
- **Precision / recall:** the share of predictions that are correct / the share of annotated objects found.
- **mAP@0.5:** average precision with an IoU matching threshold of 0.5.
- **mAP@0.5:0.95:** average precision across IoU thresholds from 0.5 to 0.95; tighter boxes score better.
- **Piece F1:** precision and recall after column assignment and overlap resolution, matched by depth-interval IoU. This differs from raw detector metrics.

The model READMEs distinguish the best operating point from the confidence used by the pipeline. Checkpoint selection is handled by the installed Ultralytics version; the figure-generation script uses a weighted mAP score to annotate the saved training curves.

## Training and evaluation

Install `.[training]` from the project root, then follow each model's execution notes. Training requires the reviewed GSMI annotations and their source images. Evaluation requires the prepared YOLO dataset and trained weights. Both datasets are split by physical tray; their validation trays share boreholes with training, so the results do not establish transfer to unseen sites.
