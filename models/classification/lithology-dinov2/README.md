# Rock type from the image: DINOv2 + logistic regression

[Project overview](../../../README.md) · [Classification](../README.md) · [Pipeline guide](../../../pipeline/README.md)

Classifies square tiles of rock surface into the rock types of the Benguerir phosphate series. The pipeline averages the tiles of a tray and then combines the result with the XRF chemistry ([lithology_fusion.py](../../../pipeline/lithology_fusion.py)).

| Property | Value |
| --- | --- |
| Task | image classification, 6 classes: limestone, dolomitic limestone, marl, flint, out of series, no rock |
| Architecture | DINOv2 ViT-S/14 (21 M parameters), **frozen**, + multinomial logistic regression (2,310 trained parameters) |
| Input | square tile cut inside the core band of a column → 224×224, RGB, ImageNet normalisation |
| Output | 6 probabilities. Tiles with `no_rock` probability ≥ 0.5 are excluded from the average when any rock tiles remain. If every tile is excluded, all tiles are averaged so the no-rock signal is retained. |
| Used in | [pipeline/lithology_image.py](../../../pipeline/lithology_image.py) |

## Architecture

![Vision Transformer architecture](../../../docs/architectures/vit.png)

*Vision Transformer, Dosovitskiy et al. (2021), figure from [google-research/vision_transformer](https://github.com/google-research/vision_transformer) (Apache 2.0).*

DINOv2 ViT-S/14 is this architecture with 14×14-pixel patches (16×16 = 256 patches for a 224×224 tile), 384-dimensional embeddings and 12 encoder blocks. DINOv2 trained it without labels; here it is **frozen**. The figure's "MLP head" is replaced by the trained part: the [class] token output (384-d) → L2 normalisation → logistic regression (6×384 weights + 6 biases) → softmax.

DINOv2 (Meta, 2023) was trained without labels on 142 M images to produce general-purpose visual features: texture, grain, colour, structure. Only the linear head on top is trained here.

**Why frozen features and not fine-tuning:** with about 500 images per class drawn from a few sources, a fine-tuned network learns each source's look (camera, lighting, background) as much as the rock. A linear head on fixed features limits trainable capacity, although it does not eliminate source bias. On the test set, the YOLO classifiers trained end to end scored higher. On the 12 published photos of real Benguerir rock, which no model saw in training, DINOv2 performed best in the recorded comparison. This small sample is exploratory evidence, not a field-validation benchmark.

| Model, same split | Test accuracy | Macro-F1 | Real Benguerir photos | Calibration error (ECE) |
| --- | --- | --- | --- | --- |
| **DINOv2 + logistic regression** | 82.8 % | 83.2 % | **6 / 12** | **2.8 %** |
| YOLOv8n-cls, fine-tuned | 84.5 % | 85.1 % | 3 / 12 | 6.6 % |
| YOLO11s-cls, fine-tuned | 79.1 % | 79.8 % | 2 / 12 | 12.6 % |

DINOv2 is also the best calibrated, which matters because its probabilities are multiplied with the chemistry's.

## Data

Public datasets, with each source's labels mapped onto the 6 classes (for example shale → marl, chert → flint, sandstone → out of series):

| Source | Content | Used for |
| --- | --- | --- |
| Mendeley Data lithology dataset (Alzubaidi et al., 2021, South Australia) | core-tray crops | limestone, marl, out of series, no rock |
| DCID drill-core image dataset (Li et al., 2025, China) | drill-core images | dolomitic limestone, marl, out of series |
| Four Kaggle rock image datasets: stealthtechnologies/rock-classification, mahmoudalforawi/igneous-metamorphic-sedimentary-rocks-and-minerals, neelgajare/rocks-dataset, salmaneunus/rock-classification | rock photos | limestone, marl, flint, out of series, dolomitic limestone |
| Wikimedia Commons | geologists' specimen photos, kept only with a confirmed licence | flint, dolomitic limestone, marl |
| GSMI tray photos | empty troughs, labels, wood | no rock |

Each class has 500 training images (light augmentation for the classes with fewer originals), about 100 validation images and about 100 test images: 3,000 / 641 / 679 in total. Test and validation images are always originals.

**Splits by physical object.** Tiles of the same drillhole, specimen or near-identical photo (DINOv2 cosine similarity ≥ 0.92) were grouped during dataset preparation to reduce leakage. A random split had put tiles of the same drillholes in train and test and inflated the scores. The reported residual near-duplicate rate is at most 1% of test images, so this is not a claim of zero leakage. The training script consumes the prepared split; it does not recreate those grouping checks.

## Training

| Setting | Value |
| --- | --- |
| Backbone | `vit_small_patch14_dinov2.lvd142m` (timm), frozen, no gradient |
| Head | scikit-learn `LogisticRegression`, multinomial, `class_weight="balanced"`, max 5,000 iterations |
| Hyper-parameter | inverse regularisation C searched in {0.1, 0.3, 1, 3, 10} on validation macro-F1 → C = 10 |
| Export | the head as two arrays (`coef` 6×384, `intercept` 6), so inference needs no scikit-learn |
| Time | under a minute after the features are computed |

Script: [train.py](train.py), feature extractor: [features.py](features.py).

## Results

The exported model (backbone + the two head arrays, exactly what the pipeline loads) evaluated on the held-out test set of 679 images. Reproduce with [evaluate.py](evaluate.py); all numbers are in [results/metrics.json](results/metrics.json).

**Accuracy 82.8 %, macro-F1 83.2 %, macro ROC-AUC 0.970, calibration error (ECE) 2.8 %.**

| Class | Support | Precision | Recall | F1 | Average precision |
| --- | --- | --- | --- | --- | --- |
| limestone | 135 | 0.766 | 0.704 | 0.734 | 0.779 |
| dolomitic limestone | 100 | 0.875 | 0.910 | 0.892 | 0.959 |
| marl | 117 | 0.724 | 0.897 | 0.802 | 0.921 |
| flint | 103 | 0.835 | 0.738 | 0.784 | 0.907 |
| out of series | 115 | 0.835 | 0.791 | 0.813 | 0.883 |
| no rock | 109 | 0.981 | 0.954 | 0.967 | 0.995 |

![Confusion matrix](results/confusion_matrix.png)

![Regularisation sweep, calibration and precision-recall curves](results/evaluation_curves.png)

- **Training curve:** the head has no epochs. Its one hyper-parameter, C, was chosen on the validation set (left panel). The score still rises at C = 10, the end of the grid, so a larger C might do slightly better.
- **Calibration** (middle): the points follow the diagonal. When the model says 85 %, it is right about 90 % of the time, a little under-confident at high confidence. That is the safe direction for a fusion that multiplies probabilities.
- The main confusion is limestone → marl (26 of 135). Both are pale carbonate rocks, and they differ mostly in clay content, which a photo shows poorly but the XRF measures directly.
- Accuracy by source ranges from 96 % (DCID) to 67 % (Kaggle, whose web labels are noisier).
- Dolomitic limestone reaches 91 % on the test set, but almost all training dolomite is dark DCID core. The pale Benguerir dolomite is read as limestone. With an XRF reading, the MgO content corrects it.

The comparison with the YOLO classifiers and the temperature-scaling result are in [results/info.json](results/info.json).

## Run this model

Install `.[training]` from the project root. Supply the prepared dataset under `models/classification/lithology-dinov2/data/dataset_A/{train,val,test}/<class>/*.jpg`. Class directory names must match `CLASSES` in [train.py](train.py), using underscores such as `dolomitic_limestone`, `out_of_series` and `no_rock`.

```bash
python models/classification/lithology-dinov2/train.py
python models/classification/lithology-dinov2/evaluate.py --weights models/classification/lithology-dinov2/runs --data models/classification/lithology-dinov2/data/dataset_A
```

The feature extractor loads pretrained DINOv2 through timm and may download it on first use. Training writes the backbone, head and run information to this model's `runs/` directory; evaluation overwrites `results/metrics.json`. The archived `results/info.json` also contains historical comparisons that a new training run does not regenerate.

Copy `dinov2_vits14_224.pt` and `dinov2_head_A.npz` from `runs/` into `weights/lithology/`, then enable `--lithology`. The standalone evaluation resizes dataset images directly; the tray pipeline also selects tiles and corrects shared colour casts, so the reported test accuracy is not an end-to-end tray score.
