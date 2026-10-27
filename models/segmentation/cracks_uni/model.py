import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def create_model(encoder_name="resnet34", encoder_weights="imagenet"):
    """
    U-Net with a ResNet34 encoder pretrained on ImageNet.
    One output channel (crack) with no activation: the loss applies the sigmoid.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )


# ── soft skeleton (Shit et al., clDice, CVPR 2021) ───────────────────────────

def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iterations):
    """Centre line of a soft mask: what is left after repeated thinning.
    `iterations` must exceed half the stroke width, or thick strokes keep their body."""
    skeleton = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class CrackLoss(nn.Module):
    """0.5 x binary cross-entropy + 0.25 x Dice + 0.25 x clDice.

    BCE judges every pixel on its own. Dice measures the overlap of the
    predicted and true strokes, insensitive to the many background pixels.
    Neither notices a short break in a line: a 2-pixel hole costs almost
    nothing, yet it opens the outline of a piece. clDice compares the centre
    lines: the true centre line must lie inside the prediction (no break) and
    the predicted one inside the truth (no stray line), so a broken outline
    costs as much as a missing one.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.25, cldice_weight=0.25, skeleton_iterations=12):
        super().__init__()
        self.weights = (bce_weight, dice_weight, cldice_weight)
        self.iterations = skeleton_iterations
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, smooth=1.0):
        logits, targets = logits.float(), targets.float()          # full precision, also under AMP
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum()
        dice = 1 - (2 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        skel_pred = soft_skeleton(probs, self.iterations)
        skel_true = soft_skeleton(targets, self.iterations)
        precision = ((skel_pred * targets).sum() + smooth) / (skel_pred.sum() + smooth)
        sensitivity = ((skel_true * probs).sum() + smooth) / (skel_true.sum() + smooth)
        cldice = 1 - 2 * precision * sensitivity / (precision + sensitivity)

        w_bce, w_dice, w_cl = self.weights
        return w_bce * bce + w_dice * dice + w_cl * cldice
