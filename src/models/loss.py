# =============================================================================
# loss.py
# =============================================================================
# Combined loss function for semantic segmentation.
#
# The total loss is a weighted sum of Dice loss and Cross-Entropy loss:
#     L_total = dice_weight * L_dice + ce_weight * L_ce
#
# Why combine both?
#   - Cross-Entropy provides stable, well-behaved gradients throughout
#     training. It treats each pixel independently and handles
#     multi-class classification naturally.
#   - Dice loss directly optimizes the overlap metric (Dice coefficient)
#     that we care about at evaluation time. It is inherently robust to
#     class imbalance because it normalizes by the total predicted and
#     ground truth areas.
#   - Neither alone is sufficient: CE alone can be dominated by the
#     background class; Dice alone can produce unstable gradients early
#     in training when predictions are poor.
#
# Additional features:
#   - Per-class weighting for CE loss (computed from training set
#     class frequencies in dataset.py).
#   - Label smoothing to prevent overconfident predictions.
#   - Multiclass Dice computed per-class then averaged (macro Dice).
# =============================================================================

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Multiclass Dice loss.

    Computes the Dice loss independently for each class (excluding
    background optionally), then averages across classes. This is the
    macro-averaged Dice loss, which gives equal weight to each class
    regardless of its pixel frequency.

    The Dice coefficient for a single class c is:
        Dice_c = (2 * sum(p_c * g_c) + smooth) / (sum(p_c) + sum(g_c) + smooth)

    The Dice loss is:
        L_dice = 1 - mean(Dice_c) for all classes c.

    Args:
        smooth: Smoothing constant to prevent division by zero and
                stabilize gradients when a class has very few pixels.
        ignore_index: Class index to exclude from loss computation
                      (typically background). Set to None to include all.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multiclass Dice loss.

        Only averages over classes that are actually present in the
        current batch (have at least 1 ground truth pixel). This
        prevents absent classes from contributing a near-1.0 loss
        that dominates the gradient signal.

        Args:
            logits: Raw model output of shape (B, C, H, W).
            targets: Ground truth class indices of shape (B, H, W),
                     dtype long, values in [0, C-1].

        Returns:
            Scalar Dice loss.
        """
        num_classes = logits.shape[1]

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute per-class Dice
        dims = (0, 2, 3)  # aggregate over batch, height, width
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Build mask: exclude ignored class AND classes absent from this batch
        gt_pixel_counts = torch.sum(targets_one_hot, dim=dims)  # (C,)
        present_mask = gt_pixel_counts > 0

        if self.ignore_index is not None and 0 <= self.ignore_index < num_classes:
            present_mask[self.ignore_index] = False

        # Average only over present classes
        if present_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        dice_loss = 1.0 - dice_per_class[present_mask].mean()

        return dice_loss


class CombinedSegmentationLoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for semantic segmentation.

    Computes:
        L = dice_weight * DiceLoss + ce_weight * CrossEntropyLoss

    Args:
        dice_weight: Weight for the Dice loss component.
        ce_weight: Weight for the Cross-Entropy loss component.
        class_weights: Optional per-class weights for CE loss, shape (C,).
                       Computed from training set class frequencies.
        label_smoothing: Label smoothing factor for CE loss. Prevents
                         the model from becoming overconfident by mixing
                         the target distribution with a uniform distribution.
        ignore_index: Class index to ignore in both losses. Set to -1
                      to include all classes.
        dice_smooth: Smoothing constant for Dice computation.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.01,
        ignore_index: int = -1,
        dice_smooth: float = 1.0,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        # Dice loss component
        dice_ignore = None if ignore_index < 0 else ignore_index
        self.dice_loss = DiceLoss(smooth=dice_smooth, ignore_index=dice_ignore)

        # Cross-Entropy loss component
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index if ignore_index >= 0 else -100,
            label_smoothing=label_smoothing,
        )

        # Store for logging
        self._last_dice_loss = 0.0
        self._last_ce_loss = 0.0

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Raw model output of shape (B, C, H, W).
            targets: Ground truth class indices of shape (B, H, W),
                     dtype long, values in [0, C-1].

        Returns:
            Scalar combined loss.
        """
        # Compute individual losses
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)

        # Store for logging
        self._last_dice_loss = dice.item()
        self._last_ce_loss = ce.item()

        # Weighted combination
        total = self.dice_weight * dice + self.ce_weight * ce

        return total

    @property
    def last_dice_loss(self) -> float:
        """Last computed Dice loss value (for logging)."""
        return self._last_dice_loss

    @property
    def last_ce_loss(self) -> float:
        """Last computed CE loss value (for logging)."""
        return self._last_ce_loss


def build_loss_from_config(
    cfg,
    class_weights: Optional[torch.Tensor] = None,
) -> CombinedSegmentationLoss:
    """Factory function to build loss from OmegaConf config.

    Args:
        cfg: OmegaConf configuration object.
        class_weights: Optional per-class weights tensor. If None and
                       class_weights_enabled is True in config, weights
                       should be computed externally and passed here.

    Returns:
        Configured CombinedSegmentationLoss instance.
    """
    loss_cfg = cfg.training.loss

    # Only use class weights if enabled in config
    weights = None
    if loss_cfg.class_weights_enabled and class_weights is not None:
        weights = class_weights

    loss_fn = CombinedSegmentationLoss(
        dice_weight=loss_cfg.dice_weight,
        ce_weight=loss_cfg.ce_weight,
        class_weights=weights,
        label_smoothing=loss_cfg.label_smoothing,
    )

    logger.info(
        "Loss function: Dice(w=%.2f) + CE(w=%.2f), "
        "label_smoothing=%.3f, class_weights=%s",
        loss_cfg.dice_weight,
        loss_cfg.ce_weight,
        loss_cfg.label_smoothing,
        "enabled" if weights is not None else "disabled",
    )

    return loss_fn