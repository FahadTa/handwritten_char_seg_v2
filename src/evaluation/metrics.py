# =============================================================================
# metrics.py
# =============================================================================
# Comprehensive evaluation metrics for character segmentation.
#
# Computes three levels of metrics:
#   1. Pixel-level: IoU, Dice, pixel accuracy, precision, recall, F1
#   2. Per-class: breakdown of each metric by character class
#   3. Per-group: aggregated by uppercase, lowercase, digits, special
#
# All metrics are computed using torchmetrics for GPU acceleration
# and DDP compatibility, plus custom numpy-based metrics for detailed
# per-class analysis after inference is complete.
# =============================================================================

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.charset import (
    NUM_CLASSES,
    get_class_names,
    get_group_indices,
)

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Accumulates predictions and computes segmentation metrics.

    Designed for post-inference evaluation: predictions and ground truth
    masks are accumulated across batches, then all metrics are computed
    at once. This avoids repeated metric computation per batch and
    enables per-class analysis that requires the full confusion matrix.

    Usage:
        metrics = SegmentationMetrics()
        for batch in dataloader:
            preds = model(batch['image']).argmax(dim=1)
            metrics.update(preds, batch['mask'])
        results = metrics.compute()
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = 0):
        """Initialize metric accumulator.

        Args:
            num_classes: Total number of classes including background.
            ignore_index: Class index to exclude from macro-averaged
                         metrics (typically 0 for background).
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Confusion matrix accumulator: shape (num_classes, num_classes)
        # confusion[i, j] = number of pixels with true class i predicted as j
        self._confusion = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )
        self._total_pixels = 0

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._confusion[:] = 0
        self._total_pixels = 0

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate predictions and targets into the confusion matrix.

        Args:
            preds: Predicted class indices, shape (B, H, W).
            targets: Ground truth class indices, shape (B, H, W).
        """
        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        # Clip to valid range
        preds_np = np.clip(preds_np, 0, self.num_classes - 1)
        targets_np = np.clip(targets_np, 0, self.num_classes - 1)

        # Update confusion matrix
        for true_cls, pred_cls in zip(targets_np, preds_np):
            self._confusion[true_cls, pred_cls] += 1

        self._total_pixels += len(preds_np)

    def update_batch_fast(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Fast batch update using bincount (avoids Python loop).

        Args:
            preds: Predicted class indices, shape (B, H, W).
            targets: Ground truth class indices, shape (B, H, W).
        """
        preds_np = preds.detach().cpu().numpy().flatten().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().flatten().astype(np.int64)

        preds_np = np.clip(preds_np, 0, self.num_classes - 1)
        targets_np = np.clip(targets_np, 0, self.num_classes - 1)

        # Flatten to 1D index: true_class * num_classes + pred_class
        indices = targets_np * self.num_classes + preds_np
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self._confusion += counts.reshape(self.num_classes, self.num_classes)

        self._total_pixels += len(preds_np)

    def compute(self) -> Dict[str, Any]:
        """Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dictionary with keys:
                'overall': dict of scalar metrics
                'per_class': dict mapping class name to metric dict
                'per_group': dict mapping group name to metric dict
                'confusion_matrix': numpy array
        """
        results = {
            "overall": self._compute_overall(),
            "per_class": self._compute_per_class(),
            "per_group": self._compute_per_group(),
            "confusion_matrix": self._confusion.copy(),
        }

        return results

    def _compute_overall(self) -> Dict[str, float]:
        """Compute overall (macro-averaged) metrics."""
        cm = self._confusion

        # Per-class TP, FP, FN
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Pixel accuracy (micro, includes background)
        pixel_acc = tp.sum() / max(cm.sum(), 1)

        # Per-class IoU
        iou_per_class = tp / np.maximum(tp + fp + fn, 1).astype(np.float64)

        # Per-class Dice
        dice_per_class = (2 * tp) / np.maximum(2 * tp + fp + fn, 1).astype(np.float64)

        # Per-class Precision, Recall
        precision_per_class = tp / np.maximum(tp + fp, 1).astype(np.float64)
        recall_per_class = tp / np.maximum(tp + fn, 1).astype(np.float64)

        # Macro averages (excluding ignored index)
        mask = np.ones(self.num_classes, dtype=bool)
        if self.ignore_index is not None:
            mask[self.ignore_index] = False

        # Only average over classes that have ground truth samples
        gt_counts = cm.sum(axis=1)
        active_mask = mask & (gt_counts > 0)

        if active_mask.sum() == 0:
            logger.warning("No active classes found for metric computation.")
            return {
                "pixel_accuracy": float(pixel_acc),
                "iou": 0.0,
                "dice": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_active_classes": 0,
            }

        mean_iou = iou_per_class[active_mask].mean()
        mean_dice = dice_per_class[active_mask].mean()
        mean_precision = precision_per_class[active_mask].mean()
        mean_recall = recall_per_class[active_mask].mean()

        # F1 from mean precision and recall
        if mean_precision + mean_recall > 0:
            mean_f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
        else:
            mean_f1 = 0.0

        return {
            "pixel_accuracy": float(pixel_acc),
            "iou": float(mean_iou),
            "dice": float(mean_dice),
            "precision": float(mean_precision),
            "recall": float(mean_recall),
            "f1": float(mean_f1),
            "num_active_classes": int(active_mask.sum()),
        }

    def _compute_per_class(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each individual class."""
        cm = self._confusion
        class_names = get_class_names()

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        gt_counts = cm.sum(axis=1)

        per_class = {}
        for cls_idx in range(self.num_classes):
            name = class_names[cls_idx]

            if gt_counts[cls_idx] == 0:
                continue

            t = tp[cls_idx]
            f_p = fp[cls_idx]
            f_n = fn[cls_idx]

            iou = float(t / max(t + f_p + f_n, 1))
            dice = float(2 * t / max(2 * t + f_p + f_n, 1))
            prec = float(t / max(t + f_p, 1))
            rec = float(t / max(t + f_n, 1))
            f1 = float(2 * prec * rec / max(prec + rec, 1e-8))

            per_class[name] = {
                "iou": iou,
                "dice": dice,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": int(gt_counts[cls_idx]),
            }

        return per_class

    def _compute_per_group(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics aggregated by character group."""
        cm = self._confusion
        groups = get_group_indices()

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        gt_counts = cm.sum(axis=1)

        per_group = {}
        for group_name, (start, end) in groups.items():
            indices = list(range(start, end + 1))

            # Filter to active classes
            active = [i for i in indices if gt_counts[i] > 0]
            if not active:
                continue

            group_tp = tp[active]
            group_fp = fp[active]
            group_fn = fn[active]

            iou_vals = group_tp / np.maximum(group_tp + group_fp + group_fn, 1).astype(np.float64)
            dice_vals = (2 * group_tp) / np.maximum(2 * group_tp + group_fp + group_fn, 1).astype(np.float64)
            prec_vals = group_tp / np.maximum(group_tp + group_fp, 1).astype(np.float64)
            rec_vals = group_tp / np.maximum(group_tp + group_fn, 1).astype(np.float64)

            mean_prec = float(prec_vals.mean())
            mean_rec = float(rec_vals.mean())
            mean_f1 = float(
                2 * mean_prec * mean_rec / max(mean_prec + mean_rec, 1e-8)
            )

            per_group[group_name] = {
                "iou": float(iou_vals.mean()),
                "dice": float(dice_vals.mean()),
                "precision": mean_prec,
                "recall": mean_rec,
                "f1": mean_f1,
                "num_classes": len(active),
                "total_support": int(gt_counts[active].sum()),
            }

        return per_group


def format_metrics_report(results: Dict[str, Any]) -> str:
    """Format evaluation results as a human-readable text report.

    Args:
        results: Output from SegmentationMetrics.compute().

    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("Segmentation Evaluation Report")
    lines.append("=" * 65)

    # Overall metrics
    overall = results["overall"]
    lines.append("")
    lines.append("Overall Metrics (macro-averaged, background excluded):")
    lines.append("-" * 50)
    lines.append(f"  Pixel Accuracy:  {overall['pixel_accuracy']:.4f}  ({overall['pixel_accuracy']*100:.2f}%)")
    lines.append(f"  IoU:             {overall['iou']:.4f}  ({overall['iou']*100:.2f}%)")
    lines.append(f"  Dice:            {overall['dice']:.4f}  ({overall['dice']*100:.2f}%)")
    lines.append(f"  Precision:       {overall['precision']:.4f}  ({overall['precision']*100:.2f}%)")
    lines.append(f"  Recall:          {overall['recall']:.4f}  ({overall['recall']*100:.2f}%)")
    lines.append(f"  F1:              {overall['f1']:.4f}  ({overall['f1']*100:.2f}%)")
    lines.append(f"  Active Classes:  {overall['num_active_classes']}")

    # Per-group metrics
    per_group = results.get("per_group", {})
    if per_group:
        lines.append("")
        lines.append("Per-Group Metrics:")
        lines.append("-" * 65)
        lines.append(f"  {'Group':<12s} {'IoU':>8s} {'Dice':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'#Cls':>5s}")
        lines.append(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
        for group_name, gm in per_group.items():
            lines.append(
                f"  {group_name:<12s} "
                f"{gm['iou']:>7.4f} "
                f"{gm['dice']:>7.4f} "
                f"{gm['precision']:>7.4f} "
                f"{gm['recall']:>7.4f} "
                f"{gm['f1']:>7.4f} "
                f"{gm['num_classes']:>5d}"
            )

    # Per-class top/bottom performers
    per_class = results.get("per_class", {})
    if per_class:
        sorted_by_iou = sorted(
            per_class.items(), key=lambda x: x[1]["iou"], reverse=True
        )

        lines.append("")
        lines.append("Top 10 Classes by IoU:")
        lines.append("-" * 55)
        lines.append(f"  {'Class':<8s} {'IoU':>8s} {'Dice':>8s} {'Prec':>8s} {'Rec':>8s} {'Support':>10s}")
        for name, cm in sorted_by_iou[:10]:
            lines.append(
                f"  {name:<8s} "
                f"{cm['iou']:>7.4f} "
                f"{cm['dice']:>7.4f} "
                f"{cm['precision']:>7.4f} "
                f"{cm['recall']:>7.4f} "
                f"{cm['support']:>10d}"
            )

        lines.append("")
        lines.append("Bottom 10 Classes by IoU:")
        lines.append("-" * 55)
        lines.append(f"  {'Class':<8s} {'IoU':>8s} {'Dice':>8s} {'Prec':>8s} {'Rec':>8s} {'Support':>10s}")
        for name, cm in sorted_by_iou[-10:]:
            lines.append(
                f"  {name:<8s} "
                f"{cm['iou']:>7.4f} "
                f"{cm['dice']:>7.4f} "
                f"{cm['precision']:>7.4f} "
                f"{cm['recall']:>7.4f} "
                f"{cm['support']:>10d}"
            )

    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)