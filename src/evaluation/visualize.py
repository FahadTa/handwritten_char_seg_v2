# =============================================================================
# visualize.py
# =============================================================================
# Visualization utilities for evaluation results.
#
# Generates:
#   1. Prediction overlay grids (input / ground truth / prediction / overlay)
#   2. Confusion matrix heatmaps
#   3. Per-class and per-group metric bar charts
#   4. Domain gap comparison charts
#   5. All figures saved as PNG files for the project report
# =============================================================================

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from src.data.charset import NUM_CLASSES, get_class_names, get_group_indices

logger = logging.getLogger(__name__)

# Deterministic color palette for class visualization
np.random.seed(42)
CLASS_PALETTE = np.random.randint(50, 256, size=(NUM_CLASSES, 3), dtype=np.uint8)
CLASS_PALETTE[0] = [0, 0, 0]  # background is black
np.random.seed(None)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class-index mask to RGB visualization.

    Args:
        mask: 2D array of class indices, shape (H, W).

    Returns:
        RGB array of shape (H, W, 3), dtype uint8.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(NUM_CLASSES):
        rgb[mask == cls_idx] = CLASS_PALETTE[cls_idx]
    return rgb


def save_prediction_grid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    preds: List[np.ndarray],
    save_path: str,
    num_samples: int = 8,
    overlay_alpha: float = 0.5,
) -> None:
    """Save a grid of prediction visualizations.

    Each row shows: Input | Ground Truth | Prediction | Overlay

    Args:
        images: List of grayscale images, each shape (H, W).
        masks: List of ground truth masks, each shape (H, W).
        preds: List of predicted masks, each shape (H, W).
        save_path: Output file path.
        num_samples: Number of rows to display.
        overlay_alpha: Alpha blending for overlay.
    """
    n = min(num_samples, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth", "Prediction", "Overlay"]

    for row in range(n):
        img = images[row]
        gt = masks[row]
        pred = preds[row]

        # Normalize image for display
        if img.max() <= 1.0:
            img_display = (img * 255).astype(np.uint8)
        else:
            img_display = img.astype(np.uint8)

        gt_colored = colorize_mask(gt)
        pred_colored = colorize_mask(pred)

        # Create overlay: blend prediction colors onto the grayscale image
        img_rgb = np.stack([img_display] * 3, axis=-1)
        overlay = (
            img_rgb.astype(np.float32) * (1 - overlay_alpha)
            + pred_colored.astype(np.float32) * overlay_alpha
        ).astype(np.uint8)

        axes[row, 0].imshow(img_display, cmap="gray")
        axes[row, 1].imshow(gt_colored)
        axes[row, 2].imshow(pred_colored)
        axes[row, 3].imshow(overlay)

        for col in range(4):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction grid: %s", save_path)


def save_confusion_matrix(
    confusion: np.ndarray,
    save_path: str,
    normalize: bool = True,
    max_classes: int = 30,
) -> None:
    """Save confusion matrix as a heatmap.

    For readability, only the top N most frequent classes are shown.

    Args:
        confusion: Confusion matrix of shape (num_classes, num_classes).
        save_path: Output file path.
        normalize: Whether to normalize rows (show proportions).
        max_classes: Maximum number of classes to display.
    """
    class_names = get_class_names()

    # Find most active classes (by total ground truth count)
    gt_counts = confusion.sum(axis=1)
    active_indices = np.where(gt_counts > 0)[0]

    # Sort by frequency, take top N
    sorted_indices = active_indices[np.argsort(gt_counts[active_indices])[::-1]]
    display_indices = sorted_indices[:max_classes]

    # Extract submatrix
    sub_matrix = confusion[np.ix_(display_indices, display_indices)]
    display_names = [class_names[i] for i in display_indices]

    if normalize:
        row_sums = sub_matrix.sum(axis=1, keepdims=True)
        sub_matrix = sub_matrix.astype(np.float64) / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(max(12, max_classes * 0.5), max(10, max_classes * 0.4)))
    sns.heatmap(
        sub_matrix,
        xticklabels=display_names,
        yticklabels=display_names,
        cmap="Blues",
        fmt=".2f" if normalize else "d",
        annot=max_classes <= 20,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (Top Classes)", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", save_path)


def save_per_group_chart(
    results: Dict[str, Any],
    save_path: str,
    metric: str = "iou",
) -> None:
    """Save per-group metric comparison bar chart.

    Args:
        results: Output from SegmentationMetrics.compute().
        save_path: Output file path.
        metric: Which metric to plot ('iou', 'dice', 'f1').
    """
    per_group = results.get("per_group", {})
    if not per_group:
        return

    groups = list(per_group.keys())
    values = [per_group[g].get(metric, 0.0) for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(groups, values, color=sns.color_palette("muted", len(groups)))

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Per-Group {metric.upper()}", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-group chart: %s", save_path)


def save_domain_gap_chart(
    gap: Dict[str, Any],
    save_path: str,
) -> None:
    """Save domain gap comparison bar chart.

    Shows synthetic vs. IAM performance side-by-side for each metric.

    Args:
        gap: Output from compute_domain_gap().
        save_path: Output file path.
    """
    overall = gap.get("overall", {})
    if not overall:
        return

    metrics = ["iou", "dice", "precision", "recall", "f1"]
    syn_vals = [overall[m]["synthetic"] for m in metrics if m in overall]
    iam_vals = [overall[m]["iam"] for m in metrics if m in overall]
    metric_labels = [m.upper() for m in metrics if m in overall]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_syn = ax.bar(x - width / 2, syn_vals, width, label="Synthetic", color="#4C72B0")
    bars_iam = ax.bar(x + width / 2, iam_vals, width, label="IAM", color="#DD8452")

    # Add value labels
    for bars in [bars_syn, bars_iam]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Domain Gap: Synthetic vs. IAM Handwriting", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved domain gap chart: %s", save_path)


def save_all_evaluation_plots(
    results: Dict[str, Any],
    output_dir: str,
    prefix: str = "",
    images: Optional[List[np.ndarray]] = None,
    masks: Optional[List[np.ndarray]] = None,
    preds: Optional[List[np.ndarray]] = None,
) -> List[str]:
    """Generate and save all evaluation plots.

    Args:
        results: Output from SegmentationMetrics.compute().
        output_dir: Directory to save plots.
        prefix: Optional filename prefix (e.g., 'synthetic_' or 'iam_').
        images: Optional list of input images for prediction grid.
        masks: Optional list of ground truth masks.
        preds: Optional list of predictions.

    Returns:
        List of saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # Prediction grid
    if images and masks and preds:
        path = str(out / f"{prefix}predictions.png")
        save_prediction_grid(images, masks, preds, path)
        saved.append(path)

    # Confusion matrix
    if "confusion_matrix" in results:
        path = str(out / f"{prefix}confusion_matrix.png")
        save_confusion_matrix(results["confusion_matrix"], path)
        saved.append(path)

    # Per-group charts
    for metric in ["iou", "dice", "f1"]:
        path = str(out / f"{prefix}per_group_{metric}.png")
        save_per_group_chart(results, path, metric=metric)
        saved.append(path)

    return saved