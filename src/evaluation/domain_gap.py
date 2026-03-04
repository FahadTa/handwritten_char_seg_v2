# =============================================================================
# domain_gap.py
# =============================================================================
# Domain gap analysis: compares model performance on synthetic data vs.
# real handwriting (IAM) to quantify the distribution shift.
#
# Produces:
#   1. Side-by-side metric comparison table
#   2. Per-group performance degradation
#   3. Per-class degradation ranking
#   4. Failure case identification
#   5. Summary report suitable for the project writeup
# =============================================================================

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.charset import get_class_names, get_group_indices

logger = logging.getLogger(__name__)


def compute_domain_gap(
    synthetic_results: Dict[str, Any],
    iam_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the domain gap between synthetic and real data performance.

    Args:
        synthetic_results: Output from SegmentationMetrics.compute()
                          on the synthetic test set.
        iam_results: Output from SegmentationMetrics.compute()
                    on the IAM test set.

    Returns:
        Dictionary with gap analysis results.
    """
    gap = {}

    # Overall metric comparison
    gap["overall"] = _compare_overall(
        synthetic_results["overall"],
        iam_results["overall"],
    )

    # Per-group comparison
    gap["per_group"] = _compare_groups(
        synthetic_results.get("per_group", {}),
        iam_results.get("per_group", {}),
    )

    # Per-class degradation ranking
    gap["per_class_degradation"] = _compute_class_degradation(
        synthetic_results.get("per_class", {}),
        iam_results.get("per_class", {}),
    )

    return gap


def _compare_overall(
    synthetic: Dict[str, float],
    iam: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Compare overall metrics between synthetic and IAM."""
    comparison = {}

    metrics = ["pixel_accuracy", "iou", "dice", "precision", "recall", "f1"]

    for metric in metrics:
        syn_val = synthetic.get(metric, 0.0)
        iam_val = iam.get(metric, 0.0)
        absolute_drop = syn_val - iam_val
        relative_drop = absolute_drop / max(syn_val, 1e-8)

        comparison[metric] = {
            "synthetic": syn_val,
            "iam": iam_val,
            "absolute_drop": absolute_drop,
            "relative_drop": relative_drop,
        }

    return comparison


def _compare_groups(
    synthetic_groups: Dict[str, Dict],
    iam_groups: Dict[str, Dict],
) -> Dict[str, Dict[str, Any]]:
    """Compare per-group metrics between synthetic and IAM."""
    comparison = {}

    all_groups = set(list(synthetic_groups.keys()) + list(iam_groups.keys()))

    for group_name in sorted(all_groups):
        syn = synthetic_groups.get(group_name, {})
        iam = iam_groups.get(group_name, {})

        syn_iou = syn.get("iou", 0.0)
        iam_iou = iam.get("iou", 0.0)

        comparison[group_name] = {
            "synthetic_iou": syn_iou,
            "iam_iou": iam_iou,
            "iou_drop": syn_iou - iam_iou,
            "synthetic_dice": syn.get("dice", 0.0),
            "iam_dice": iam.get("dice", 0.0),
            "dice_drop": syn.get("dice", 0.0) - iam.get("dice", 0.0),
        }

    return comparison


def _compute_class_degradation(
    synthetic_classes: Dict[str, Dict],
    iam_classes: Dict[str, Dict],
) -> List[Dict[str, Any]]:
    """Rank classes by performance degradation from synthetic to IAM.

    Returns a list sorted by IoU degradation (largest drop first).
    """
    degradation = []

    all_classes = set(list(synthetic_classes.keys()) + list(iam_classes.keys()))

    for class_name in all_classes:
        syn = synthetic_classes.get(class_name, {})
        iam = iam_classes.get(class_name, {})

        syn_iou = syn.get("iou", 0.0)
        iam_iou = iam.get("iou", 0.0)

        degradation.append({
            "class": class_name,
            "synthetic_iou": syn_iou,
            "iam_iou": iam_iou,
            "iou_drop": syn_iou - iam_iou,
            "synthetic_dice": syn.get("dice", 0.0),
            "iam_dice": iam.get("dice", 0.0),
        })

    # Sort by largest IoU drop
    degradation.sort(key=lambda x: x["iou_drop"], reverse=True)

    return degradation


def format_domain_gap_report(gap: Dict[str, Any]) -> str:
    """Format domain gap analysis as a human-readable report.

    Args:
        gap: Output from compute_domain_gap().

    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Domain Gap Analysis: Synthetic vs. IAM Handwriting")
    lines.append("=" * 70)

    # Overall comparison
    overall = gap.get("overall", {})
    if overall:
        lines.append("")
        lines.append("Overall Metric Comparison:")
        lines.append("-" * 70)
        lines.append(
            f"  {'Metric':<18s} {'Synthetic':>10s} {'IAM':>10s} "
            f"{'Drop':>10s} {'Rel Drop':>10s}"
        )
        lines.append(
            f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}"
        )

        for metric, values in overall.items():
            lines.append(
                f"  {metric:<18s} "
                f"{values['synthetic']:>9.4f} "
                f"{values['iam']:>9.4f} "
                f"{values['absolute_drop']:>+9.4f} "
                f"{values['relative_drop']:>9.1%}"
            )

    # Per-group comparison
    per_group = gap.get("per_group", {})
    if per_group:
        lines.append("")
        lines.append("Per-Group IoU Comparison:")
        lines.append("-" * 60)
        lines.append(
            f"  {'Group':<12s} {'Syn IoU':>10s} {'IAM IoU':>10s} {'Drop':>10s}"
        )
        lines.append(
            f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}"
        )

        for group_name, values in per_group.items():
            lines.append(
                f"  {group_name:<12s} "
                f"{values['synthetic_iou']:>9.4f} "
                f"{values['iam_iou']:>9.4f} "
                f"{values['iou_drop']:>+9.4f}"
            )

    # Most degraded classes
    degradation = gap.get("per_class_degradation", [])
    if degradation:
        lines.append("")
        lines.append("Top 10 Most Degraded Classes (by IoU drop):")
        lines.append("-" * 60)
        lines.append(
            f"  {'Class':<8s} {'Syn IoU':>10s} {'IAM IoU':>10s} {'Drop':>10s}"
        )

        for entry in degradation[:10]:
            lines.append(
                f"  {entry['class']:<8s} "
                f"{entry['synthetic_iou']:>9.4f} "
                f"{entry['iam_iou']:>9.4f} "
                f"{entry['iou_drop']:>+9.4f}"
            )

        # Least degraded (or improved)
        lines.append("")
        lines.append("Top 10 Least Degraded Classes:")
        lines.append("-" * 60)
        lines.append(
            f"  {'Class':<8s} {'Syn IoU':>10s} {'IAM IoU':>10s} {'Drop':>10s}"
        )

        for entry in degradation[-10:]:
            lines.append(
                f"  {entry['class']:<8s} "
                f"{entry['synthetic_iou']:>9.4f} "
                f"{entry['iam_iou']:>9.4f} "
                f"{entry['iou_drop']:>+9.4f}"
            )

    # Summary statistics
    if degradation:
        drops = [e["iou_drop"] for e in degradation]
        lines.append("")
        lines.append("Domain Gap Summary:")
        lines.append("-" * 40)
        lines.append(f"  Mean IoU drop:     {np.mean(drops):>+.4f}")
        lines.append(f"  Median IoU drop:   {np.median(drops):>+.4f}")
        lines.append(f"  Max IoU drop:      {np.max(drops):>+.4f}")
        lines.append(f"  Min IoU drop:      {np.min(drops):>+.4f}")
        lines.append(f"  Std IoU drop:      {np.std(drops):>.4f}")

    lines.append("")
    lines.append("NOTE: IAM metrics use pseudo ground-truth masks generated")
    lines.append("via binarization + connected component analysis. These are")
    lines.append("approximate and intended for relative domain gap assessment,")
    lines.append("not absolute accuracy measurement on real data.")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)