# =============================================================================
# evaluate_iam.py
# =============================================================================
# Evaluate a trained model on the IAM Handwriting Database and produce
# a domain gap analysis comparing synthetic vs. real performance.
#
# Usage:
#   python scripts/evaluate_iam.py --checkpoint outputs/checkpoints/best.ckpt
#   python scripts/evaluate_iam.py --checkpoint best.ckpt --synthetic-results outputs/evaluation/synthetic/metrics.json
# =============================================================================

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf

from src.data.charset import NUM_CLASSES
from src.data.iam_adapter import create_iam_dataloader
from src.evaluation.domain_gap import compute_domain_gap, format_domain_gap_report
from src.evaluation.metrics import SegmentationMetrics, format_metrics_report
from src.evaluation.visualize import (
    save_all_evaluation_plots,
    save_domain_gap_chart,
)
from src.training.lightning_module import CharSegModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model on IAM Handwriting Database.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--iam-root",
        type=str,
        default=None,
        help="Override IAM database root directory.",
    )
    parser.add_argument(
        "--synthetic-results",
        type=str,
        default=None,
        help="Path to synthetic evaluation metrics.json for domain gap analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation/iam",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of IAM samples to evaluate (for quick testing).",
    )
    parser.add_argument(
        "--num-vis-samples",
        type=int,
        default=16,
        help="Number of samples for prediction visualization.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["attention_unet", "swin_unet"],
        help="Override model architecture (must match the checkpoint).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Override architecture if specified
    if args.architecture is not None:
        cfg.model.architecture = args.architecture

    iam_root = args.iam_root or cfg.data.paths.iam_root

    # Verify IAM directory exists
    if not Path(iam_root).exists():
        logger.error(
            "IAM directory not found: %s\n"
            "Please download the IAM Handwriting Database and extract it to this path.\n"
            "Register at: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
            iam_root,
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("IAM Handwriting Database Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("IAM root:   %s", iam_root)

    # Load model
    # strict=False handles class_weights and loss buffers saved during training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharSegModule.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        map_location=device,
        strict=False,
    )
    model.eval()
    model.to(device)

    logger.info("Model loaded: %s (%d parameters)",
                cfg.model.architecture,
                model.model.get_num_parameters())

    # Set up IAM data
    iam_loader, iam_dataset = create_iam_dataloader(
        iam_root=iam_root,
        split=cfg.evaluation.iam.split,
        mode="lines",
        batch_size=args.batch_size,
        num_workers=4,
        image_height=cfg.data.synthetic.image_height,
        image_width=cfg.data.synthetic.image_width,
        binarization_threshold=None,  # Use Otsu
        min_component_area=cfg.evaluation.iam.min_component_area,
        max_samples=args.max_samples,
    )

    if iam_dataset.num_samples == 0:
        logger.error("No IAM samples found. Check directory structure and split files.")
        sys.exit(1)

    logger.info("IAM test set: %d samples", iam_dataset.num_samples)

    # Run evaluation
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=0)

    vis_images = []
    vis_masks = []
    vis_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(iam_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            metrics.update_batch_fast(preds, masks)

            # Collect visualization samples
            if len(vis_images) < args.num_vis_samples:
                for i in range(min(images.shape[0], args.num_vis_samples - len(vis_images))):
                    vis_images.append(images[i, 0].cpu().numpy())
                    vis_masks.append(masks[i].cpu().numpy())
                    vis_preds.append(preds[i].cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                logger.info("  Processed %d/%d batches",
                            batch_idx + 1, len(iam_loader))

    # Compute IAM metrics
    iam_results = metrics.compute()

    # Print IAM report
    report = format_metrics_report(iam_results)
    logger.info("\n%s", report)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save IAM text report
    report_path = output_dir / "iam_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Save IAM metrics as JSON
    iam_json = {
        "overall": iam_results["overall"],
        "per_group": iam_results["per_group"],
    }
    iam_json_path = output_dir / "iam_metrics.json"
    with open(iam_json_path, "w") as f:
        json.dump(iam_json, f, indent=2)

    # Save IAM visualizations
    save_all_evaluation_plots(
        results=iam_results,
        output_dir=str(output_dir),
        prefix="iam_",
        images=vis_images,
        masks=vis_masks,
        preds=vis_preds,
    )

    # =========================================================================
    # Domain Gap Analysis
    # =========================================================================

    if args.synthetic_results:
        synthetic_json_path = Path(args.synthetic_results)
        if synthetic_json_path.exists():
            logger.info("Computing domain gap analysis...")

            with open(synthetic_json_path, "r") as f:
                synthetic_json = json.load(f)

            # Build full result dicts for comparison
            synthetic_results_full = {
                "overall": synthetic_json["overall"],
                "per_group": synthetic_json.get("per_group", {}),
                "per_class": synthetic_json.get("per_class", {}),
            }
            iam_results_full = {
                "overall": iam_results["overall"],
                "per_group": iam_results.get("per_group", {}),
                "per_class": iam_results.get("per_class", {}),
            }

            gap = compute_domain_gap(synthetic_results_full, iam_results_full)

            # Print domain gap report
            gap_report = format_domain_gap_report(gap)
            logger.info("\n%s", gap_report)

            # Save domain gap report
            gap_report_path = output_dir / "domain_gap_report.txt"
            with open(gap_report_path, "w") as f:
                f.write(gap_report)

            # Save domain gap JSON
            gap_json_path = output_dir / "domain_gap.json"
            gap_serializable = {
                "overall": gap["overall"],
                "per_group": gap["per_group"],
            }
            with open(gap_json_path, "w") as f:
                json.dump(gap_serializable, f, indent=2)

            # Save domain gap chart
            save_domain_gap_chart(
                gap=gap,
                save_path=str(output_dir / "domain_gap_chart.png"),
            )

            logger.info("Domain gap analysis saved to: %s", output_dir)
        else:
            logger.warning(
                "Synthetic results not found: %s. "
                "Run evaluate.py first, then pass --synthetic-results.",
                synthetic_json_path,
            )
    else:
        logger.info(
            "No --synthetic-results provided. Skipping domain gap analysis. "
            "To enable, run evaluate.py first, then pass: "
            "--synthetic-results outputs/evaluation/synthetic/metrics.json"
        )

    logger.info("=" * 60)
    logger.info("IAM evaluation complete. Results in: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()