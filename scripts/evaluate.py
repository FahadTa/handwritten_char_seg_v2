# =============================================================================
# evaluate.py
# =============================================================================
# Evaluate a trained model on the synthetic test set.
#
# Usage:
#   python scripts/evaluate.py --checkpoint outputs/checkpoints/best.ckpt
#   python scripts/evaluate.py --checkpoint outputs/checkpoints/best.ckpt --devices 1
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
from src.data.dataset import CharSegDataModule
from src.evaluation.metrics import SegmentationMetrics, format_metrics_report
from src.evaluation.visualize import save_all_evaluation_plots
from src.training.lightning_module import CharSegModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model on synthetic test set.",
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
        "--output-dir",
        type=str,
        default="outputs/evaluation/synthetic",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs (use 1 for evaluation).",
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
    cfg.data.loader.batch_size = args.batch_size

    # Override architecture if specified
    if args.architecture is not None:
        cfg.model.architecture = args.architecture

    logger.info("=" * 60)
    logger.info("Synthetic Test Set Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Architecture: %s", cfg.model.architecture)

    # Load model from checkpoint
    # strict=False handles class_weights and loss buffers that are saved
    # in the checkpoint but not present when loading without training data
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

    # Set up test data
    data_module = CharSegDataModule(
        synthetic_root=cfg.data.paths.synthetic_root,
        batch_size=args.batch_size,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=False,
        persistent_workers=False,
        image_height=cfg.data.synthetic.image_height,
        image_width=cfg.data.synthetic.image_width,
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    logger.info("Test set: %d samples", data_module.test_dataset.num_samples)

    # Run evaluation
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=0)

    vis_images = []
    vis_masks = []
    vis_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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

            if (batch_idx + 1) % 20 == 0:
                logger.info("  Processed %d/%d batches",
                            batch_idx + 1, len(test_loader))

    # Compute metrics
    results = metrics.compute()

    # Print report
    report = format_metrics_report(results)
    logger.info("\n%s", report)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report saved: %s", report_path)

    # Save metrics as JSON
    json_results = {
        "overall": results["overall"],
        "per_group": results["per_group"],
    }
    json_path = output_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("Metrics JSON saved: %s", json_path)

    # Save visualizations
    saved_plots = save_all_evaluation_plots(
        results=results,
        output_dir=str(output_dir),
        prefix="synthetic_",
        images=vis_images,
        masks=vis_masks,
        preds=vis_preds,
    )
    logger.info("Saved %d visualization plots.", len(saved_plots))

    logger.info("=" * 60)
    logger.info("Evaluation complete. Results in: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()