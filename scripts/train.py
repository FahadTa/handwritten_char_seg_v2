# =============================================================================
# train.py
# =============================================================================
# CLI entry point for training the segmentation model.
#
# Usage:
#   python scripts/train.py
#   python scripts/train.py --config configs/config.yaml
#   python scripts/train.py --architecture swin_unet
#   python scripts/train.py --devices 1 --fast-dev-run
#
# The script:
#   1. Loads configuration from YAML with CLI overrides
#   2. Initializes W&B experiment tracking
#   3. Sets up data module (datasets + dataloaders)
#   4. Builds the Lightning module (model + loss + metrics)
#   5. Configures the Trainer with callbacks and DDP
#   6. Runs training
# =============================================================================

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data.charset import NUM_CLASSES
from src.data.dataset import CharSegDataModule, create_datamodule_from_config
from src.training.callbacks import build_callbacks
from src.training.lightning_module import CharSegModule

# Suppress noisy warnings in DDP
warnings.filterwarnings("ignore", ".*does not have many workers.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train handwritten character segmentation model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["attention_unet", "swin_unet"],
        help="Override model architecture.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Override number of GPUs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override maximum number of epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size per GPU.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1 train + 1 val batch for debugging.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run W&B in offline mode (no internet required).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for training."""
    args = parse_args()

    # =========================================================================
    # Configuration
    # =========================================================================

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    cfg = OmegaConf.load(str(config_path))

    # Apply CLI overrides
    if args.architecture is not None:
        cfg.model.architecture = args.architecture
    if args.devices is not None:
        cfg.training.devices = args.devices
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.data.loader.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.seed is not None:
        cfg.project.seed = args.seed

    # Handle single GPU case (no DDP needed)
    if cfg.training.devices == 1:
        cfg.training.strategy = "auto"

    # Set seed for reproducibility
    pl.seed_everything(cfg.project.seed, workers=True)

    # =========================================================================
    # Print Training Summary
    # =========================================================================

    logger.info("=" * 60)
    logger.info("Handwritten Character Segmentation v2 - Training")
    logger.info("=" * 60)
    logger.info("Model:         %s", cfg.model.architecture)
    logger.info("Num classes:   %d", cfg.model.num_classes)
    logger.info("Image size:    %dx%d",
                cfg.data.synthetic.image_height,
                cfg.data.synthetic.image_width)
    logger.info("Batch size:    %d per GPU x %d GPUs x %d accum = %d effective",
                cfg.data.loader.batch_size,
                cfg.training.devices,
                cfg.training.accumulate_grad_batches,
                cfg.data.loader.batch_size
                * cfg.training.devices
                * cfg.training.accumulate_grad_batches)
    logger.info("Learning rate: %.1e", cfg.training.learning_rate)
    logger.info("Max epochs:    %d", cfg.training.max_epochs)
    logger.info("Precision:     %s", cfg.training.precision)
    logger.info("Strategy:      %s", cfg.training.strategy)
    logger.info("Seed:          %d", cfg.project.seed)
    logger.info("=" * 60)

    # =========================================================================
    # Data
    # =========================================================================

    logger.info("Setting up data module...")
    data_module = create_datamodule_from_config(cfg)
    data_module.setup(stage="fit")

    # =========================================================================
    # Model
    # =========================================================================

    logger.info("Building model...")
    model = CharSegModule(
        cfg=cfg,
        class_weights=data_module.class_weights,
    )

    # =========================================================================
    # W&B Logger
    # =========================================================================

    wandb_mode = "offline" if args.wandb_offline else "online"
    os.environ.setdefault("WANDB_MODE", wandb_mode)

    wandb_logger = pl.loggers.WandbLogger(
        project=cfg.project.wandb_project,
        entity=cfg.project.wandb_entity,
        name=f"{cfg.model.architecture}_seed{cfg.project.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=False,
        save_dir=cfg.data.paths.output_dir,
    )

    # =========================================================================
    # Callbacks
    # =========================================================================

    callbacks = build_callbacks(cfg)

    # =========================================================================
    # Trainer
    # =========================================================================

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=False,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )

    # =========================================================================
    # Training
    # =========================================================================

    logger.info("Starting training...")

    trainer.fit(
        model=model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
        ckpt_path=args.resume,
    )

    # =========================================================================
    # Post-Training Summary
    # =========================================================================

    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info("Best checkpoint: %s", best_ckpt)
    logger.info("Best val/iou:    %.4f", best_score if best_score else 0.0)
    logger.info("=" * 60)

    # Close W&B run
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()