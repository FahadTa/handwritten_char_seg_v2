# =============================================================================
# callbacks.py
# =============================================================================
# Custom and configured Lightning callbacks for training.
#
# Provides a single factory function that returns all callbacks needed
# by the Trainer, configured from the project YAML config.
# =============================================================================

import logging
from typing import Any, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

logger = logging.getLogger(__name__)


class LogModelSummaryCallback(pl.Callback):
    """Log model parameter summary at the start of training.

    Prints the per-component parameter breakdown to the console
    and to W&B (if available) for reference.
    """

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if hasattr(pl_module, "model") and hasattr(pl_module.model, "get_parameter_summary"):
            summary = pl_module.model.get_parameter_summary()
            logger.info("\n%s", summary)

            # Log to W&B as a text artifact
            for lg in trainer.loggers:
                if isinstance(lg, pl.loggers.WandbLogger):
                    try:
                        lg.experiment.summary["model_summary"] = summary
                    except Exception:
                        pass


class GradientNormLogger(pl.Callback):
    """Log gradient norms for monitoring training stability.

    High gradient norms can indicate exploding gradients or learning
    rate issues. Logged every N steps to avoid excessive overhead.
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self._log_every = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Any,
    ) -> None:
        if trainer.global_step % self._log_every != 0:
            return

        total_norm = 0.0
        for param in pl_module.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        pl_module.log(
            "train/gradient_norm",
            total_norm,
            on_step=True,
            on_epoch=False,
        )


def build_callbacks(cfg) -> List[pl.Callback]:
    """Build all training callbacks from config.

    Args:
        cfg: OmegaConf configuration object.

    Returns:
        List of configured Lightning callbacks.
    """
    callbacks = []

    # Model checkpointing
    ckpt_cfg = cfg.training.checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.data.paths.output_dir}/checkpoints",
        filename="epoch_{epoch:03d}_iou_{val/iou:.4f}",
        monitor=ckpt_cfg.monitor,
        mode=ckpt_cfg.mode,
        save_top_k=ckpt_cfg.save_top_k,
        save_last=ckpt_cfg.save_last,
        auto_insert_metric_name=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    es_cfg = cfg.training.early_stopping
    early_stopping = EarlyStopping(
        monitor=es_cfg.monitor,
        patience=es_cfg.patience,
        mode=es_cfg.mode,
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Model summary logger
    callbacks.append(LogModelSummaryCallback())

    # Gradient norm logger
    callbacks.append(GradientNormLogger(
        log_every_n_steps=cfg.training.log_every_n_steps,
    ))

    # Rich progress bar
    callbacks.append(RichProgressBar())

    logger.info("Configured %d training callbacks.", len(callbacks))

    return callbacks