# =============================================================================
# lightning_module.py
# =============================================================================
# PyTorch Lightning module for handwritten character segmentation.
#
# This module encapsulates the full training and validation logic:
#   - Forward pass through the segmentation model
#   - Loss computation (combined Dice + CE)
#   - Optimizer and scheduler configuration
#   - Metric computation (IoU, Dice, pixel accuracy) via torchmetrics
#   - W&B logging of losses, metrics, and sample predictions
#
# Design decisions:
#   - Metrics are computed using torchmetrics for GPU-accelerated,
#     DDP-compatible aggregation across devices.
#   - Sample predictions are logged as W&B images at the end of each
#     validation epoch for qualitative monitoring.
#   - The model architecture is selected at init time based on config,
#     so a single Lightning module works for both U-Net and SwinUNet.
# =============================================================================

import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.data.charset import NUM_CLASSES, get_class_names
from src.models.loss import CombinedSegmentationLoss, build_loss_from_config
from src.models.unet import AttentionUNet, build_attention_unet_from_config
from src.models.swin_unet import SwinUNet, build_swin_unet_from_config

logger = logging.getLogger(__name__)


class CharSegModule(pl.LightningModule):
    """Lightning module for handwritten character segmentation.

    Handles model instantiation, training/validation steps, optimizer
    and scheduler configuration, and metric logging.

    Args:
        cfg: OmegaConf configuration object.
        class_weights: Optional per-class weights for loss computation.
    """

    def __init__(
        self,
        cfg: Any,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(ignore=["class_weights"])

        # Build model based on architecture choice
        self.model = self._build_model()

        # Build loss function
        self.loss_fn = build_loss_from_config(cfg, class_weights)

        # Store class weights for potential device transfer
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)

        # Metrics for training
        self.train_iou = MulticlassJaccardIndex(
            num_classes=NUM_CLASSES, average="macro", ignore_index=0,
        )
        self.train_dice = MulticlassF1Score(
            num_classes=NUM_CLASSES, average="macro", ignore_index=0,
        )

        # Metrics for validation
        self.val_iou = MulticlassJaccardIndex(
            num_classes=NUM_CLASSES, average="macro", ignore_index=0,
        )
        self.val_dice = MulticlassF1Score(
            num_classes=NUM_CLASSES, average="macro", ignore_index=0,
        )
        self.val_pixel_acc = MulticlassAccuracy(
            num_classes=NUM_CLASSES, average="micro",
        )
        self.val_precision = MulticlassPrecision(
            num_classes=NUM_CLASSES, average="macro", ignore_index=0,
        )
        self.val_recall = MulticlassRecall(
            num_classes=NUM_CLASSES, average="macro", ignore_index=0,
        )

        # Cache for validation sample predictions (for visualization)
        self._val_samples: list = []
        self._max_vis_samples = 8

        logger.info(
            "Initialized CharSegModule with %s (%d parameters)",
            cfg.model.architecture,
            self.model.get_num_parameters(),
        )

    def _build_model(self) -> nn.Module:
        """Instantiate the segmentation model based on config."""
        architecture = self.cfg.model.architecture

        if architecture == "attention_unet":
            return build_attention_unet_from_config(self.cfg)
        elif architecture == "swin_unet":
            return build_swin_unet_from_config(self.cfg)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input images, shape (B, 1, H, W).

        Returns:
            Logits, shape (B, num_classes, H, W).
        """
        return self.model(x)

    # =========================================================================
    # Training
    # =========================================================================

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute a single training step.

        Args:
            batch: Dictionary with 'image' and 'mask' tensors.
            batch_idx: Index of the current batch.

        Returns:
            Scalar training loss.
        """
        images = batch["image"]
        masks = batch["mask"]

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = self.loss_fn(logits, masks)

        # Compute predictions for metrics
        preds = logits.argmax(dim=1)

        # Update metrics
        self.train_iou.update(preds, masks)
        self.train_dice.update(preds, masks)

        # Log losses
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("train/dice_loss", self.loss_fn.last_dice_loss,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/ce_loss", self.loss_fn.last_ce_loss,
                 on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log training metrics at the end of each epoch."""
        self.log("train/iou", self.train_iou.compute(),
                 prog_bar=True, sync_dist=True)
        self.log("train/dice", self.train_dice.compute(),
                 sync_dist=True)

        self.train_iou.reset()
        self.train_dice.reset()

    # =========================================================================
    # Validation
    # =========================================================================

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Execute a single validation step.

        Args:
            batch: Dictionary with 'image' and 'mask' tensors.
            batch_idx: Index of the current batch.
        """
        images = batch["image"]
        masks = batch["mask"]

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = self.loss_fn(logits, masks)

        # Compute predictions
        preds = logits.argmax(dim=1)

        # Update metrics
        self.val_iou.update(preds, masks)
        self.val_dice.update(preds, masks)
        self.val_pixel_acc.update(preds, masks)
        self.val_precision.update(preds, masks)
        self.val_recall.update(preds, masks)

        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("val/dice_loss", self.loss_fn.last_dice_loss,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/ce_loss", self.loss_fn.last_ce_loss,
                 on_step=False, on_epoch=True, sync_dist=True)

        # Cache samples for visualization (first batch only, rank 0)
        if batch_idx == 0 and len(self._val_samples) < self._max_vis_samples:
            num_to_save = min(
                self._max_vis_samples - len(self._val_samples),
                images.shape[0],
            )
            for i in range(num_to_save):
                self._val_samples.append({
                    "image": images[i].detach().cpu(),
                    "mask": masks[i].detach().cpu(),
                    "pred": preds[i].detach().cpu(),
                })

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics and sample predictions."""
        # Log scalar metrics
        val_iou = self.val_iou.compute()
        val_dice = self.val_dice.compute()

        self.log("val/iou", val_iou, prog_bar=True, sync_dist=True)
        self.log("val/dice", val_dice, sync_dist=True)
        self.log("val/pixel_accuracy", self.val_pixel_acc.compute(),
                 sync_dist=True)
        self.log("val/precision", self.val_precision.compute(),
                 sync_dist=True)
        self.log("val/recall", self.val_recall.compute(),
                 sync_dist=True)

        # Log sample predictions to W&B
        self._log_prediction_samples()

        # Reset metrics and sample cache
        self.val_iou.reset()
        self.val_dice.reset()
        self.val_pixel_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self._val_samples.clear()

    def _log_prediction_samples(self) -> None:
        """Log prediction visualizations to W&B.

        Creates a grid of [input, ground truth, prediction] for
        qualitative monitoring during training.
        """
        if not self._val_samples:
            return

        # Only log from rank 0 in DDP
        if self.global_rank != 0:
            return

        wandb_logger = self._get_wandb_logger()
        if wandb_logger is None:
            return

        try:
            import wandb
            import numpy as np

            images = []
            for sample in self._val_samples[:4]:
                img = sample["image"].squeeze(0).numpy()
                mask = sample["mask"].numpy()
                pred = sample["pred"].numpy()

                # Normalize image to [0, 255] for display
                img_display = (img * 255).astype(np.uint8)

                # Create colored mask and prediction visualizations
                mask_colored = self._colorize_mask(mask)
                pred_colored = self._colorize_mask(pred)

                images.append(wandb.Image(
                    img_display,
                    caption="Input",
                ))
                images.append(wandb.Image(
                    mask_colored,
                    caption="Ground Truth",
                ))
                images.append(wandb.Image(
                    pred_colored,
                    caption="Prediction",
                ))

            wandb_logger.experiment.log(
                {"val/predictions": images},
                step=self.global_step,
            )
        except Exception as exc:
            logger.warning("Failed to log prediction samples: %s", exc)

    def _get_wandb_logger(self):
        """Retrieve the W&B logger from the trainer's loggers."""
        if self.trainer is None:
            return None

        for lg in self.trainer.loggers:
            if isinstance(lg, pl.loggers.WandbLogger):
                return lg
        return None

    @staticmethod
    def _colorize_mask(mask, max_classes: int = NUM_CLASSES):
        """Convert a class-index mask to an RGB visualization.

        Uses a fixed color palette where each class gets a distinct
        color. Background (class 0) is black.

        Args:
            mask: 2D numpy array of class indices.
            max_classes: Maximum number of classes for the palette.

        Returns:
            RGB numpy array of shape (H, W, 3), dtype uint8.
        """
        import numpy as np

        # Generate deterministic color palette
        np.random.seed(42)
        palette = np.random.randint(50, 256, size=(max_classes, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]  # background is black
        np.random.seed(None)

        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx in range(max_classes):
            colored[mask == cls_idx] = palette[cls_idx]

        return colored

    # =========================================================================
    # Optimizer and Scheduler
    # =========================================================================

    def configure_optimizers(self) -> Dict:
        """Configure optimizer and learning rate scheduler.

        Uses AdamW with cosine annealing, as specified in the config.
        Returns a dictionary compatible with Lightning's optimizer API.
        """
        train_cfg = self.cfg.training

        # Build optimizer
        if train_cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
                betas=(0.9, 0.999),
            )
        elif train_cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )
        elif train_cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")

        # Build scheduler
        if train_cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.max_epochs,
                eta_min=train_cfg.min_learning_rate,
            )
        elif train_cfg.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=10,
                min_lr=train_cfg.min_learning_rate,
            )
        elif train_cfg.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            raise ValueError(f"Unknown scheduler: {train_cfg.scheduler}")

        # Log current learning rate
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        if train_cfg.scheduler == "plateau":
            scheduler_config["monitor"] = train_cfg.early_stopping.monitor

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }