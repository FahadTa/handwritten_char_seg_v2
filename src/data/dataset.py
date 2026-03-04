# =============================================================================
# dataset.py
# =============================================================================
# PyTorch Dataset and Lightning DataModule for the synthetic handwritten
# character segmentation data.
#
# Design decisions:
#   - The Dataset loads pre-generated images and masks from disk (PNG files
#     produced by synthetic_generator.py).
#   - Augmentations are applied on-the-fly during __getitem__, not
#     pre-computed. This maximizes effective data diversity at the cost
#     of slightly higher CPU load per batch.
#   - Images are normalized to [0, 1] float32 and unsqueezed to (1, H, W)
#     for the single grayscale channel.
#   - Masks remain as int64 tensors of shape (H, W) with values in
#     [0, NUM_CLASSES-1], which is what CrossEntropyLoss expects.
#   - The DataModule encapsulates all data loading logic and is passed
#     directly to the Lightning Trainer.
# =============================================================================

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import (
    build_augmentation_from_config,
    build_train_augmentation,
    build_val_augmentation,
)
from src.data.charset import NUM_CLASSES

logger = logging.getLogger(__name__)


class CharSegDataset(Dataset):
    """Dataset for handwritten character segmentation.

    Loads grayscale images and their corresponding segmentation masks
    from disk, applies augmentations, and returns normalized tensors.

    Directory structure expected:
        root/
            images/
                000000.png
                000001.png
                ...
            masks/
                000000.png
                000001.png
                ...

    Image and mask files are matched by filename.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        image_height: int = 512,
        image_width: int = 512,
    ):
        """Initialize the dataset.

        Args:
            root_dir: Path to the split directory containing 'images/'
                      and 'masks/' subdirectories.
            transform: Albumentations augmentation pipeline. If None,
                       no augmentation is applied.
            image_height: Expected image height for validation.
            image_width: Expected image width for validation.
        """
        self._root_dir = Path(root_dir)
        self._transform = transform
        self._image_height = image_height
        self._image_width = image_width

        self._images_dir = self._root_dir / "images"
        self._masks_dir = self._root_dir / "masks"

        if not self._images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self._images_dir}"
            )
        if not self._masks_dir.exists():
            raise FileNotFoundError(
                f"Masks directory not found: {self._masks_dir}"
            )

        # Discover and sort image files
        self._image_paths = sorted(self._images_dir.glob("*.png"))
        if not self._image_paths:
            raise FileNotFoundError(
                f"No PNG images found in: {self._images_dir}"
            )

        # Verify each image has a corresponding mask
        self._mask_paths = []
        for img_path in self._image_paths:
            mask_path = self._masks_dir / img_path.name
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for image {img_path.name}: "
                    f"expected {mask_path}"
                )
            self._mask_paths.append(mask_path)

        logger.info(
            "Dataset loaded: %d samples from %s",
            len(self._image_paths),
            self._root_dir,
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and preprocess a single sample.

        Returns:
            Dictionary with keys:
                'image': Float tensor of shape (1, H, W) in [0, 1].
                'mask':  Long tensor of shape (H, W) with class indices.
        """
        # Load image as grayscale
        image = cv2.imread(str(self._image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to load image: {self._image_paths[idx]}")

        # Load mask as-is (single channel, values are class indices)
        mask = cv2.imread(str(self._mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {self._mask_paths[idx]}")

        # Resize if dimensions do not match expected size
        if image.shape[0] != self._image_height or image.shape[1] != self._image_width:
            image = cv2.resize(
                image,
                (self._image_width, self._image_height),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = cv2.resize(
                mask,
                (self._image_width, self._image_height),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply augmentations (jointly to image and mask)
        if self._transform is not None:
            augmented = self._transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Clamp mask values to valid range
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Convert image to float32 tensor, normalize to [0, 1]
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0)  # (1, H, W)

        # Convert mask to int64 tensor
        mask_tensor = torch.from_numpy(mask.astype(np.int64))  # (H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }

    @property
    def num_samples(self) -> int:
        return len(self._image_paths)


# =============================================================================
# Class Weight Computation
# =============================================================================

def compute_class_weights(
    dataset: CharSegDataset,
    num_classes: int = NUM_CLASSES,
    max_samples: int = 500,
    smoothing: float = 1e-3,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset subset.

    Scans a random subset of the dataset to estimate pixel-level class
    frequencies, then computes inverse-frequency weights. Classes with
    zero frequency receive a weight of 0.0 (they are effectively ignored
    in the loss).

    The background class (index 0) receives a reduced weight because it
    dominates the pixel count and would otherwise overwhelm character
    class gradients.

    Args:
        dataset: The dataset to scan.
        num_classes: Total number of classes.
        max_samples: Maximum number of samples to scan.
        smoothing: Smoothing constant to prevent division by zero.

    Returns:
        Float tensor of shape (num_classes,) with per-class weights.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    num_to_scan = min(max_samples, len(dataset))

    # Sample indices without replacement
    indices = np.random.choice(len(dataset), size=num_to_scan, replace=False)

    for idx in indices:
        sample = dataset[idx]
        mask = sample["mask"].numpy()
        for cls in range(num_classes):
            counts[cls] += np.sum(mask == cls)

    # Compute inverse frequency weights
    total_pixels = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float64)

    for cls in range(num_classes):
        if counts[cls] > 0:
            weights[cls] = total_pixels / (num_classes * counts[cls] + smoothing)
        else:
            weights[cls] = 0.0

    # Reduce background weight (class 0 dominates pixel counts)
    if weights[0] > 0:
        weights[0] *= 0.1

    # Normalize so mean of non-zero weights is 1.0
    nonzero_mask = weights > 0
    if nonzero_mask.any():
        mean_weight = weights[nonzero_mask].mean()
        if mean_weight > 0:
            weights[nonzero_mask] /= mean_weight

    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# Lightning DataModule
# =============================================================================

class CharSegDataModule:
    """PyTorch Lightning-compatible DataModule for character segmentation.

    Encapsulates dataset creation, augmentation pipeline construction,
    and dataloader configuration. Passed directly to the Lightning Trainer.

    Note: We do not subclass pl.LightningDataModule to avoid importing
    pytorch_lightning at the module level (keeps this file testable
    without GPU/Lightning dependencies). The Trainer accepts any object
    with setup() and train_dataloader/val_dataloader/test_dataloader methods.
    """

    def __init__(
        self,
        synthetic_root: str,
        batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_height: int = 512,
        image_width: int = 512,
        augmentation_config: Optional[dict] = None,
    ):
        """Initialize the DataModule.

        Args:
            synthetic_root: Root directory of the synthetic dataset,
                           containing 'train/', 'val/', 'test/' subdirs.
            batch_size: Batch size per GPU.
            num_workers: Number of data loading workers per GPU.
            pin_memory: Whether to pin tensor memory for faster GPU transfer.
            persistent_workers: Keep workers alive between epochs.
            image_height: Image height.
            image_width: Image width.
            augmentation_config: Optional dict with augmentation parameters.
                                If None, default augmentation settings are used.
        """
        self._synthetic_root = Path(synthetic_root)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers and num_workers > 0
        self._image_height = image_height
        self._image_width = image_width
        self._augmentation_config = augmentation_config or {}

        self.train_dataset: Optional[CharSegDataset] = None
        self.val_dataset: Optional[CharSegDataset] = None
        self.test_dataset: Optional[CharSegDataset] = None
        self.class_weights: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets for the requested stage.

        Args:
            stage: One of 'fit', 'validate', 'test', or None (all).
        """
        train_transform = build_train_augmentation(**self._augmentation_config)
        val_transform = build_val_augmentation(
            image_height=self._image_height,
            image_width=self._image_width,
        )

        if stage in ("fit", None):
            train_dir = self._synthetic_root / "train"
            if train_dir.exists():
                self.train_dataset = CharSegDataset(
                    root_dir=str(train_dir),
                    transform=train_transform,
                    image_height=self._image_height,
                    image_width=self._image_width,
                )
                logger.info(
                    "Train dataset: %d samples", self.train_dataset.num_samples
                )

                # Compute class weights from training data
                self.class_weights = compute_class_weights(self.train_dataset)
                logger.info("Class weights computed from training data.")

        if stage in ("fit", "validate", None):
            val_dir = self._synthetic_root / "val"
            if val_dir.exists():
                self.val_dataset = CharSegDataset(
                    root_dir=str(val_dir),
                    transform=val_transform,
                    image_height=self._image_height,
                    image_width=self._image_width,
                )
                logger.info(
                    "Val dataset: %d samples", self.val_dataset.num_samples
                )

        if stage in ("test", None):
            test_dir = self._synthetic_root / "test"
            if test_dir.exists():
                self.test_dataset = CharSegDataset(
                    root_dir=str(test_dir),
                    transform=val_transform,
                    image_height=self._image_height,
                    image_width=self._image_width,
                )
                logger.info(
                    "Test dataset: %d samples", self.test_dataset.num_samples
                )

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")

        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader().")

        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader().")

        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            drop_last=False,
        )


def create_datamodule_from_config(cfg) -> CharSegDataModule:
    """Factory function to create a CharSegDataModule from OmegaConf config.

    Args:
        cfg: OmegaConf configuration object (the full config).

    Returns:
        Configured CharSegDataModule instance.
    """
    aug_cfg = cfg.data.augmentation

    # Build augmentation kwargs only if augmentation is enabled
    augmentation_config = {}
    if aug_cfg.enabled:
        augmentation_config = {
            "rotation_limit": aug_cfg.rotation_limit,
            "perspective_limit": aug_cfg.perspective_limit,
            "elastic_alpha": aug_cfg.elastic_alpha,
            "elastic_sigma": aug_cfg.elastic_sigma,
            "erosion_kernel_max": aug_cfg.erosion_kernel_max,
            "dilation_kernel_max": aug_cfg.dilation_kernel_max,
            "brightness_limit": aug_cfg.brightness_limit,
            "contrast_limit": aug_cfg.contrast_limit,
            "gaussian_noise_var_limit": aug_cfg.gaussian_noise_var_limit,
            "gaussian_blur_limit": aug_cfg.gaussian_blur_limit,
            "p_geometric": aug_cfg.p_geometric,
            "p_morphological": aug_cfg.p_morphological,
            "p_photometric": aug_cfg.p_photometric,
            "p_noise": aug_cfg.p_noise,
            "image_height": cfg.data.synthetic.image_height,
            "image_width": cfg.data.synthetic.image_width,
        }

    return CharSegDataModule(
        synthetic_root=cfg.data.paths.synthetic_root,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory,
        persistent_workers=cfg.data.loader.persistent_workers,
        image_height=cfg.data.synthetic.image_height,
        image_width=cfg.data.synthetic.image_width,
        augmentation_config=augmentation_config,
    )