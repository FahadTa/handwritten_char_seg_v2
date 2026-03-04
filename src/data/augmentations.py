# =============================================================================
# augmentations.py
# =============================================================================
# Augmentation pipeline for training. All geometric transforms are applied
# jointly to both image and mask via Albumentations.
#
# Augmentation groups (as requested by Dr. Christlein):
#   1. Affine transforms: rotation, scaling, translation, shear
#   2. Elastic deformation: simulates natural handwriting distortion
#   3. Morphological: erosion (thins strokes), dilation (thickens strokes)
#   4. Noise: Gaussian noise, Gaussian blur
#   5. Photometric: brightness, contrast adjustments
#
# Note: Slant/cursive augmentation is handled at render time in
# synthetic_generator.py, not here. This module handles post-render
# augmentations applied during training data loading.
#
# Design decisions:
#   - Morphological erosion/dilation are implemented as custom Albumentations
#     transforms so they integrate cleanly into the pipeline.
#   - Erosion is applied ONLY to the image (thins ink strokes visually) but
#     NOT to the mask, because the mask should reflect the original character
#     footprint. The mask is updated via a re-binarization step after erosion.
#   - Dilation is applied ONLY to the image similarly.
#   - All geometric transforms (affine, elastic, perspective) are applied to
#     both image and mask jointly by Albumentations.
#   - Photometric transforms (brightness, contrast, noise, blur) are applied
#     ONLY to the image, as they do not affect segmentation labels.
# =============================================================================

import logging
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Transforms
# =============================================================================

class MorphologicalErosion(ImageOnlyTransform):
    """Apply morphological erosion to simulate thin pen strokes.

    Erosion shrinks the foreground (dark) regions of the image, making
    strokes appear thinner. This simulates writing with a fine-tipped pen
    or light pen pressure.

    Applied only to the image, not the mask.
    """

    def __init__(
        self,
        kernel_size_max: int = 2,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.kernel_size_max = kernel_size_max

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        kernel_size = np.random.randint(1, self.kernel_size_max + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        # For a grayscale image with dark text on light background,
        # dilation in image space = erosion of the dark foreground
        # We want to thin the dark strokes, so we dilate (lighten)
        return cv2.dilate(img, kernel, iterations=1)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("kernel_size_max",)


class MorphologicalDilation(ImageOnlyTransform):
    """Apply morphological dilation to simulate thick pen strokes.

    Dilation expands the foreground (dark) regions of the image, making
    strokes appear thicker. This simulates writing with a broad-tipped pen
    or heavy pen pressure.

    Applied only to the image, not the mask.
    """

    def __init__(
        self,
        kernel_size_max: int = 2,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.kernel_size_max = kernel_size_max

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        kernel_size = np.random.randint(1, self.kernel_size_max + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        # Erode in image space = dilate the dark foreground strokes
        return cv2.erode(img, kernel, iterations=1)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("kernel_size_max",)


class SimulatedInkVariation(ImageOnlyTransform):
    """Simulate ink density variation along strokes.

    Applies localized brightness perturbation to foreground pixels only,
    simulating uneven ink flow in a real pen.

    Applied only to the image, not the mask.
    """

    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.7, 1.3),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.intensity_range = intensity_range

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        # Identify foreground pixels (dark text on light background)
        threshold = 200
        foreground_mask = img < threshold

        if not np.any(foreground_mask):
            return img

        result = img.copy()

        # Generate smooth random intensity map
        h, w = img.shape[:2]
        small_h, small_w = max(1, h // 16), max(1, w // 16)
        noise = np.random.uniform(
            self.intensity_range[0],
            self.intensity_range[1],
            size=(small_h, small_w),
        ).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply intensity variation only to foreground
        fg_values = result[foreground_mask].astype(np.float32)
        fg_values = fg_values * noise[foreground_mask]
        fg_values = np.clip(fg_values, 0, 255).astype(np.uint8)
        result[foreground_mask] = fg_values

        return result

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("intensity_range",)


# =============================================================================
# Pipeline Construction
# =============================================================================

def build_train_augmentation(
    rotation_limit: int = 3,
    perspective_limit: float = 0.05,
    elastic_alpha: float = 30.0,
    elastic_sigma: float = 5.0,
    erosion_kernel_max: int = 2,
    dilation_kernel_max: int = 2,
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    gaussian_noise_var_limit: float = 15.0,
    gaussian_blur_limit: int = 3,
    p_geometric: float = 0.5,
    p_morphological: float = 0.3,
    p_photometric: float = 0.5,
    p_noise: float = 0.3,
    image_height: int = 512,
    image_width: int = 512,
) -> A.Compose:
    """Build the training augmentation pipeline.

    The pipeline is organized into four sequential groups:
        1. Geometric transforms (applied to both image and mask)
        2. Morphological transforms (image only)
        3. Photometric transforms (image only)
        4. Noise transforms (image only)

    Geometric transforms modify spatial layout, so they must be applied
    jointly to image and mask. All other transforms only affect pixel
    intensities and are image-only.

    Args:
        rotation_limit: Maximum rotation in degrees.
        perspective_limit: Perspective distortion scale.
        elastic_alpha: Elastic deformation magnitude.
        elastic_sigma: Elastic deformation smoothness.
        erosion_kernel_max: Maximum kernel size for erosion.
        dilation_kernel_max: Maximum kernel size for dilation.
        brightness_limit: Brightness adjustment range.
        contrast_limit: Contrast adjustment range.
        gaussian_noise_var_limit: Gaussian noise variance limit.
        gaussian_blur_limit: Maximum Gaussian blur kernel size.
        p_geometric: Probability of applying geometric group.
        p_morphological: Probability of applying morphological group.
        p_photometric: Probability of applying photometric group.
        p_noise: Probability of applying noise group.
        image_height: Target image height (for resize safety).
        image_width: Target image width (for resize safety).

    Returns:
        Albumentations Compose pipeline configured for joint
        image-mask augmentation.
    """
    # Ensure blur limit is odd
    if gaussian_blur_limit % 2 == 0:
        gaussian_blur_limit += 1

    transform = A.Compose(
        [
            # -----------------------------------------------------------------
            # Group 1: Geometric transforms (joint image + mask)
            # -----------------------------------------------------------------
            A.OneOf(
                [
                    A.Affine(
                        rotate=(-rotation_limit, rotation_limit),
                        shear=(-5, 5),
                        scale=(0.95, 1.05),
                        translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                        border_mode=cv2.BORDER_CONSTANT,
                        value=255,
                        mask_value=0,
                        p=1.0,
                    ),
                    A.Perspective(
                        scale=(0.02, perspective_limit),
                        border_mode=cv2.BORDER_CONSTANT,
                        value=255,
                        mask_value=0,
                        p=1.0,
                    ),
                ],
                p=p_geometric,
            ),

            A.ElasticTransform(
                alpha=elastic_alpha,
                sigma=elastic_sigma,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                mask_value=0,
                p=p_geometric * 0.5,
            ),

            # -----------------------------------------------------------------
            # Group 2: Morphological transforms (image only)
            # -----------------------------------------------------------------
            A.OneOf(
                [
                    MorphologicalErosion(
                        kernel_size_max=erosion_kernel_max,
                        p=1.0,
                    ),
                    MorphologicalDilation(
                        kernel_size_max=dilation_kernel_max,
                        p=1.0,
                    ),
                ],
                p=p_morphological,
            ),

            SimulatedInkVariation(
                intensity_range=(0.7, 1.3),
                p=p_morphological * 0.5,
            ),

            # -----------------------------------------------------------------
            # Group 3: Photometric transforms (image only)
            # -----------------------------------------------------------------
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=contrast_limit,
                        p=1.0,
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=1.0,
                    ),
                ],
                p=p_photometric,
            ),

            # -----------------------------------------------------------------
            # Group 4: Noise and blur (image only)
            # -----------------------------------------------------------------
            A.OneOf(
                [
                    A.GaussNoise(
                        var_limit=(0, gaussian_noise_var_limit),
                        p=1.0,
                    ),
                    A.GaussianBlur(
                        blur_limit=(3, gaussian_blur_limit),
                        p=1.0,
                    ),
                ],
                p=p_noise,
            ),
        ],
    )

    return transform


def build_val_augmentation(
    image_height: int = 512,
    image_width: int = 512,
) -> A.Compose:
    """Build the validation/test augmentation pipeline.

    No augmentations are applied during validation. This pipeline exists
    for API consistency so the dataset class can always call transform()
    without conditional logic.

    Args:
        image_height: Target image height.
        image_width: Target image width.

    Returns:
        Albumentations Compose pipeline (identity transform).
    """
    transform = A.Compose([])
    return transform


def build_augmentation_from_config(cfg, is_train: bool = True) -> A.Compose:
    """Factory function to build augmentation pipeline from OmegaConf config.

    Args:
        cfg: OmegaConf configuration object (the full config).
        is_train: If True, build training augmentations. If False,
                  build validation augmentations (no-op).

    Returns:
        Albumentations Compose pipeline.
    """
    aug_cfg = cfg.data.augmentation
    syn_cfg = cfg.data.synthetic

    if not is_train or not aug_cfg.enabled:
        return build_val_augmentation(
            image_height=syn_cfg.image_height,
            image_width=syn_cfg.image_width,
        )

    return build_train_augmentation(
        rotation_limit=aug_cfg.rotation_limit,
        perspective_limit=aug_cfg.perspective_limit,
        elastic_alpha=aug_cfg.elastic_alpha,
        elastic_sigma=aug_cfg.elastic_sigma,
        erosion_kernel_max=aug_cfg.erosion_kernel_max,
        dilation_kernel_max=aug_cfg.dilation_kernel_max,
        brightness_limit=aug_cfg.brightness_limit,
        contrast_limit=aug_cfg.contrast_limit,
        gaussian_noise_var_limit=aug_cfg.gaussian_noise_var_limit,
        gaussian_blur_limit=aug_cfg.gaussian_blur_limit,
        p_geometric=aug_cfg.p_geometric,
        p_morphological=aug_cfg.p_morphological,
        p_photometric=aug_cfg.p_photometric,
        p_noise=aug_cfg.p_noise,
        image_height=syn_cfg.image_height,
        image_width=syn_cfg.image_width,
    )