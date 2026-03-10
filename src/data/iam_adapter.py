# =============================================================================
# iam_adapter.py
# =============================================================================
# Adapter for the IAM Handwriting Database to enable real-data evaluation.
#
# Expected directory structure:
#   data/iam/
#       lines/              # Line-level images
#           a01/
#               a01-000u/
#                   a01-000u-00.png
#                   ...
#       ascii/
#           lines.txt       # Line transcriptions
#       split/
#           trainset.txt
#           validationset1.txt
#           validationset2.txt
#           testset.txt
#
# Transcription format in lines.txt:
#   a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from
#   Fields: line_id, segmentation_result, graylevel, num_components,
#           x, y, w, h, transcription (words separated by |)
#
# This adapter:
#   1. Parses lines.txt to get transcriptions
#   2. Loads line images
#   3. Generates pseudo ground-truth segmentation masks via
#      binarization + connected component analysis
#   4. Provides a Dataset interface compatible with evaluation code
# =============================================================================

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.charset import BACKGROUND_INDEX, NUM_CLASSES, char_to_index

logger = logging.getLogger(__name__)


# =============================================================================
# IAM Lines.txt Parsing
# =============================================================================

def parse_lines_txt(lines_txt_path: str) -> Dict[str, Dict]:
    """Parse the IAM lines.txt transcription file.

    Args:
        lines_txt_path: Path to the lines.txt file.

    Returns:
        Dictionary mapping line IDs to metadata:
            {
                'a01-000u-00': {
                    'status': 'ok',
                    'graylevel': 154,
                    'text': 'A MOVE to stop Mr. Gaitskell from',
                    'bbox': (408, 746, 1661, 89),
                },
                ...
            }
    """
    lines = {}

    with open(lines_txt_path, "r") as f:
        for raw_line in f:
            raw_line = raw_line.strip()

            # Skip comments and empty lines
            if not raw_line or raw_line.startswith("#"):
                continue

            parts = raw_line.split(" ")
            if len(parts) < 9:
                continue

            line_id = parts[0]
            status = parts[1]
            graylevel = int(parts[2])
            # parts[3] = num_components
            x = int(parts[4])
            y = int(parts[5])
            w = int(parts[6])
            h = int(parts[7])

            # Transcription: words separated by |, rejoin with spaces
            transcription_raw = " ".join(parts[8:])
            transcription = transcription_raw.replace("|", " ")

            lines[line_id] = {
                "status": status,
                "graylevel": graylevel,
                "text": transcription,
                "bbox": (x, y, w, h),
            }

    logger.info("Parsed %d lines from %s", len(lines), lines_txt_path)
    return lines


def load_iam_splits(split_dir: str) -> Dict[str, List[str]]:
    """Load IAM train/val/test split definitions.

    The split files contain form IDs (e.g., 'a01-000u') one per line.

    Args:
        split_dir: Directory containing split files.

    Returns:
        Dictionary mapping split name to list of form IDs.
    """
    splits = {}
    split_files = {
        "train": ["trainset.txt"],
        "val": ["validationset1.txt", "validationset2.txt"],
        "test": ["testset.txt"],
    }

    for split_name, filenames in split_files.items():
        form_ids = []
        for filename in filenames:
            filepath = Path(split_dir) / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            form_ids.append(line)
        splits[split_name] = form_ids
        logger.info("IAM split '%s': %d forms", split_name, len(form_ids))

    return splits


# =============================================================================
# Pseudo Ground-Truth Mask Generation
# =============================================================================

def generate_pseudo_mask(
    image: np.ndarray,
    transcription: str,
    binarization_threshold: Optional[int] = None,
    min_component_area: int = 10,
) -> np.ndarray:
    """Generate a pseudo ground-truth segmentation mask for a real image.

    Strategy:
        1. Binarize the image to separate ink from background
        2. Find connected components (each is a candidate glyph)
        3. Sort components left-to-right (with line grouping)
        4. Map components to transcription characters sequentially

    Args:
        image: Grayscale image of handwritten text, shape (H, W).
        transcription: The text content of the image.
        binarization_threshold: Fixed threshold. If None, Otsu's method
                                is used for automatic thresholding.
        min_component_area: Minimum pixel area for a connected component
                           to be considered a valid character.

    Returns:
        Segmentation mask of shape (H, W), dtype uint8, with class indices.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Step 1: Binarize
    if binarization_threshold is None:
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            image, binarization_threshold, 255, cv2.THRESH_BINARY_INV
        )

    # Step 2: Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Step 3: Filter small components and sort by position
    components = []
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        cx = centroids[label_idx][0]
        cy = centroids[label_idx][1]

        components.append({
            "label": label_idx,
            "cx": cx,
            "cy": cy,
            "area": area,
        })

    if not components:
        return mask

    # Group components into lines by vertical position and sort
    components = _group_and_sort_components(components, h)

    # Step 4: Extract non-space characters from transcription
    char_labels = []
    for ch in transcription:
        idx = char_to_index(ch)
        if idx != BACKGROUND_INDEX and ch != " ":
            char_labels.append(idx)

    # Step 5: Map components to characters
    num_to_map = min(len(components), len(char_labels))

    for i in range(num_to_map):
        comp = components[i]
        class_idx = char_labels[i]
        component_pixels = labels == comp["label"]
        mask[component_pixels] = class_idx

    return mask


def _group_and_sort_components(
    components: List[Dict],
    image_height: int,
    line_threshold_ratio: float = 0.03,
) -> List[Dict]:
    """Group components into text lines and sort left-to-right within lines.

    Args:
        components: List of component dictionaries with 'cx', 'cy' keys.
        image_height: Image height for computing the line threshold.
        line_threshold_ratio: Fraction of image height used as the
                              vertical grouping threshold.

    Returns:
        Flat list of components sorted in reading order.
    """
    line_threshold = image_height * line_threshold_ratio

    # Sort by vertical position first
    components_sorted = sorted(components, key=lambda c: c["cy"])

    # Group into lines
    lines = []
    current_line = [components_sorted[0]]

    for comp in components_sorted[1:]:
        if abs(comp["cy"] - current_line[-1]["cy"]) < line_threshold:
            current_line.append(comp)
        else:
            lines.append(current_line)
            current_line = [comp]
    lines.append(current_line)

    # Sort each line left-to-right, then flatten
    result = []
    for line in lines:
        line_sorted = sorted(line, key=lambda c: c["cx"])
        result.extend(line_sorted)

    return result


# =============================================================================
# IAM Dataset
# =============================================================================

class IAMDataset(Dataset):
    """Dataset for IAM Handwriting Database evaluation.

    Loads IAM line images, generates pseudo ground-truth masks from
    transcriptions, and resizes to model input dimensions.
    """

    def __init__(
        self,
        iam_root: str,
        split: str = "test",
        image_height: int = 512,
        image_width: int = 512,
        binarization_threshold: Optional[int] = None,
        min_component_area: int = 10,
        max_samples: Optional[int] = None,
    ):
        """Initialize the IAM dataset.

        Args:
            iam_root: Root directory of the IAM database.
            split: Which split to use ('train', 'val', 'test').
            image_height: Target image height.
            image_width: Target image width.
            binarization_threshold: Threshold for mask generation.
            min_component_area: Minimum area for connected components.
            max_samples: Maximum number of samples to load (for debugging).
        """
        self._iam_root = Path(iam_root)
        self._split = split
        self._image_height = image_height
        self._image_width = image_width
        self._binarization_threshold = binarization_threshold
        self._min_component_area = min_component_area

        self._samples: List[Dict] = []
        self._load_samples(max_samples)

    def _load_samples(self, max_samples: Optional[int] = None) -> None:
        """Discover and load sample metadata from the IAM database."""
        lines_dir = self._iam_root / "lines"
        lines_txt = self._iam_root / "ascii" / "lines.txt"
        split_dir = self._iam_root / "split"

        if not lines_dir.exists():
            logger.warning("IAM lines directory not found: %s", lines_dir)
            return

        if not lines_txt.exists():
            logger.warning("IAM lines.txt not found: %s", lines_txt)
            return

        # Parse transcriptions
        all_lines = parse_lines_txt(str(lines_txt))

        # Load split definitions
        split_forms = None
        if split_dir.exists():
            splits = load_iam_splits(str(split_dir))
            split_forms = set(splits.get(self._split, []))
            logger.info(
                "Split '%s' has %d forms",
                self._split,
                len(split_forms) if split_forms else 0,
            )

        # Match line images with transcriptions
        for line_id, line_info in all_lines.items():
            transcription = line_info["text"]
            if not transcription.strip():
                continue

            # Line ID format: a01-000u-00
            parts = line_id.split("-")
            if len(parts) < 3:
                continue

            form_id = f"{parts[0]}-{parts[1]}"

            # Check if this line belongs to the requested split
            # Split files may contain either line IDs (e.g., m01-049-00)
            # or form IDs (e.g., a01-000u). Check both.
            if split_forms is not None:
                if line_id not in split_forms and form_id not in split_forms:
                    continue

            # Construct image path: lines/a01/a01-000u/a01-000u-00.png
            subdir1 = parts[0]
            subdir2 = form_id
            img_filename = f"{line_id}.png"
            img_path = lines_dir / subdir1 / subdir2 / img_filename

            if not img_path.exists():
                continue

            self._samples.append({
                "image_path": str(img_path),
                "transcription": transcription,
                "line_id": line_id,
                "graylevel": line_info.get("graylevel", None),
            })

            if max_samples and len(self._samples) >= max_samples:
                break

        logger.info(
            "IAM dataset (%s): loaded %d samples",
            self._split, len(self._samples),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load an IAM sample with pseudo ground-truth mask.

        If an image cannot be loaded, returns a blank sample instead
        of crashing the DataLoader.

        Returns:
            Dictionary with keys:
                'image': Float tensor of shape (1, H, W) in [0, 1].
                'mask':  Long tensor of shape (H, W) with class indices.
                'transcription': Original transcription string.
                'line_id': IAM line identifier.
        """
        sample = self._samples[idx]

        try:
            # Load grayscale image
            image = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"Failed to load: {sample['image_path']}")

            # Generate pseudo ground-truth mask
            mask = generate_pseudo_mask(
                image=image,
                transcription=sample["transcription"],
                binarization_threshold=self._binarization_threshold,
                min_component_area=self._min_component_area,
            )

            # Resize to model input dimensions
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

        except Exception as exc:
            logger.warning("Skipping sample %s: %s", sample["line_id"], exc)
            # Return blank sample
            image = np.ones(
                (self._image_height, self._image_width), dtype=np.uint8
            ) * 255
            mask = np.zeros(
                (self._image_height, self._image_width), dtype=np.uint8
            )

        # Clamp mask values
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Convert to tensors
        image_tensor = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).unsqueeze(0)

        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "transcription": sample["transcription"],
            "line_id": sample["line_id"],
        }

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    @property
    def transcriptions(self) -> List[str]:
        """Return all transcriptions for analysis."""
        return [s["transcription"] for s in self._samples]


def create_iam_dataloader(
    iam_root: str,
    split: str = "test",
    batch_size: int = 4,
    num_workers: int = 4,
    image_height: int = 512,
    image_width: int = 512,
    binarization_threshold: Optional[int] = None,
    min_component_area: int = 10,
    max_samples: Optional[int] = None,
    **kwargs,
) -> Tuple[DataLoader, IAMDataset]:
    """Create a DataLoader for IAM evaluation.

    Args:
        iam_root: Root directory of the IAM database.
        split: Which split to use.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        image_height: Target image height.
        image_width: Target image width.
        binarization_threshold: Threshold for mask generation.
        min_component_area: Minimum area for connected components.
        max_samples: Maximum number of samples.

    Returns:
        Tuple of (DataLoader, IAMDataset).
    """
    dataset = IAMDataset(
        iam_root=iam_root,
        split=split,
        image_height=image_height,
        image_width=image_width,
        binarization_threshold=binarization_threshold,
        min_component_area=min_component_area,
        max_samples=max_samples,
    )

    def collate_fn(batch):
        images = torch.stack([s["image"] for s in batch])
        masks = torch.stack([s["mask"] for s in batch])
        transcriptions = [s["transcription"] for s in batch]
        line_ids = [s["line_id"] for s in batch]

        return {
            "image": images,
            "mask": masks,
            "transcription": transcriptions,
            "line_id": line_ids,
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return loader, dataset


def create_iam_adapter_from_config(cfg) -> Tuple[DataLoader, IAMDataset]:
    """Factory function to create IAM DataLoader from OmegaConf config."""
    return create_iam_dataloader(
        iam_root=cfg.data.paths.iam_root,
        split=cfg.evaluation.iam.split,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        image_height=cfg.data.synthetic.image_height,
        image_width=cfg.data.synthetic.image_width,
        binarization_threshold=None,
        min_component_area=cfg.evaluation.iam.min_component_area,
    )