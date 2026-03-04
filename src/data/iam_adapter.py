# =============================================================================
# iam_adapter.py
# =============================================================================
# Adapter for the IAM Handwriting Database to enable real-data evaluation.
#
# The IAM database structure:
#   iam_root/
#       img/                  # Line or word images
#       xml/                  # XML annotations with transcriptions
#       ascii/                # Text transcription files
#       split/                # Train/val/test split definitions
#
# This adapter:
#   1. Loads IAM line/word images
#   2. Generates pseudo ground-truth segmentation masks by:
#      a. Binarizing the real handwriting image (Otsu threshold)
#      b. Using the transcription to assign class labels
#      c. Using connected component analysis to segment individual
#         characters from left to right
#      d. Matching components to transcription characters sequentially
#   3. Resizes everything to the model's expected input resolution
#   4. Provides a Dataset interface compatible with our evaluation code
#
# IMPORTANT: The generated masks are pseudo ground-truth. Real handwriting
# does not have pixel-perfect character-level annotations. The masks are
# approximate and intended for domain-gap analysis, not for computing
# absolute accuracy numbers on real data. The evaluation report will
# clearly state this limitation.
# =============================================================================

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.charset import BACKGROUND_INDEX, NUM_CLASSES, char_to_index

logger = logging.getLogger(__name__)


# =============================================================================
# IAM XML Parsing
# =============================================================================

def parse_iam_xml(xml_path: str) -> Dict[str, Dict]:
    """Parse an IAM XML annotation file to extract line-level information.

    Args:
        xml_path: Path to the XML file.

    Returns:
        Dictionary mapping line IDs to their metadata:
            {
                line_id: {
                    'text': transcription string,
                    'words': [
                        {'text': word_str, 'x': int, 'y': int,
                         'w': int, 'h': int},
                        ...
                    ]
                }
            }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = {}

    for line_elem in root.iter("line"):
        line_id = line_elem.get("id", "")
        line_text = line_elem.get("text", "")

        # Unescape common XML entities in IAM transcriptions
        line_text = line_text.replace("&amp;", "&")
        line_text = line_text.replace("&quot;", '"')
        line_text = line_text.replace("&apos;", "'")
        line_text = line_text.replace("&lt;", "<")
        line_text = line_text.replace("&gt;", ">")

        words = []
        for word_elem in line_elem.iter("word"):
            word_text = word_elem.get("text", "")
            word_text = word_text.replace("&amp;", "&")
            word_text = word_text.replace("&quot;", '"')
            word_text = word_text.replace("&apos;", "'")

            # Bounding box (may not always be present)
            x = int(word_elem.get("x", 0))
            y = int(word_elem.get("y", 0))
            w = int(word_elem.get("w", 0))
            h = int(word_elem.get("h", 0))

            words.append({
                "text": word_text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })

        lines[line_id] = {
            "text": line_text,
            "words": words,
        }

    return lines


def load_iam_splits(split_dir: str) -> Dict[str, List[str]]:
    """Load IAM train/val/test split definitions.

    The split files contain form IDs (e.g., 'a01-000u') one per line.
    These are used to determine which lines belong to which split.

    Args:
        split_dir: Directory containing split files (trainset.txt,
                   validationset1.txt, testset.txt, etc.).

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

    This is approximate because:
        - Connected components may merge touching characters
        - Components may split a single character (e.g., dotted 'i')
        - The sequential mapping assumes left-to-right reading order

    These limitations are acceptable for domain-gap analysis, where
    we care about relative performance degradation rather than
    absolute accuracy.

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
    for label_idx in range(1, num_labels):  # skip background (0)
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        cx = centroids[label_idx][0]
        cy = centroids[label_idx][1]
        x = stats[label_idx, cv2.CC_STAT_LEFT]
        y = stats[label_idx, cv2.CC_STAT_TOP]
        comp_w = stats[label_idx, cv2.CC_STAT_WIDTH]
        comp_h = stats[label_idx, cv2.CC_STAT_HEIGHT]

        components.append({
            "label": label_idx,
            "cx": cx,
            "cy": cy,
            "x": x,
            "y": y,
            "w": comp_w,
            "h": comp_h,
            "area": area,
        })

    if not components:
        return mask

    # Group components into lines by vertical position
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

    # Any remaining components beyond the transcription length
    # are left as background (0)

    return mask


def _group_and_sort_components(
    components: List[Dict],
    image_height: int,
    line_threshold_ratio: float = 0.03,
) -> List[Dict]:
    """Group components into text lines and sort left-to-right within lines.

    Components are grouped by their vertical centroid. Components whose
    centroids are within a threshold distance vertically are considered
    to be on the same line. Lines are sorted top-to-bottom, and
    components within each line are sorted left-to-right.

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

    Supports two modes:
        1. 'lines': Load line-level images from the IAM lines/ directory
        2. 'words': Load word-level images from the IAM words/ directory
    """

    def __init__(
        self,
        iam_root: str,
        split: str = "test",
        mode: str = "lines",
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
            mode: 'lines' or 'words'.
            image_height: Target image height.
            image_width: Target image width.
            binarization_threshold: Threshold for mask generation.
            min_component_area: Minimum area for connected components.
            max_samples: Maximum number of samples to load (for debugging).
        """
        self._iam_root = Path(iam_root)
        self._split = split
        self._mode = mode
        self._image_height = image_height
        self._image_width = image_width
        self._binarization_threshold = binarization_threshold
        self._min_component_area = min_component_area

        self._samples: List[Dict] = []
        self._load_samples(max_samples)

    def _load_samples(self, max_samples: Optional[int] = None) -> None:
        """Discover and load sample metadata from the IAM database.

        Scans XML files to find lines/words in the requested split,
        then verifies that the corresponding image files exist.
        """
        xml_dir = self._iam_root / "xml"
        img_base_dir = self._iam_root / self._mode

        if not xml_dir.exists():
            logger.warning("IAM xml directory not found: %s", xml_dir)
            return

        if not img_base_dir.exists():
            logger.warning("IAM image directory not found: %s", img_base_dir)
            return

        # Load split definitions if available
        split_dir = self._iam_root / "split"
        split_forms = None
        if split_dir.exists():
            splits = load_iam_splits(str(split_dir))
            split_forms = set(splits.get(self._split, []))

        # Parse all XML files
        for xml_file in sorted(xml_dir.glob("*.xml")):
            form_id = xml_file.stem

            # Filter by split if split info is available
            if split_forms is not None and form_id not in split_forms:
                continue

            try:
                lines = parse_iam_xml(str(xml_file))
            except ET.ParseError:
                logger.warning("Failed to parse XML: %s", xml_file)
                continue

            for line_id, line_info in lines.items():
                transcription = line_info["text"]
                if not transcription.strip():
                    continue

                # Construct image path following IAM directory structure
                # IAM lines path: lines/a01/a01-000u/a01-000u-00.png
                parts = line_id.split("-")
                if len(parts) < 3:
                    continue

                subdir1 = parts[0]
                subdir2 = f"{parts[0]}-{parts[1]}"
                img_filename = f"{line_id}.png"
                img_path = img_base_dir / subdir1 / subdir2 / img_filename

                if not img_path.exists():
                    continue

                self._samples.append({
                    "image_path": str(img_path),
                    "transcription": transcription,
                    "line_id": line_id,
                })

                if max_samples and len(self._samples) >= max_samples:
                    break

            if max_samples and len(self._samples) >= max_samples:
                break

        logger.info(
            "IAM dataset (%s, %s): loaded %d samples",
            self._split, self._mode, len(self._samples),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load an IAM sample with pseudo ground-truth mask.

        Returns:
            Dictionary with keys:
                'image': Float tensor of shape (1, H, W) in [0, 1].
                'mask':  Long tensor of shape (H, W) with class indices.
                'transcription': Original transcription string.
                'line_id': IAM line identifier.
        """
        sample = self._samples[idx]

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

        # Clamp mask values
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Convert to tensors
        image_tensor = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).unsqueeze(0)  # (1, H, W)

        mask_tensor = torch.from_numpy(mask.astype(np.int64))  # (H, W)

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
    mode: str = "lines",
    batch_size: int = 4,
    num_workers: int = 4,
    image_height: int = 512,
    image_width: int = 512,
    binarization_threshold: Optional[int] = None,
    min_component_area: int = 10,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, IAMDataset]:
    """Create a DataLoader for IAM evaluation.

    Returns both the DataLoader and the Dataset for access to
    metadata (transcriptions, line IDs).

    Args:
        iam_root: Root directory of the IAM database.
        split: Which split to use.
        mode: 'lines' or 'words'.
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
        mode=mode,
        image_height=image_height,
        image_width=image_width,
        binarization_threshold=binarization_threshold,
        min_component_area=min_component_area,
        max_samples=max_samples,
    )

    # Custom collate to handle string fields
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
    """Factory function to create IAM DataLoader from OmegaConf config.

    Args:
        cfg: OmegaConf configuration object (the full config).

    Returns:
        Tuple of (DataLoader, IAMDataset).
    """
    return create_iam_dataloader(
        iam_root=cfg.data.paths.iam_root,
        split=cfg.evaluation.iam.split,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        image_height=cfg.data.synthetic.image_height,
        image_width=cfg.data.synthetic.image_width,
        binarization_threshold=cfg.evaluation.iam.binarization_threshold,
        min_component_area=cfg.evaluation.iam.min_component_area,
    )