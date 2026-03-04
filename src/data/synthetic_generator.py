# =============================================================================
# synthetic_generator.py
# =============================================================================
# Generates synthetic handwritten text images with pixel-level segmentation
# masks. This is the core data pipeline for training.
#
# Key design decisions:
#   1. Masks are generated via binarization + AND (not bounding box fills).
#      Each character is rendered in isolation, binarized, and the resulting
#      pixel mask is assigned the character's class index. This produces
#      masks that follow the actual glyph shape.
#
#   2. Slant augmentation is applied at render time via affine shear. The
#      same shear matrix is applied to both image and mask, ensuring
#      perfect alignment. This simulates cursive handwriting.
#
#   3. Text is sourced from Wikipedia for linguistic diversity.
#
#   4. Multiple handwritten fonts are sampled randomly per image to increase
#      style variation.
# =============================================================================

import logging
import os
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.data.charset import CHARSET, BACKGROUND_INDEX, char_to_index

logger = logging.getLogger(__name__)


# =============================================================================
# Text Sampling
# =============================================================================

# Fallback corpus used when Wikipedia is unavailable (e.g., no internet on HPC)
_FALLBACK_CORPUS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
    "Two driven jocks help fax my big quiz.",
    "A large fawn jumped quickly over white zinc boxes.",
    "The jay, pig, fox, zebra and my wolves quack!",
    "Bright vixens jump; dozy fowl quack.",
    "Jackdaws love my big sphinx of quartz.",
    "Crazy Frederick bought many very exquisite opal jewels.",
    "We promptly judged antique ivory buckles for the next prize.",
    "Sixty zippers were quickly picked from the woven jute bag.",
    "Grumpy wizards make toxic brew for the evil queen and jack.",
    "Jim quickly realized that the beautiful gowns are expensive.",
    "The wizard quickly jinxed the gnomes before they vaporized.",
    "All questions asked by five watched experts amaze the judge.",
    "Jack quietly moved up front and seized the big ball of wax.",
    "How razorback jumping frogs can level six piqued gymnasts!",
    "A quick movement of the enemy will jeopardize six gunboats.",
    "Few quips galvanized the mock jury box.",
    "The quick onyx goblin jumps over the lazy dwarf.",
    "Waxy and quivering, jocks fumble the pizza.",
    "My girl wove six dozen plaid jackets before she quit.",
    "Playing jazz vibe chords quickly excites my wife.",
    "Brawny gods just flocked up to quiz and vex him.",
    "Adjusting quiver and bow, Zephyr killed the fox.",
    "The exploration of Mars continues to captivate scientists worldwide.",
    "Quantum computing promises to revolutionize data processing methods.",
    "Historical archives contain invaluable records of ancient civilizations.",
]


class TextSampler:
    """Provides text samples for rendering onto synthetic images.

    Attempts to use Wikipedia articles for diverse text. Falls back to
    a built-in pangram corpus if Wikipedia is unavailable.
    """

    def __init__(self, source: str = "wikipedia", cache_size: int = 200):
        self._source = source
        self._cache: List[str] = []
        self._cache_size = cache_size
        self._fallback_corpus = list(_FALLBACK_CORPUS)
        self._wiki_available = False

        if source == "wikipedia":
            self._try_init_wikipedia()

        if not self._wiki_available:
            logger.info("Using fallback text corpus (no Wikipedia access).")
            self._cache = list(self._fallback_corpus)

    def _try_init_wikipedia(self) -> None:
        """Attempt to initialize the Wikipedia API client."""
        try:
            import wikipediaapi
            self._wiki = wikipediaapi.Wikipedia(
                user_agent="HandwrittenCharSegV2/1.0",
                language="en",
            )
            self._wiki_available = True
            self._prefetch_articles()
            logger.info(
                "Wikipedia API initialized. Cached %d text chunks.",
                len(self._cache),
            )
        except ImportError:
            logger.warning("wikipedia-api not installed. Using fallback corpus.")
        except Exception as exc:
            logger.warning("Wikipedia init failed: %s. Using fallback corpus.", exc)

    def _prefetch_articles(self) -> None:
        """Fetch a batch of Wikipedia articles and split into paragraphs."""
        seed_titles = [
            "Python (programming language)", "Machine learning",
            "History of Europe", "Solar System", "Classical music",
            "Olympic Games", "Artificial intelligence", "Photography",
            "World War II", "Renaissance", "Quantum mechanics",
            "Climate change", "Human trafficking", "Biodiversity",
            "Internet", "Mathematics", "Philosophy", "Literature",
            "Architecture", "Economics", "Medicine", "Chemistry",
        ]
        random.shuffle(seed_titles)

        for title in seed_titles:
            if len(self._cache) >= self._cache_size:
                break
            try:
                page = self._wiki.page(title)
                if page.exists():
                    text = page.text
                    paragraphs = [
                        p.strip() for p in text.split("\n")
                        if len(p.strip()) > 50
                    ]
                    self._cache.extend(paragraphs)
            except Exception:
                continue

        if not self._cache:
            self._cache = list(self._fallback_corpus)

    def sample(self, min_chars: int = 40, max_chars: int = 300) -> str:
        """Return a random text snippet within the specified length range.

        Args:
            min_chars: Minimum number of characters in the returned text.
            max_chars: Maximum number of characters in the returned text.

        Returns:
            A text string suitable for rendering.
        """
        source_text = random.choice(self._cache)

        # If the source text is shorter than min_chars, concatenate multiple
        if len(source_text) < min_chars:
            while len(source_text) < min_chars:
                source_text += " " + random.choice(self._cache)

        # Truncate to max_chars at a word boundary
        if len(source_text) > max_chars:
            truncated = source_text[:max_chars]
            last_space = truncated.rfind(" ")
            if last_space > min_chars:
                truncated = truncated[:last_space]
            source_text = truncated

        # Filter to characters in our charset only
        filtered = []
        for ch in source_text:
            if ch in CHARSET:
                filtered.append(ch)
            elif ch in ("\n", "\t", "\r"):
                filtered.append(" ")
            # else: drop the character silently

        result = "".join(filtered)

        # Collapse multiple consecutive spaces
        while "  " in result:
            result = result.replace("  ", " ")

        return result.strip()


# =============================================================================
# Font Management
# =============================================================================

class FontManager:
    """Discovers and serves handwritten fonts from the fonts directory."""

    def __init__(self, fonts_dir: str):
        self._fonts_dir = Path(fonts_dir)
        self._font_paths: List[Path] = []
        self._discover_fonts()

    def _discover_fonts(self) -> None:
        """Scan the fonts directory for .ttf and .otf files."""
        if not self._fonts_dir.exists():
            raise FileNotFoundError(
                f"Fonts directory not found: {self._fonts_dir}"
            )

        extensions = {".ttf", ".otf"}
        for path in sorted(self._fonts_dir.iterdir()):
            if path.suffix.lower() in extensions:
                self._font_paths.append(path)

        if not self._font_paths:
            raise FileNotFoundError(
                f"No .ttf or .otf fonts found in: {self._fonts_dir}"
            )

        logger.info("Discovered %d fonts in %s", len(self._font_paths), self._fonts_dir)

    def load_font(self, size: int, font_path: Optional[Path] = None) -> ImageFont.FreeTypeFont:
        """Load a font at the specified size.

        Args:
            size: Font size in pixels.
            font_path: Specific font to load. If None, a random font is chosen.

        Returns:
            A PIL FreeTypeFont object.
        """
        if font_path is None:
            font_path = random.choice(self._font_paths)

        return ImageFont.truetype(str(font_path), size=size)

    @property
    def num_fonts(self) -> int:
        return len(self._font_paths)

    @property
    def font_paths(self) -> List[Path]:
        return list(self._font_paths)


# =============================================================================
# Slant Transform
# =============================================================================

def apply_slant_transform(
    image: np.ndarray,
    slant_angle: float,
    border_value: int = 255,
) -> np.ndarray:
    """Apply horizontal shear to simulate cursive slant.

    The shear is applied as an affine transformation:
        x' = x + slant * y
        y' = y

    Args:
        image: Input image as numpy array (H, W) or (H, W, C).
        slant_angle: Shear factor. Positive values slant right,
                     negative values slant left. Typical range: [-0.4, 0.4].
        border_value: Value to fill new border pixels with.

    Returns:
        Sheared image with the same dimensions as input.
    """
    h, w = image.shape[:2]

    # Affine shear matrix: [[1, slant, 0], [0, 1, 0]]
    # We offset x by -slant*h/2 to keep the text roughly centered
    offset_x = -slant_angle * h / 2.0
    shear_matrix = np.array(
        [[1.0, slant_angle, offset_x],
         [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    sheared = cv2.warpAffine(
        image,
        shear_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return sheared


# =============================================================================
# Pixel Mask Generation (Binarization + AND)
# =============================================================================

@dataclass
class CharPlacement:
    """Describes the placement of a single character on the canvas."""
    char: str
    class_index: int
    x: int
    y: int
    font: ImageFont.FreeTypeFont


def _render_single_char_mask(
    char: str,
    font: ImageFont.FreeTypeFont,
    canvas_size: Tuple[int, int],
    x: int,
    y: int,
    binarization_threshold: int = 128,
) -> np.ndarray:
    """Render a single character and produce its binary pixel mask.

    This implements the professor's requirement: render the character,
    binarize it, and use the binarized pixels as the mask. This gives
    us pixel-accurate masks that follow the actual glyph shape rather
    than filling a rectangular bounding box.

    Args:
        char: The character to render.
        font: PIL font to use for rendering.
        canvas_size: (width, height) of the full canvas.
        x: Horizontal position of the character.
        y: Vertical position (baseline-relative).
        binarization_threshold: Threshold for converting the rendered
            grayscale glyph to a binary mask. Pixels darker than this
            value are considered foreground.

    Returns:
        Binary mask of shape (H, W) where foreground pixels are 1
        and background pixels are 0.
    """
    w, h = canvas_size

    # Render the character on a white canvas
    char_canvas = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(char_canvas)
    draw.text((x, y), char, font=font, fill=0)

    # Convert to numpy and binarize
    char_array = np.array(char_canvas, dtype=np.uint8)

    # Binarize: pixels where ink was drawn become 1 (foreground)
    # The character is drawn as dark (0) on white (255), so foreground
    # is where pixel value < threshold
    binary_mask = (char_array < binarization_threshold).astype(np.uint8)

    return binary_mask


# =============================================================================
# Main Generator
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for the synthetic data generator."""
    image_height: int = 512
    image_width: int = 512
    min_font_size: int = 28
    max_font_size: int = 48
    lines_per_image_min: int = 3
    lines_per_image_max: int = 7
    background_color: int = 255
    text_color_min: int = 0
    text_color_max: int = 60
    slant_range_min: float = -0.4
    slant_range_max: float = 0.4
    slant_probability: float = 0.5
    binarization_threshold: int = 128
    line_spacing_factor: float = 1.4
    margin_left: int = 15
    margin_right: int = 15
    margin_top: int = 15


class SyntheticGenerator:
    """Generates synthetic handwritten text images with pixel-level masks.

    The mask generation follows the binarization + AND approach:
    for each character, we render it in isolation, binarize the result
    to get the actual pixel footprint, and assign those pixels the
    character's class index in the segmentation mask.
    """

    def __init__(
        self,
        fonts_dir: str,
        output_dir: str,
        config: Optional[GeneratorConfig] = None,
        text_source: str = "wikipedia",
        seed: int = 42,
    ):
        self._config = config or GeneratorConfig()
        self._output_dir = Path(output_dir)
        self._font_manager = FontManager(fonts_dir)
        self._text_sampler = TextSampler(source=text_source)
        self._seed = seed

        random.seed(seed)
        np.random.seed(seed)

    def generate_dataset(
        self,
        num_train: int,
        num_val: int,
        num_test: int,
    ) -> Dict[str, int]:
        """Generate the complete synthetic dataset.

        Creates train/val/test splits with images and corresponding
        segmentation masks.

        Args:
            num_train: Number of training images.
            num_val: Number of validation images.
            num_test: Number of test images.

        Returns:
            Dictionary with counts of successfully generated images per split.
        """
        splits = {
            "train": num_train,
            "val": num_val,
            "test": num_test,
        }

        counts = {}
        for split_name, num_images in splits.items():
            counts[split_name] = self._generate_split(split_name, num_images)

        logger.info("Dataset generation complete: %s", counts)
        return counts

    def _generate_split(self, split_name: str, num_images: int) -> int:
        """Generate all images for a single split.

        Args:
            split_name: One of 'train', 'val', 'test'.
            num_images: Number of images to generate.

        Returns:
            Count of successfully generated images.
        """
        images_dir = self._output_dir / split_name / "images"
        masks_dir = self._output_dir / split_name / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        apply_slant = split_name == "train"  # slant only during training

        for idx in tqdm(range(num_images), desc=f"Generating {split_name}"):
            try:
                image, mask = self._generate_single_image(
                    apply_slant=apply_slant,
                )

                # Save image as grayscale PNG
                image_path = images_dir / f"{idx:06d}.png"
                cv2.imwrite(str(image_path), image)

                # Save mask as 8-bit PNG (class indices as pixel values)
                mask_path = masks_dir / f"{idx:06d}.png"
                cv2.imwrite(str(mask_path), mask)

                success_count += 1

            except Exception as exc:
                logger.warning(
                    "Failed to generate %s image %d: %s", split_name, idx, exc
                )
                continue

        logger.info(
            "Split '%s': generated %d/%d images.",
            split_name, success_count, num_images,
        )
        return success_count

    def _generate_single_image(
        self,
        apply_slant: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single image and its pixel-level segmentation mask.

        Pipeline:
            1. Sample text and font
            2. Compute character placements (positions on canvas)
            3. Render the full text image
            4. For each character, render in isolation, binarize, and
               write its class index into the mask at foreground pixels
            5. Optionally apply slant transform to both image and mask

        Args:
            apply_slant: Whether to apply random slant augmentation.

        Returns:
            Tuple of (image, mask) as numpy arrays.
            - image: shape (H, W), dtype uint8, grayscale
            - mask:  shape (H, W), dtype uint8, class indices
        """
        cfg = self._config
        canvas_w = cfg.image_width
        canvas_h = cfg.image_height

        # Sample rendering parameters
        font_size = random.randint(cfg.min_font_size, cfg.max_font_size)
        font = self._font_manager.load_font(font_size)
        text_color = random.randint(cfg.text_color_min, cfg.text_color_max)
        num_lines = random.randint(cfg.lines_per_image_min, cfg.lines_per_image_max)

        # Estimate line height from font metrics
        test_bbox = font.getbbox("Ag")
        line_height = int((test_bbox[3] - test_bbox[1]) * cfg.line_spacing_factor)

        # Sample text and break into lines that fit the canvas width
        text = self._text_sampler.sample(
            min_chars=num_lines * 20,
            max_chars=num_lines * 80,
        )
        lines = self._wrap_text(text, font, canvas_w - cfg.margin_left - cfg.margin_right)
        lines = lines[:num_lines]  # limit to requested number of lines

        # Compute character placements
        placements = self._compute_placements(
            lines, font, cfg.margin_left, cfg.margin_top, line_height
        )

        # Render the full text image
        image = self._render_text_image(
            placements, font, canvas_w, canvas_h, cfg.background_color, text_color
        )

        # Generate pixel-level mask via binarization + AND
        mask = self._generate_pixel_mask(
            placements, font, canvas_w, canvas_h, cfg.binarization_threshold
        )

        # Apply slant transform
        if apply_slant and random.random() < cfg.slant_probability:
            slant_angle = random.uniform(cfg.slant_range_min, cfg.slant_range_max)
            image = apply_slant_transform(
                image, slant_angle, border_value=cfg.background_color
            )
            mask = apply_slant_transform(
                mask, slant_angle, border_value=BACKGROUND_INDEX
            )

        return image, mask

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> List[str]:
        """Wrap text into lines that fit within max_width pixels.

        Args:
            text: Input text string.
            font: Font used for rendering.
            max_width: Maximum line width in pixels.

        Returns:
            List of text lines.
        """
        words = text.split(" ")
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def _compute_placements(
        self,
        lines: List[str],
        font: ImageFont.FreeTypeFont,
        margin_left: int,
        margin_top: int,
        line_height: int,
    ) -> List[CharPlacement]:
        """Compute the (x, y) position for each character.

        We render character by character to get exact positions. This
        is slower than rendering the whole string at once, but gives
        us precise per-character coordinates needed for mask generation.

        Args:
            lines: List of text lines.
            font: Font used for rendering.
            margin_left: Left margin in pixels.
            margin_top: Top margin in pixels.
            line_height: Vertical distance between lines.

        Returns:
            List of CharPlacement objects.
        """
        placements = []
        cursor_y = margin_top

        for line in lines:
            cursor_x = margin_left

            for char in line:
                class_idx = char_to_index(char)

                # Skip background-mapped characters (not in vocabulary)
                if class_idx == BACKGROUND_INDEX and char != " ":
                    continue

                # Get character width for advancing the cursor
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]

                # For space characters, just advance the cursor
                if char == " ":
                    space_bbox = font.getbbox(" ")
                    cursor_x += (space_bbox[2] - space_bbox[0])
                    continue

                placements.append(CharPlacement(
                    char=char,
                    class_index=class_idx,
                    x=cursor_x,
                    y=cursor_y,
                    font=font,
                ))

                # Advance cursor by character width plus a small gap
                cursor_x += char_width

            cursor_y += line_height

        return placements

    def _render_text_image(
        self,
        placements: List[CharPlacement],
        font: ImageFont.FreeTypeFont,
        width: int,
        height: int,
        background_color: int,
        text_color: int,
    ) -> np.ndarray:
        """Render the full text image from character placements.

        Args:
            placements: List of character placements.
            font: Font used for rendering.
            width: Canvas width.
            height: Canvas height.
            background_color: Background pixel value.
            text_color: Text ink pixel value.

        Returns:
            Grayscale image as numpy array of shape (H, W), dtype uint8.
        """
        canvas = Image.new("L", (width, height), color=background_color)
        draw = ImageDraw.Draw(canvas)

        for p in placements:
            draw.text((p.x, p.y), p.char, font=p.font, fill=text_color)

        return np.array(canvas, dtype=np.uint8)

    def _generate_pixel_mask(
        self,
        placements: List[CharPlacement],
        font: ImageFont.FreeTypeFont,
        width: int,
        height: int,
        binarization_threshold: int,
    ) -> np.ndarray:
        """Generate pixel-level segmentation mask using binarization + AND.

        For each character:
            1. Render it alone on a blank canvas
            2. Binarize to get its pixel footprint
            3. Write the character's class index at those pixel locations

        Later characters overwrite earlier ones at overlapping pixels.
        This naturally handles the (rare) case of overlapping glyphs
        by assigning the topmost character's label.

        Args:
            placements: List of character placements.
            font: Font used for rendering.
            width: Canvas width.
            height: Canvas height.
            binarization_threshold: Grayscale threshold for binarization.

        Returns:
            Segmentation mask of shape (H, W), dtype uint8.
            Pixel values are class indices (0 = background).
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        canvas_size = (width, height)

        for p in placements:
            # Render this character in isolation and binarize
            char_binary = _render_single_char_mask(
                char=p.char,
                font=p.font,
                canvas_size=canvas_size,
                x=p.x,
                y=p.y,
                binarization_threshold=binarization_threshold,
            )

            # AND operation: assign class index only where the character
            # has actual ink pixels (binary mask == 1)
            foreground = char_binary == 1
            mask[foreground] = p.class_index

        return mask


# =============================================================================
# Entry point helper
# =============================================================================

def create_generator_from_config(cfg) -> SyntheticGenerator:
    """Factory function to create a SyntheticGenerator from an OmegaConf config.

    Args:
        cfg: OmegaConf configuration object (the full config).

    Returns:
        Configured SyntheticGenerator instance.
    """
    gen_config = GeneratorConfig(
        image_height=cfg.data.synthetic.image_height,
        image_width=cfg.data.synthetic.image_width,
        min_font_size=cfg.data.synthetic.min_font_size,
        max_font_size=cfg.data.synthetic.max_font_size,
        lines_per_image_min=cfg.data.synthetic.lines_per_image_min,
        lines_per_image_max=cfg.data.synthetic.lines_per_image_max,
        background_color=cfg.data.synthetic.background_color,
        text_color_min=cfg.data.synthetic.text_color_min,
        text_color_max=cfg.data.synthetic.text_color_max,
        slant_range_min=cfg.data.augmentation.slant_range_min,
        slant_range_max=cfg.data.augmentation.slant_range_max,
        slant_probability=cfg.data.augmentation.p_slant,
    )

    return SyntheticGenerator(
        fonts_dir=cfg.data.synthetic.fonts_dir,
        output_dir=cfg.data.paths.synthetic_root,
        config=gen_config,
        text_source=cfg.data.synthetic.text_source,
        seed=cfg.project.seed,
    )