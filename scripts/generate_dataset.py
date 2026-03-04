# =============================================================================
# generate_dataset.py
# =============================================================================
# CLI entry point for generating the synthetic handwritten character
# segmentation dataset.
#
# Usage:
#   python scripts/generate_dataset.py
#   python scripts/generate_dataset.py --config configs/config.yaml
#   python scripts/generate_dataset.py --num-train 500 --num-val 100 --num-test 100
#
# The script:
#   1. Loads configuration from YAML
#   2. Initializes the synthetic generator (fonts, text sampler)
#   3. Generates train/val/test splits with images and pixel-level masks
#   4. Prints dataset statistics on completion
# =============================================================================

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.data.charset import NUM_CLASSES, CHARSET
from src.data.synthetic_generator import SyntheticGenerator, GeneratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic handwritten character segmentation dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=None,
        help="Override number of training images.",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Override number of validation images.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=None,
        help="Override number of test images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--fonts-dir",
        type=str,
        default=None,
        help="Override fonts directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for dataset generation."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    cfg = OmegaConf.load(str(config_path))

    # Apply CLI overrides
    if args.num_train is not None:
        cfg.data.synthetic.num_train = args.num_train
    if args.num_val is not None:
        cfg.data.synthetic.num_val = args.num_val
    if args.num_test is not None:
        cfg.data.synthetic.num_test = args.num_test
    if args.output_dir is not None:
        cfg.data.paths.synthetic_root = args.output_dir
    if args.fonts_dir is not None:
        cfg.data.synthetic.fonts_dir = args.fonts_dir
    if args.seed is not None:
        cfg.project.seed = args.seed

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("Synthetic Dataset Generation")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  Fonts directory:  %s", cfg.data.synthetic.fonts_dir)
    logger.info("  Output directory: %s", cfg.data.paths.synthetic_root)
    logger.info("  Image size:       %dx%d",
                cfg.data.synthetic.image_height,
                cfg.data.synthetic.image_width)
    logger.info("  Font size range:  %d-%d",
                cfg.data.synthetic.min_font_size,
                cfg.data.synthetic.max_font_size)
    logger.info("  Lines per image:  %d-%d",
                cfg.data.synthetic.lines_per_image_min,
                cfg.data.synthetic.lines_per_image_max)
    logger.info("  Num classes:      %d", NUM_CLASSES)
    logger.info("  Charset size:     %d characters", len(CHARSET))
    logger.info("  Random seed:      %d", cfg.project.seed)
    logger.info("Split sizes:")
    logger.info("  Train: %d", cfg.data.synthetic.num_train)
    logger.info("  Val:   %d", cfg.data.synthetic.num_val)
    logger.info("  Test:  %d", cfg.data.synthetic.num_test)
    logger.info(
        "  Total: %d",
        cfg.data.synthetic.num_train
        + cfg.data.synthetic.num_val
        + cfg.data.synthetic.num_test,
    )
    logger.info("=" * 60)

    # Build generator configuration
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

    # Initialize generator
    generator = SyntheticGenerator(
        fonts_dir=cfg.data.synthetic.fonts_dir,
        output_dir=cfg.data.paths.synthetic_root,
        config=gen_config,
        text_source=cfg.data.synthetic.text_source,
        seed=cfg.project.seed,
    )

    # Generate dataset
    start_time = time.time()

    counts = generator.generate_dataset(
        num_train=cfg.data.synthetic.num_train,
        num_val=cfg.data.synthetic.num_val,
        num_test=cfg.data.synthetic.num_test,
    )

    elapsed = time.time() - start_time

    # Print results
    logger.info("=" * 60)
    logger.info("Generation Complete")
    logger.info("=" * 60)
    logger.info("Results:")
    for split_name, count in counts.items():
        expected = getattr(cfg.data.synthetic, f"num_{split_name}")
        status = "OK" if count == expected else "PARTIAL"
        logger.info("  %s: %d/%d [%s]", split_name, count, expected, status)
    logger.info("Total time: %.1f seconds (%.2f sec/image)",
                elapsed,
                elapsed / max(sum(counts.values()), 1))
    logger.info("Output: %s", cfg.data.paths.synthetic_root)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()