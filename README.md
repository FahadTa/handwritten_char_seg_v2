# Handwritten Character Segmentation v2

Pixel-level semantic segmentation of handwritten characters using deep learning, with evaluation on real handwriting data from the IAM Handwriting Database.

**Author:** Fahad Tariq (Matriculation: 23186464)  
**Supervisor:** Dr.-Ing. Vincent Christlein  
**Institution:** Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg

## Overview

This project extends the [v1 handwritten character segmentation system](https://github.com/FahadTa/handwritten_char_seg_v2) with three key improvements:

1. **Pixel-level masks via binarization + AND** — Masks follow actual glyph shapes instead of rectangular bounding boxes
2. **Slant augmentation and extended augmentations** — Render-time cursive simulation plus affine, elastic, morphological, and photometric transforms
3. **Real data evaluation on IAM** — Domain gap analysis comparing synthetic vs. real handwriting performance

Two architectures are trained and compared: **Attention U-Net** and **SwinUNet** (Swin Transformer U-Net).

## Results

### Synthetic Test Set

| Metric | Attention U-Net | SwinUNet |
|--------|----------------|----------|
| Best Val IoU | **98.08%** | 95.10% |
| Pixel Accuracy | 98.59% | **99.03%** |
| Macro IoU | 61.20% | **68.20%** |
| Macro Dice | 75.07% | **80.32%** |
| Precision | 61.30% | **68.61%** |
| Recall | **99.63%** | 98.91% |
| Parameters | **29.3M** | 41.4M |

### IAM Handwriting Database (Real Data)

| Metric | Attention U-Net | SwinUNet |
|--------|----------------|----------|
| Pixel Accuracy | **87.01%** | 84.48% |
| Macro IoU | 0.16% | **0.26%** |
| Macro Dice | 0.32% | **0.52%** |

The large domain gap between synthetic and real data confirms that synthetic-to-real transfer remains the primary challenge for handwritten character segmentation.

## Project Structure

```
handwritten_char_seg_v2/
├── configs/
│   └── config.yaml                 # Central configuration
├── scripts/
│   ├── generate_dataset.py         # Synthetic dataset generation
│   ├── train.py                    # Model training
│   ├── evaluate.py                 # Synthetic test evaluation
│   └── evaluate_iam.py             # IAM evaluation + domain gap
├── src/
│   ├── data/
│   │   ├── charset.py              # Character set definition (80 classes)
│   │   ├── synthetic_generator.py  # Synthetic data with pixel masks
│   │   ├── augmentations.py        # Augmentation pipeline
│   │   ├── dataset.py              # PyTorch Dataset + DataModule
│   │   └── iam_adapter.py          # IAM database adapter
│   ├── models/
│   │   ├── unet.py                 # Attention U-Net
│   │   ├── swin_unet.py            # SwinUNet
│   │   └── loss.py                 # Combined Dice + CE loss
│   ├── training/
│   │   ├── lightning_module.py     # PyTorch Lightning module
│   │   └── callbacks.py            # Training callbacks
│   └── evaluation/
│       ├── metrics.py              # Segmentation metrics
│       ├── domain_gap.py           # Domain gap analysis
│       └── visualize.py            # Visualization utilities
├── fonts/                          # Handwritten .ttf fonts
├── data/                           # Generated datasets (gitignored)
├── outputs/                        # Checkpoints and results (gitignored)
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.11
- PyTorch 2.1.0 with CUDA 11.8
- 4 NVIDIA GPUs (for distributed training)

### Installation

```bash
# Clone the repository
git clone https://github.com/FahadTa/handwritten_char_seg_v2.git
cd handwritten_char_seg_v2

# Create conda environment
conda create -n charseg python=3.11
conda activate charseg

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Fonts

Place handwritten-style `.ttf` font files in the `fonts/` directory. These can be downloaded from Google Fonts (filter by "Handwriting" category).

### IAM Database (for real data evaluation)

1. Register at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
2. Download: `lines.tgz`, `xml.tgz`, `ascii.tgz`
3. Download split files from "Large Writer Independent Text Line Recognition Task"
4. Extract into `data/iam/`:

```
data/iam/
├── lines/          # Line-level images
├── xml/            # XML annotations
├── ascii/          # Text transcriptions (lines.txt, words.txt)
└── split/          # trainset.txt, validationset1.txt, validationset2.txt, testset.txt
```

## Usage

### 1. Generate Synthetic Dataset

```bash
# Full dataset (12,000 train / 1,500 val / 1,500 test)
python scripts/generate_dataset.py

# Quick test (50 images)
python scripts/generate_dataset.py --num-train 50 --num-val 10 --num-test 10
```

### 2. Train Models

```bash
# Sanity check (1 batch, 1 GPU)
python scripts/train.py --fast-dev-run --devices 1 --wandb-offline

# Train Attention U-Net (4 GPUs)
python scripts/train.py --config configs/config.yaml

# Train SwinUNet (4 GPUs)
python scripts/train.py --architecture swin_unet --config configs/config.yaml

# Resume interrupted training
python scripts/train.py --resume outputs/checkpoints/last.ckpt
```

### 3. Evaluate on Synthetic Test Set

```bash
# Attention U-Net
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints_unet/best.ckpt \
    --output-dir outputs/evaluation/unet_synthetic

# SwinUNet
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints_swin/best.ckpt \
    --architecture swin_unet \
    --output-dir outputs/evaluation/swin_synthetic
```

### 4. Evaluate on IAM (Real Data)

```bash
# Attention U-Net with domain gap analysis
python scripts/evaluate_iam.py \
    --checkpoint outputs/checkpoints_unet/best.ckpt \
    --synthetic-results outputs/evaluation/unet_synthetic/metrics.json \
    --output-dir outputs/evaluation/unet_iam \
    --split test

# SwinUNet with domain gap analysis
python scripts/evaluate_iam.py \
    --checkpoint outputs/checkpoints_swin/best.ckpt \
    --architecture swin_unet \
    --synthetic-results outputs/evaluation/swin_synthetic/metrics.json \
    --output-dir outputs/evaluation/swin_iam \
    --split test
```

## Methodology

### Synthetic Data Generation

Text is sampled from Wikipedia and rendered onto 512x512 canvases using handwritten-style fonts. Pixel-level segmentation masks are generated by rendering each character in isolation, binarizing the result, and assigning the character's class index only at foreground (ink) pixels. This binarization + AND approach produces masks that follow actual glyph contours.

### Augmentation Pipeline

**Render-time:** Horizontal shear [-0.4, 0.4] simulating cursive slant, applied identically to image and mask.

**Training-time (via Albumentations):**
- Geometric: affine (rotation, shear, scale, translation), perspective, elastic deformation
- Morphological: erosion (thin strokes), dilation (thick strokes) simulating pen pressure
- Photometric: brightness, contrast, CLAHE
- Noise: Gaussian noise, Gaussian blur

### Architectures

**Attention U-Net:** Encoder-decoder CNN with attention gates in skip connections. 5 encoder blocks [64, 128, 256, 512, 1024 channels], 4 decoder blocks with bilinear upsampling. Attention gates suppress irrelevant background features. 29.3M parameters.

**SwinUNet:** Encoder-decoder Transformer with Swin Transformer blocks using window-based self-attention and shifted windows. Patch embedding (4x4), 4 encoder stages [96, 192, 384, 768 dims], patch merging/expanding for resolution changes. 41.4M parameters.

### Training

- Loss: Combined Dice + Cross-Entropy (0.5 each) with sqrt-dampened class weights
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Scheduler: Cosine annealing (100 epochs, min_lr=1e-6)
- Distributed: 4 GPU DDP, batch size 8/GPU, gradient accumulation 2x (effective batch 64)
- Precision: 16-bit mixed precision
- Early stopping: patience 25, monitoring val/iou

### IAM Evaluation

Pseudo ground-truth masks for IAM images are generated by Otsu binarization followed by connected component analysis. Components are sorted in reading order and mapped to transcription characters sequentially. These masks are approximate and intended for domain gap assessment.

## Experiment Tracking

All experiments are tracked with Weights & Biases:
- Entity: `fahadtariqadmission-student`
- Project: `handwritten-char-seg-v2`

## Dependencies

Core: PyTorch 2.1, PyTorch Lightning 2.1.4, torchmetrics 1.3.2  
Augmentation: Albumentations 1.4.4, OpenCV 4.9  
Tracking: Weights & Biases 0.16.6  
Config: OmegaConf 2.3.0  

See `requirements.txt` for complete list.

## References

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018
3. Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation", ECCV 2022
4. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
5. Marti & Bunke, "The IAM-database: An English Sentence Database for Offline Handwriting Recognition", IJDAR 2002