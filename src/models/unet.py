# =============================================================================
# unet.py
# =============================================================================
# Attention U-Net for semantic segmentation of handwritten characters.
#
# Architecture overview:
#   Encoder: 5 blocks of [Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU]
#            with MaxPool2d between blocks for downsampling.
#   Bottleneck: Same as encoder block but at the deepest resolution.
#   Decoder: 4 blocks of [Upsample -> AttentionGate -> Concat -> Conv -> Conv]
#   Output:  1x1 convolution to num_classes channels.
#
# Attention gates compute a gating signal from the decoder features and
# apply it to the encoder skip connection. This suppresses irrelevant
# spatial regions (background) and amplifies relevant regions (character
# boundaries), which is particularly important for our task where the
# majority of pixels are background.
#
# Design decisions:
#   - Bilinear upsampling + Conv instead of transposed convolution to
#     avoid checkerboard artifacts.
#   - Dropout after each decoder block for regularization.
#   - He (Kaiming) initialization for all conv layers.
#   - Channel dimensions: [64, 128, 256, 512, 1024] (configurable).
# =============================================================================

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Double convolution block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU.

    This is the fundamental building block used in both encoder and decoder.
    Each convolution preserves spatial dimensions via padding=1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        layers = []

        # First convolution
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm)
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Second convolution
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm)
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock followed by MaxPool for downsampling.

    Returns both the convolution output (for skip connection) and the
    downsampled output (for the next encoder stage).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, use_batch_norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip: Feature map before pooling (for skip connection).
            pooled: Downsampled feature map (for next encoder stage).
        """
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class AttentionGate(nn.Module):
    """Attention gate for skip connections.

    Computes an attention map from the gating signal (decoder features)
    and the skip connection (encoder features). The attention map is
    element-wise multiplied with the skip features to suppress irrelevant
    regions.

    The mechanism:
        1. Project both gating and skip features to an intermediate space
        2. Sum them and apply ReLU
        3. Project to a single-channel attention map via 1x1 conv + sigmoid
        4. Multiply the skip features by the attention map

    Reference:
        Oktay et al., "Attention U-Net: Learning Where to Look for the
        Pancreas", MIDL 2018.
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2
            if inter_channels == 0:
                inter_channels = 1

        # Project gating signal to intermediate space
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Project skip connection to intermediate space
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        gate: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention gating to skip connection features.

        Args:
            gate: Gating signal from decoder, shape (B, gate_ch, H_g, W_g).
            skip: Skip connection from encoder, shape (B, skip_ch, H_s, W_s).

        Returns:
            Attended skip features, same shape as skip input.
        """
        # Project both to intermediate space
        g = self.w_gate(gate)
        s = self.w_skip(skip)

        # Align spatial dimensions (gate may be smaller than skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(
                g, size=s.shape[2:], mode="bilinear", align_corners=False
            )

        # Compute attention coefficients
        attention = self.relu(g + s)
        attention = self.psi(attention)

        # Apply attention to skip features
        return skip * attention


class DecoderBlock(nn.Module):
    """Decoder block: Upsample -> AttentionGate -> Concat -> ConvBlock.

    Upsamples the input features, applies attention gating to the
    corresponding encoder skip connection, concatenates them, and
    processes through a ConvBlock.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # Bilinear upsample + 1x1 conv to halve channels
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Attention gate
        self.attention = AttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
        )

        # ConvBlock after concatenation
        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            use_batch_norm,
        )

        # Optional dropout
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input from previous decoder stage.
            skip: Skip connection from corresponding encoder stage.

        Returns:
            Decoded feature map.
        """
        # Upsample
        x = self.upsample(x)

        # Pad if needed to match skip spatial dims
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = F.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2,
                 diff_h // 2, diff_h - diff_h // 2],
            )

        # Apply attention to skip connection
        skip_attended = self.attention(gate=x, skip=skip)

        # Concatenate and process
        x = torch.cat([x, skip_attended], dim=1)
        x = self.conv(x)
        x = self.dropout(x)

        return x


# =============================================================================
# Full Architecture
# =============================================================================

class AttentionUNet(nn.Module):
    """Attention U-Net for semantic segmentation.

    A U-Net with attention gates in the skip connections. The encoder
    extracts hierarchical features through progressive downsampling,
    the bottleneck captures the deepest semantic representation, and
    the decoder reconstructs spatial resolution while using attention
    to focus on relevant encoder features.

    Args:
        in_channels: Number of input channels (1 for grayscale).
        num_classes: Number of output segmentation classes.
        encoder_channels: List of channel counts for encoder stages.
            Default: [64, 128, 256, 512, 1024] where the last value
            is the bottleneck.
        dropout_rate: Dropout probability in decoder blocks.
        use_batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 80,
        encoder_channels: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 1024]

        assert len(encoder_channels) == 5, (
            f"Expected 5 encoder channel values, got {len(encoder_channels)}"
        )

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder path
        self.enc1 = EncoderBlock(in_channels, encoder_channels[0], use_batch_norm)
        self.enc2 = EncoderBlock(encoder_channels[0], encoder_channels[1], use_batch_norm)
        self.enc3 = EncoderBlock(encoder_channels[1], encoder_channels[2], use_batch_norm)
        self.enc4 = EncoderBlock(encoder_channels[2], encoder_channels[3], use_batch_norm)

        # Bottleneck
        self.bottleneck = ConvBlock(encoder_channels[3], encoder_channels[4], use_batch_norm)

        # Decoder path
        self.dec4 = DecoderBlock(
            encoder_channels[4], encoder_channels[3], encoder_channels[3],
            use_batch_norm, dropout_rate,
        )
        self.dec3 = DecoderBlock(
            encoder_channels[3], encoder_channels[2], encoder_channels[2],
            use_batch_norm, dropout_rate,
        )
        self.dec2 = DecoderBlock(
            encoder_channels[2], encoder_channels[1], encoder_channels[1],
            use_batch_norm, dropout_rate,
        )
        self.dec1 = DecoderBlock(
            encoder_channels[1], encoder_channels[0], encoder_channels[0],
            use_batch_norm, dropout_rate,
        )

        # Output head
        self.output_conv = nn.Conv2d(
            encoder_channels[0], num_classes, kernel_size=1
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming (He) initialization to all convolutional layers.

        He initialization is preferred for ReLU-activated networks because
        it accounts for the variance reduction caused by ReLU.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
            Apply softmax/argmax externally for probabilities/predictions.
        """
        # Encoder
        skip1, x = self.enc1(x)   # skip1: (B, 64,  H,    W)
        skip2, x = self.enc2(x)   # skip2: (B, 128, H/2,  W/2)
        skip3, x = self.enc3(x)   # skip3: (B, 256, H/4,  W/4)
        skip4, x = self.enc4(x)   # skip4: (B, 512, H/8,  W/8)

        # Bottleneck
        x = self.bottleneck(x)     # x:     (B, 1024, H/16, W/16)

        # Decoder
        x = self.dec4(x, skip4)    # (B, 512, H/8,  W/8)
        x = self.dec3(x, skip3)    # (B, 256, H/4,  W/4)
        x = self.dec2(x, skip2)    # (B, 128, H/2,  W/2)
        x = self.dec1(x, skip1)    # (B, 64,  H,    W)

        # Output
        logits = self.output_conv(x)  # (B, num_classes, H, W)

        return logits

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_summary(self) -> str:
        """Return a human-readable summary of parameter counts per component."""
        components = {
            "Encoder 1": self.enc1,
            "Encoder 2": self.enc2,
            "Encoder 3": self.enc3,
            "Encoder 4": self.enc4,
            "Bottleneck": self.bottleneck,
            "Decoder 4": self.dec4,
            "Decoder 3": self.dec3,
            "Decoder 2": self.dec2,
            "Decoder 1": self.dec1,
            "Output": self.output_conv,
        }

        lines = ["Attention U-Net Parameter Summary", "=" * 45]
        total = 0
        for name, module in components.items():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total += count
            lines.append(f"  {name:<15s}: {count:>12,d}")
        lines.append("-" * 45)
        lines.append(f"  {'Total':<15s}: {total:>12,d}")
        return "\n".join(lines)


def build_attention_unet(
    in_channels: int = 1,
    num_classes: int = 80,
    encoder_channels: Optional[List[int]] = None,
    dropout_rate: float = 0.1,
    use_batch_norm: bool = True,
) -> AttentionUNet:
    """Factory function to build an Attention U-Net.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        encoder_channels: Encoder channel dimensions.
        dropout_rate: Dropout probability.
        use_batch_norm: Whether to use batch normalization.

    Returns:
        Configured AttentionUNet instance.
    """
    model = AttentionUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_channels=encoder_channels,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )

    logger.info(
        "Built Attention U-Net: %d parameters",
        model.get_num_parameters(),
    )

    return model


def build_attention_unet_from_config(cfg) -> AttentionUNet:
    """Build Attention U-Net from OmegaConf config.

    Args:
        cfg: OmegaConf configuration object.

    Returns:
        Configured AttentionUNet instance.
    """
    unet_cfg = cfg.model.attention_unet
    return build_attention_unet(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        encoder_channels=list(unet_cfg.encoder_channels),
        dropout_rate=unet_cfg.dropout_rate,
        use_batch_norm=unet_cfg.use_batch_norm,
    )