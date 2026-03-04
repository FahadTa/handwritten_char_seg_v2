# =============================================================================
# swin_unet.py
# =============================================================================
# Swin Transformer U-Net (SwinUNet) for semantic segmentation.
#
# Architecture overview:
#   Encoder: Patch embedding + 4 stages of Swin Transformer blocks
#            with patch merging between stages for downsampling.
#   Bottleneck: Swin Transformer blocks at the deepest resolution.
#   Decoder: 4 stages of Swin Transformer blocks with patch expanding
#            (upsampling) and skip connections from the encoder.
#   Output: Linear projection to num_classes.
#
# Key concepts:
#   - Window-based self-attention: Attention is computed within local
#     windows of fixed size, making complexity linear in image size
#     rather than quadratic.
#   - Shifted windows: Alternating blocks shift the window partition
#     to allow cross-window information flow.
#   - Patch merging/expanding: Analogous to pooling/upsampling in CNNs
#     but operating on token sequences.
#
# Reference:
#   Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical
#   Image Segmentation", ECCV 2022 Workshop.
#
# Design decisions:
#   - Relative position bias instead of absolute position embedding
#     for better generalization to different image sizes.
#   - Pre-norm (LayerNorm before attention/MLP) for training stability.
#   - Stochastic depth (drop path) for regularization.
#   - Same output interface as AttentionUNet: (B, num_classes, H, W).
# =============================================================================

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size: Size of each square window.

    Returns:
        Windows tensor of shape (num_windows * B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Reverse window partition back to feature map.

    Args:
        windows: Windows tensor of shape (num_windows * B, window_size, window_size, C).
        window_size: Size of each square window.
        H: Original feature map height.
        W: Original feature map width.

    Returns:
        Feature map tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =============================================================================
# Drop Path (Stochastic Depth)
# =============================================================================

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.

    During training, randomly drops entire residual branches with
    probability drop_prob. During evaluation, acts as identity.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


# =============================================================================
# MLP Block
# =============================================================================

class MLP(nn.Module):
    """Two-layer MLP with GELU activation used in each Transformer block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# Window Attention
# =============================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    Attention is computed within local windows rather than globally,
    reducing complexity from O(n^2) to O(n * window_size^2).

    Args:
        dim: Number of input channels.
        window_size: Size of the attention window.
        num_heads: Number of attention heads.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        # (2*Wh-1) * (2*Ww-1) entries, one per relative position pair
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index for each token pair in window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens of shape (num_windows*B, N, C) where N = window_size^2.
            mask: Attention mask for shifted windows, shape (num_windows, N, N) or None.

        Returns:
            Output tokens of shape (num_windows*B, N, C).
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Swin Transformer Block
# =============================================================================

class SwinTransformerBlock(nn.Module):
    """A single Swin Transformer block with optional window shift.

    Consists of:
        1. Layer norm -> Window attention -> Residual
        2. Layer norm -> MLP -> Residual

    When shift_size > 0, the feature map is cyclically shifted before
    window partition and shifted back after, enabling cross-window
    information flow.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

        # Attention mask will be computed in forward based on input size
        self.attn_mask = None
        self._last_H = 0
        self._last_W = 0

    def _compute_attention_mask(
        self, H: int, W: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted window attention."""
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, H*W, C).
            H: Feature map height.
            W: Feature map width.

        Returns:
            Output tensor of shape (B, H*W, C).
        """
        B, L, C = x.shape
        assert L == H * W, f"Input length {L} != H*W ({H}*{W}={H * W})"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature map to be divisible by window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Compute attention mask if needed
        if H != self._last_H or W != self._last_W:
            self.attn_mask = self._compute_attention_mask(Hp, Wp, x.device)
            self._last_H = H
            self._last_W = W

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Reverse window partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # Residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# =============================================================================
# Patch Operations
# =============================================================================

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and embed them.

    Converts (B, C, H, W) image to (B, H/patch*W/patch, embed_dim) tokens.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 96,
        patch_size: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input image (B, C, H, W).

        Returns:
            Tuple of (tokens, H_patches, W_patches) where tokens has
            shape (B, H_patches * W_patches, embed_dim).
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, embed_dim)
        x = self.norm(x)
        return x, Hp, Wp


class PatchMerging(nn.Module):
    """Merge 2x2 neighboring patches to downsample by 2x.

    Analogous to strided convolution or pooling in CNNs.
    Reduces spatial resolution by 2x and doubles channel count.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input tokens (B, H*W, C).
            H, W: Spatial dimensions.

        Returns:
            Tuple of (merged tokens, H//2, W//2).
        """
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)

        # Pad if not even
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w

        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4C)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2C)

        return x, H // 2, W // 2


class PatchExpanding(nn.Module):
    """Expand patches to upsample by 2x.

    Analogous to transposed convolution or upsampling in CNNs.
    Doubles spatial resolution and halves channel count.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input tokens (B, H*W, C).
            H, W: Spatial dimensions.

        Returns:
            Tuple of (expanded tokens, H*2, W*2).
        """
        B, L, C = x.shape
        assert L == H * W

        x = self.expand(x)  # (B, H*W, 2C)
        x = x.view(B, H, W, 2 * C)

        # Rearrange to double spatial dimensions
        # Split channels into 4 groups, place in 2x2 spatial pattern
        C_new = C // 2
        x = x.view(B, H, W, 2, 2, C_new)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C_new)

        x = self.norm(x)
        x = x.view(B, -1, C_new)

        return x, H * 2, W * 2


class FinalPatchExpanding(nn.Module):
    """Final patch expansion to recover original image resolution.

    Upsamples by patch_size factor (typically 4x) to go from the
    first decoder stage back to pixel-level resolution.
    """

    def __init__(self, dim: int, patch_size: int = 4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * patch_size * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input tokens (B, H*W, C).

        Returns:
            Tuple of (expanded tokens, H*patch_size, W*patch_size).
        """
        B, L, C = x.shape
        x = self.expand(x)  # (B, H*W, patch^2 * C)
        x = x.view(B, H, W, self.patch_size, self.patch_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * self.patch_size, W * self.patch_size, C)
        x = self.norm(x)
        x = x.view(B, -1, C)

        return x, H * self.patch_size, W * self.patch_size


# =============================================================================
# Swin Transformer Stage
# =============================================================================

class SwinTransformerStage(nn.Module):
    """A stage consisting of multiple Swin Transformer blocks.

    Even-indexed blocks use regular windows, odd-indexed blocks use
    shifted windows. This alternation enables cross-window connections.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path if isinstance(drop_path, float)
                    else drop_path[i],
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, H, W)
        return x


# =============================================================================
# Full SwinUNet Architecture
# =============================================================================

class SwinUNet(nn.Module):
    """Swin Transformer U-Net for semantic segmentation.

    Encoder-decoder architecture where both encoder and decoder use
    Swin Transformer blocks instead of convolutions. Skip connections
    are added between corresponding encoder and decoder stages via
    linear projection and concatenation.

    Args:
        in_channels: Number of input channels (1 for grayscale).
        num_classes: Number of output segmentation classes.
        embed_dim: Base embedding dimension (doubled at each stage).
        depths: Number of Swin blocks at each encoder stage.
        num_heads: Number of attention heads at each stage.
        window_size: Window size for local attention.
        patch_size: Initial patch embedding size.
        mlp_ratio: MLP hidden dim ratio.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 80,
        embed_dim: int = 96,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        window_size: int = 8,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        depth_offset = 0
        for i in range(self.num_stages):
            stage_dim = embed_dim * (2 ** i)
            stage = SwinTransformerStage(
                dim=stage_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth_offset:depth_offset + depths[i]],
            )
            self.encoder_stages.append(stage)
            depth_offset += depths[i]

            # Add downsampling between stages (not after last)
            if i < self.num_stages - 1:
                self.downsample_layers.append(PatchMerging(stage_dim))

        # Bottleneck norm
        bottleneck_dim = embed_dim * (2 ** (self.num_stages - 1))
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)

        # Decoder stages
        self.decoder_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        for i in range(self.num_stages - 1):
            # Decoder works from deepest to shallowest
            dec_idx = self.num_stages - 2 - i
            dec_dim = embed_dim * (2 ** dec_idx)
            up_dim = embed_dim * (2 ** (dec_idx + 1))

            # Upsample
            self.upsample_layers.append(PatchExpanding(up_dim))

            # Skip connection projection (concat then project)
            self.skip_projections.append(
                nn.Linear(dec_dim + dec_dim, dec_dim, bias=False)
            )

            # Decoder Swin stage
            stage = SwinTransformerStage(
                dim=dec_dim,
                depth=depths[dec_idx],
                num_heads=num_heads[dec_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.0,  # No stochastic depth in decoder
            )
            self.decoder_stages.append(stage)

        # Final upsampling to pixel resolution
        self.final_expand = FinalPatchExpanding(embed_dim, patch_size)
        self.output_proj = nn.Linear(embed_dim, num_classes, bias=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using truncated normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        B, C, H_orig, W_orig = x.shape

        # Patch embedding
        x, H, W = self.patch_embed(x)

        # Encoder (save skip connections)
        skips = []
        for i in range(self.num_stages):
            x = self.encoder_stages[i](x, H, W)
            if i < self.num_stages - 1:
                skips.append((x, H, W))
                x, H, W = self.downsample_layers[i](x, H, W)

        # Bottleneck
        x = self.bottleneck_norm(x)

        # Decoder
        for i in range(self.num_stages - 1):
            # Upsample
            x, H, W = self.upsample_layers[i](x, H, W)

            # Skip connection (from corresponding encoder stage)
            skip_x, skip_H, skip_W = skips[self.num_stages - 2 - i]

            # Align spatial dimensions if needed
            if x.shape[1] != skip_x.shape[1]:
                min_len = min(x.shape[1], skip_x.shape[1])
                x = x[:, :min_len, :]
                skip_x = skip_x[:, :min_len, :]
                H = min(H, skip_H)
                W = min(W, skip_W)

            # Concatenate and project
            x = torch.cat([x, skip_x], dim=-1)
            x = self.skip_projections[i](x)

            # Decoder stage
            x = self.decoder_stages[i](x, H, W)

        # Final expansion to pixel resolution
        x, H, W = self.final_expand(x, H, W)

        # Output projection
        x = self.output_proj(x)  # (B, H*W, num_classes)
        x = x.view(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, num_classes, H, W)

        # Ensure output matches input spatial dimensions
        if H != H_orig or W != W_orig:
            x = F.interpolate(
                x, size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )

        return x

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_summary(self) -> str:
        """Return a human-readable summary of parameter counts."""
        components = {
            "Patch Embed": self.patch_embed,
            "Encoder": self.encoder_stages,
            "Downsample": self.downsample_layers,
            "Bottleneck Norm": self.bottleneck_norm,
            "Upsample": self.upsample_layers,
            "Skip Proj": self.skip_projections,
            "Decoder": self.decoder_stages,
            "Final Expand": self.final_expand,
            "Output Proj": self.output_proj,
        }

        lines = ["SwinUNet Parameter Summary", "=" * 45]
        total = 0
        for name, module in components.items():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total += count
            lines.append(f"  {name:<15s}: {count:>12,d}")
        lines.append("-" * 45)
        lines.append(f"  {'Total':<15s}: {total:>12,d}")
        return "\n".join(lines)


# =============================================================================
# Factory Functions
# =============================================================================

def build_swin_unet(
    in_channels: int = 1,
    num_classes: int = 80,
    embed_dim: int = 96,
    depths: Optional[List[int]] = None,
    num_heads: Optional[List[int]] = None,
    window_size: int = 8,
    patch_size: int = 4,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
) -> SwinUNet:
    """Factory function to build a SwinUNet."""
    model = SwinUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        patch_size=patch_size,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )

    logger.info(
        "Built SwinUNet: %d parameters",
        model.get_num_parameters(),
    )

    return model


def build_swin_unet_from_config(cfg) -> SwinUNet:
    """Build SwinUNet from OmegaConf config."""
    swin_cfg = cfg.model.swin_unet
    return build_swin_unet(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=swin_cfg.embed_dim,
        depths=list(swin_cfg.depths),
        num_heads=list(swin_cfg.num_heads),
        window_size=swin_cfg.window_size,
        patch_size=swin_cfg.patch_size,
        mlp_ratio=swin_cfg.mlp_ratio,
        drop_rate=swin_cfg.drop_rate,
        attn_drop_rate=swin_cfg.attn_drop_rate,
        drop_path_rate=swin_cfg.drop_path_rate,
    )