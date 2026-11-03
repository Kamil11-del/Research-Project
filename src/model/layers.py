"""
layers.py — Mamba-based encoder and decoder layer building blocks.

Replaces ActFusion's attn.py (MixedConvAttModule / MixedConvAttModuleV2).

Encoder side : MultiScaleMambaLayer
                 dilated conv  +  BidirectionalMamba  +  FFN

Decoder side : MultiScaleCrossMambaLayer
                 dilated conv  +  BidirectionalCrossMamba  +  FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Install with: pip install mamba-ssm causal-conv1d>=1.1.0")

from .cross_mamba import CrossMamba, BidirectionalCrossMamba


# ─────────────────────────────────────────────────────────────────────────────
# Encoder: BidirectionalMamba
# ─────────────────────────────────────────────────────────────────────────────

class BidirectionalMamba(nn.Module):
    """
    Runs Mamba forward and backward, then fuses.

    Args:
        d_model      : Feature dimension.
        d_state      : SSM state dimension.
        d_conv       : Conv kernel size inside Mamba.
        expand       : Expansion factor.
        fusion_type  : "gate", "concat", or "sum".
        dropout      : Dropout rate.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, fusion_type="gate", dropout=0.1):
        super().__init__()
        self.d_model     = d_model
        self.fusion_type = fusion_type

        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        if fusion_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
        elif fusion_type == "concat":
            self.fusion = nn.Linear(d_model * 2, d_model)
        elif fusion_type == "sum":
            pass
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose 'gate', 'concat', or 'sum'.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, L, D]  (sequence-first)
        Returns:
            output : [B, L, D]
        """
        out_fwd = self.mamba_fwd(x)

        x_bwd   = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])

        if self.fusion_type == "gate":
            combined = torch.cat([out_fwd, out_bwd], dim=-1)
            gate     = self.gate(combined)
            out      = gate * out_fwd + (1 - gate) * out_bwd
        elif self.fusion_type == "concat":
            out = self.fusion(torch.cat([out_fwd, out_bwd], dim=-1))
        else:  # sum
            out = (out_fwd + out_bwd) * 0.5

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Encoder Layer: MultiScaleMambaLayer
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleMambaLayer(nn.Module):
    """
    Single encoder layer.

    Local branch  : dilated Conv1d  (captures short-range structure)
    Global branch : BidirectionalMamba  (captures long-range dependencies)
    FFN           : expansion Conv1d block

    Inputs/outputs are channel-first: [B, D, T].
    """

    def __init__(
        self,
        d_model,
        kernel_size=3,
        dilation=1,
        d_state=32,
        d_conv=4,
        expand=2,
        dropout=0.1,
        ffn_ratio=1.5,
        bidirectional=True,
        fusion_type="gate",
    ):
        super().__init__()
        self.d_model = d_model

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = (kernel_size // 2) * dilation
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation),
        )

        if bidirectional:
            self.mamba = BidirectionalMamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, fusion_type=fusion_type, dropout=dropout,
            )
        else:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        hidden_dim = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.InstanceNorm1d(d_model, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, D, T]  (channel-first)
        Returns:
            output : [B, D, T]
        """
        x_drop = self.dropout(x)
        out1   = self.conv_block(x_drop)                              # [B, D, T]
        out2   = self.mamba(x_drop.transpose(1, 2)).transpose(1, 2)  # [B, D, T]

        out = self.norm(out1 + out2)
        out = self.ffn(out)
        return x + out


# ─────────────────────────────────────────────────────────────────────────────
# Decoder Layer: MultiScaleCrossMambaLayer
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleCrossMambaLayer(nn.Module):
    """
    Single decoder layer for diffusion-based action segmentation.

    Local branch  : dilated Conv1d
    Cross branch  : BidirectionalCrossMamba (conditions on encoder features)
    FFN           : expansion Conv1d block

    Inputs/outputs are channel-first: [B, D, T].
    The cross-attention context is also channel-first: [B, D_ctx, T].
    """

    def __init__(
        self,
        d_model: int,
        d_context: int,
        kernel_size: int,
        dilation: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_expand: float = 1.5,
        dropout: float = 0.1,
        bidirectional: bool = True,
        fusion_type: str = "gate",
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = (kernel_size // 2) * dilation
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation),
        )

        if bidirectional:
            self.cross_mamba = BidirectionalCrossMamba(
                d_model=d_model, d_context=d_context,
                d_state=d_state, d_conv=d_conv, expand=expand,
                fusion_type=fusion_type, dropout=dropout, **factory_kwargs,
            )
        else:
            self.cross_mamba = CrossMamba(
                d_model=d_model, d_context=d_context,
                d_state=d_state, d_conv=d_conv, expand=expand, **factory_kwargs,
            )

        ffn_hidden = int(d_model * ffn_expand)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, ffn_hidden, 1),
            nn.GELU(),
            nn.Conv1d(ffn_hidden, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.InstanceNorm1d(d_model, track_running_stats=False)

    def forward(self, x: torch.Tensor, x_cross: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : [B, D,     T]  decoder embeddings (channel-first)
            x_cross : [B, D_ctx, T]  encoder features   (channel-first)
        Returns:
            output  : [B, D, T]
        """
        x_drop       = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)  # [B, D, T]

        # CrossMamba expects sequence-first [B, T, D]
        x_mamba       = x_drop.transpose(1, 2)        # [B, T, D]
        x_cross_mamba = x_cross_drop.transpose(1, 2)  # [B, T, D_ctx]
        out2 = self.cross_mamba(x_mamba, x_cross_mamba).transpose(1, 2)  # [B, D, T]

        out = self.ffn(self.norm(out1 + out2))
        return x + out
