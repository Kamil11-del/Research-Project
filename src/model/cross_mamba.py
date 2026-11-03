# Copyright (c) 2024, Cross-Contextual Mamba for Diffusion-based Action Segmentation
# Based on official Mamba implementation by Tri Dao and Albert Gu

"""
CrossMamba: Cross-Contextual Mamba Layer

Modifies standard Mamba so that all SSM parameters (Δ, B, C) are conditioned
on both the main input x and a conditioning context z via gated fusion:
    Δ_t = Linear_x(x_t) + σ(Gate_Δ(z_t)) * Linear_z(z_t) + bias
    B_t = Linear_x(x_t) + σ(Gate_B(z_t)) * Linear_z(z_t)
    C_t = Linear_x(x_t) + σ(Gate_C(z_t)) * Linear_z(z_t)

O(T) linear complexity — equivalent to cross-attention-like grounding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    SELECTIVE_SCAN_AVAILABLE = True
except ImportError:
    selective_scan_fn = None
    SELECTIVE_SCAN_AVAILABLE = False
    print("Warning: mamba_ssm not installed. CrossMamba will not be available.")

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class CrossMamba(nn.Module):
    """
    Cross-Contextual Mamba Layer.

    All SSM parameters (Δ, B, C) are conditioned on both the main input x
    and a conditioning context z (e.g. encoder backbone features).

    Args:
        d_model        : Main input/output dimension.
        d_context      : Context (conditioning) input dimension.
        d_state        : SSM state dimension (default: 16).
        d_conv         : Depthwise conv kernel size (default: 4).
        expand         : Expansion factor for inner dimension (default: 2).
        dt_rank        : Rank of Δ projection ("auto" = ceil(d_model/16)).
        dt_min/dt_max  : Range for Δ initialization.
        dt_init        : "random" or "constant".
        dt_scale       : Scale for dt_proj initialization.
        dt_init_floor  : Floor for dt initialization.
        conv_bias      : Whether conv1d has bias.
        bias           : Whether linear layers have bias.
        project_context: Project context to d_inner before fusion (default: True).
    """

    def __init__(
        self,
        d_model: int,
        d_context: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        project_context: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if not SELECTIVE_SCAN_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for CrossMamba. "
                "Install with: pip install mamba-ssm causal-conv1d>=1.1.0"
            )

        self.d_model = d_model
        self.d_context = d_context
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.project_context = project_context

        # ── Input Projections ──────────────────────────────────────────
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if project_context:
            self.context_proj = nn.Linear(d_context, self.d_inner, bias=bias, **factory_kwargs)
        else:
            assert d_context == self.d_inner, (
                f"When project_context=False, d_context ({d_context}) must equal d_inner ({self.d_inner})"
            )
            self.context_proj = nn.Identity()

        # ── Depthwise Convolution ──────────────────────────────────────
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        # ── SSM Parameter Projections ──────────────────────────────────
        # From main input x → dt_rank + 2*d_state
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        # From context z → dt, B, C (cross-conditioning, the key modification)
        self.z_dt_proj = nn.Linear(self.d_inner, self.dt_rank, bias=False, **factory_kwargs)
        self.z_B_proj  = nn.Linear(self.d_inner, self.d_state,  bias=False, **factory_kwargs)
        self.z_C_proj  = nn.Linear(self.d_inner, self.d_state,  bias=False, **factory_kwargs)

        # Gate projections — sigmoid gates control context contribution
        self.z_dt_gate = nn.Linear(self.d_inner, self.dt_rank, bias=True, **factory_kwargs)
        self.z_B_gate  = nn.Linear(self.d_inner, self.d_state,  bias=True, **factory_kwargs)
        self.z_C_gate  = nn.Linear(self.d_inner, self.d_state,  bias=True, **factory_kwargs)

        # dt projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # ── Parameter Initialization ───────────────────────────────────
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"dt_init={dt_init} not supported")

        # dt bias: F.softplus(dt_bias) ∈ [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # Context projections — small init for stability
        nn.init.normal_(self.z_dt_proj.weight, std=0.02)
        nn.init.normal_(self.z_B_proj.weight,  std=0.02)
        nn.init.normal_(self.z_C_proj.weight,  std=0.02)

        # Gate biases = -2 → sigmoid ≈ 0 at start (minimal context influence)
        nn.init.normal_(self.z_dt_gate.weight, std=0.02)
        nn.init.normal_(self.z_B_gate.weight,  std=0.02)
        nn.init.normal_(self.z_C_gate.weight,  std=0.02)
        nn.init.constant_(self.z_dt_gate.bias, -2.0)
        nn.init.constant_(self.z_B_gate.bias,  -2.0)
        nn.init.constant_(self.z_C_gate.bias,  -2.0)

        # ── SSM Parameters ─────────────────────────────────────────────
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # ── Output Projection ──────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states : [B, L, D]      — main input (e.g. noisy action labels)
            context       : [B, L, D_ctx]  — conditioning context (encoder features)
        Returns:
            output        : [B, L, D]
        """
        batch, seqlen, _ = hidden_states.shape

        # Project context: [B, L, D_ctx] → [B, L, D_in]
        context_feat = self.context_proj(context)

        # Project main input efficiently: [B, L, D] → [B, D_in*2, L]
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)  # Each: [B, D_in, L]

        # Depthwise conv on x
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # Flatten for linear layers
        x_flat       = rearrange(x,            "b d l -> (b l) d")
        context_flat = rearrange(context_feat, "b l d -> (b l) d")

        # SSM parameters from main input
        x_dbl = self.x_proj(x_flat)
        dt_x, B_x, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # SSM parameter contributions from context
        dt_z = self.z_dt_proj(context_flat)
        B_z  = self.z_B_proj(context_flat)
        C_z  = self.z_C_proj(context_flat)

        # Gated cross-contextual fusion
        dt_gate = torch.sigmoid(self.z_dt_gate(context_flat))
        B_gate  = torch.sigmoid(self.z_B_gate(context_flat))
        C_gate  = torch.sigmoid(self.z_C_gate(context_flat))

        dt_combined = dt_x + dt_gate * dt_z
        B_combined  = B_x  + B_gate  * B_z
        C_combined  = C    + C_gate  * C_z

        # dt: [D_in, B*L] → [B, D_in, L]
        dt = self.dt_proj.weight @ dt_combined.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

        # Reshape B, C for selective_scan: [B, d_state, L]
        B_out = rearrange(B_combined, "(b l) n -> b n l", l=seqlen).contiguous()
        C_out = rearrange(C_combined, "(b l) n -> b n l", l=seqlen).contiguous()

        A = -torch.exp(self.A_log.float())

        y = selective_scan_fn(
            x,
            dt,
            A,
            B_out,
            C_out,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )  # [B, D_in, L]

        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class BidirectionalCrossMamba(nn.Module):
    """
    Bidirectional Cross-Mamba.

    Runs CrossMamba in forward + backward directions and fuses results.
    Suitable for offline video processing where the full sequence is available.

    Args:
        d_model      : Main input/output dimension.
        d_context    : Context input dimension.
        d_state      : SSM state dimension.
        d_conv       : Depthwise conv kernel size.
        expand       : Expansion factor.
        fusion_type  : "concat", "sum", or "gate".
        dropout      : Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_context: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fusion_type: str = "gate",
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.fusion_type = fusion_type

        self.cross_mamba_fwd = CrossMamba(
            d_model=d_model, d_context=d_context,
            d_state=d_state, d_conv=d_conv, expand=expand, **factory_kwargs,
        )
        self.cross_mamba_bwd = CrossMamba(
            d_model=d_model, d_context=d_context,
            d_state=d_state, d_conv=d_conv, expand=expand, **factory_kwargs,
        )

        if fusion_type == "concat":
            self.fusion = nn.Linear(d_model * 2, d_model, **factory_kwargs)
        elif fusion_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model, **factory_kwargs),
                nn.Sigmoid(),
            )
        elif fusion_type == "sum":
            pass
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose 'concat', 'gate', or 'sum'.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : [B, L, D]      — main input
            context : [B, L, D_ctx]  — context features
        Returns:
            output  : [B, L, D]
        """
        out_fwd = self.cross_mamba_fwd(x, context)

        x_bwd       = torch.flip(x,       dims=[1])
        context_bwd = torch.flip(context, dims=[1])
        out_bwd = self.cross_mamba_bwd(x_bwd, context_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])

        if self.fusion_type == "concat":
            out = self.fusion(torch.cat([out_fwd, out_bwd], dim=-1))
        elif self.fusion_type == "gate":
            combined = torch.cat([out_fwd, out_bwd], dim=-1)
            gate = self.gate(combined)
            out = gate * out_fwd + (1 - gate) * out_bwd
        else:  # sum
            out = (out_fwd + out_bwd) / 2

        return out
