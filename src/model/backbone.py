"""
backbone.py — Mamba-based Encoder and Decoder models.

Replaces ActFusion's backbone.py (which used MixedConvAttModule / MixedConvAttModuleV2).
The public API is kept identical to ActFusion:
    EncoderModel(...) — drop-in replacement for the original EncoderModel
    DecoderModel(...) — drop-in replacement for the original DecoderModel

Internal architecture:
    Encoder : MultiScaleMambaModule  (stacked MultiScaleMambaLayer with 2^i dilation)
    Decoder : MultiScaleCrossMambaModule (stacked MultiScaleCrossMambaLayer with 2^i dilation,
               conditioned on encoder backbone features via CrossMamba)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MultiScaleMambaLayer, MultiScaleCrossMambaLayer


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal timestep embedding (same as in DDPM / ActFusion).

    Args:
        timesteps     : [B]  — integer diffusion timesteps
        embedding_dim : int  — output dimension
    Returns:
        emb : [B, embedding_dim]
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def swish(x):
    return x * torch.sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# Internal Modules
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleMambaModule(nn.Module):
    """
    Stacked MultiScaleMambaLayer with exponentially increasing dilation (2^i).
    Used inside EncoderModel.

    Args:
        num_layers    : Number of layers.
        d_model       : Feature dimension.
        kernel_size   : Conv kernel size.
        d_state       : Mamba SSM state dimension.
        d_conv        : Mamba depthwise conv kernel size.
        expand        : Mamba expansion factor.
        dropout       : Dropout rate.
        bidirectional : Use bidirectional Mamba.
        fusion_type   : Fusion type for bidirectional Mamba ("gate", "concat", "sum").
    """

    def __init__(
        self, num_layers, d_model, kernel_size,
        d_state=64, d_conv=4, expand=2, dropout=0.1,
        bidirectional=True, fusion_type="gate",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiScaleMambaLayer(
                d_model=d_model,
                kernel_size=kernel_size,
                dilation=2 ** i,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ffn_ratio=1.5,
                dropout=dropout,
                bidirectional=bidirectional,
                fusion_type=fusion_type,
            )
            for i in range(num_layers)
        ])

    def forward(self, x, feature_layer_indices=None):
        """
        Args:
            x                    : [B, D, T]  channel-first
            feature_layer_indices: list of layer indices to collect (None = no collection)
        Returns:
            If feature_layer_indices is None : x [B, D, T]
            Else                             : (x, features) where features = cat of selected layers
        """
        if feature_layer_indices is None:
            for layer in self.layers:
                x = layer(x)
            return x

        features = []
        for l_id, layer in enumerate(self.layers):
            x = layer(x)
            if l_id in feature_layer_indices:
                features.append(x)

        features = torch.cat(features, dim=1) if features else None
        return x, features


class MultiScaleCrossMambaModule(nn.Module):
    """
    Stacked MultiScaleCrossMambaLayer with exponentially increasing dilation (2^i).
    Used inside DecoderModel.

    Args:
        num_layers    : Number of layers.
        num_f_maps    : Decoder feature dimension.
        input_dim_cross: Encoder feature dimension (for cross-conditioning).
        kernel_size   : Conv kernel size.
        dropout_rate  : Dropout rate.
        time_emb_dim  : Timestep embedding dimension (for diffusion conditioning).
        d_state       : Mamba SSM state dimension.
        d_conv        : Mamba depthwise conv kernel size.
        expand        : Mamba expansion factor.
        bidirectional : Use bidirectional CrossMamba.
        fusion_type   : Fusion type for bidirectional CrossMamba.
    """

    def __init__(
        self, num_layers, num_f_maps, input_dim_cross, kernel_size,
        dropout_rate, time_emb_dim=None,
        d_state=16, d_conv=4, expand=2,
        bidirectional=True, fusion_type="gate",
    ):
        super().__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                MultiScaleCrossMambaLayer(
                    d_model=num_f_maps,
                    d_context=input_dim_cross,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    ffn_expand=1.5,
                    dropout=dropout_rate,
                    bidirectional=bidirectional,
                    fusion_type=fusion_type,
                )
            )
            for i in range(num_layers)
        ])

    def forward(self, x, x_cross, time_emb=None):
        """
        Args:
            x        : [B, D,     T]  decoder features (channel-first)
            x_cross  : [B, D_ctx, T]  encoder features (channel-first)
            time_emb : [B, time_emb_dim]  diffusion timestep embedding
        Returns:
            x : [B, D, T]
        """
        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:, :, None]
        for layer in self.layers:
            x = layer(x, x_cross)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Public API — drop-in replacements for ActFusion's EncoderModel / DecoderModel
# ─────────────────────────────────────────────────────────────────────────────

class EncoderModel(nn.Module):
    """
    Mamba-based encoder.  Drop-in replacement for ActFusion's EncoderModel.

    Architecture:
        conv_in (1x1)  →  MultiScaleMambaModule  →  conv_out (1x1)

    Supports feature collection at specified layer indices (for decoder conditioning).

    Args:
        num_layers           : Number of Mamba encoder layers.
        num_f_maps           : Internal feature dimension.
        input_dim            : Input video feature dimension.
        num_classes          : Number of action classes.
        kernel_size          : Dilated conv kernel size inside each layer.
        normal_dropout_rate  : Dropout applied inside Mamba layers.
        channel_dropout_rate : Dropout2d applied to input channels.
        temporal_dropout_rate: Dropout2d applied to temporal dimension.
        feature_layer_indices: Which layer outputs to collect for the decoder.
        d_state              : Mamba SSM state dimension (new, default 16).
        d_conv               : Mamba depthwise conv kernel size (new, default 4).
        expand               : Mamba expansion factor (new, default 2).
        bidirectional        : Use bidirectional Mamba (new, default True).
        fusion_type          : Fusion strategy (new, default "gate").
        args                 : Unused; kept for API compatibility.
    """

    def __init__(
        self,
        num_layers,
        num_f_maps,
        input_dim,
        num_classes,
        kernel_size,
        normal_dropout_rate,
        channel_dropout_rate,
        temporal_dropout_rate,
        feature_layer_indices=None,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        fusion_type="gate",
        args=None,
    ):
        super().__init__()

        self.num_classes          = num_classes
        self.feature_layer_indices = feature_layer_indices

        self.dropout_channel  = nn.Dropout2d(p=channel_dropout_rate)
        self.dropout_temporal = nn.Dropout2d(p=temporal_dropout_rate)
        self.conv_in          = nn.Conv1d(input_dim, num_f_maps, 1)

        self.encoder = MultiScaleMambaModule(
            num_layers=num_layers,
            d_model=num_f_maps,
            kernel_size=kernel_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=normal_dropout_rate,
            bidirectional=bidirectional,
            fusion_type=fusion_type,
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, get_features=False, selected_idx=None, mask=None):
        """
        Args:
            x            : [B, input_dim, T]
            get_features : If True, also return concatenated backbone features.
            selected_idx : Unused (kept for API compat with ActFusion).
            mask         : Unused (kept for API compat with ActFusion).
        Returns:
            If get_features: (logits [B, C, T], features [B, F, T])
            Else           :  logits [B, C, T]
        """
        if get_features:
            assert self.feature_layer_indices is not None and len(self.feature_layer_indices) > 0

            features = []
            if -1 in self.feature_layer_indices:   # raw video feature
                features.append(x)

            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            x = self.conv_in(x)

            layer_indices = [i for i in self.feature_layer_indices if i >= 0]
            x, layer_features = self.encoder(x, feature_layer_indices=layer_indices)

            if layer_features is not None:
                features.append(layer_features)

            out = self.conv_out(x)

            if -2 in self.feature_layer_indices:   # encoder prediction
                features.append(F.softmax(out, dim=1))

            all_features = torch.cat(features, dim=1)
            return out, all_features

        else:
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            x = self.conv_in(x)
            x = self.encoder(x)
            return self.conv_out(x)


class DecoderModel(nn.Module):
    """
    Mamba-based decoder with diffusion timestep conditioning.
    Drop-in replacement for ActFusion's DecoderModel.

    Architecture:
        time_emb → conv_in (1x1) → MultiScaleCrossMambaModule → conv_out (1x1)

    Args:
        input_dim     : Encoder feature dimension (for cross-conditioning).
        num_classes   : Number of action classes.
        num_layers    : Number of CrossMamba decoder layers.
        num_f_maps    : Internal feature dimension.
        time_emb_dim  : Diffusion timestep embedding dimension.
        kernel_size   : Dilated conv kernel size inside each layer.
        dropout_rate  : Dropout rate.
        d_state       : Mamba SSM state dimension (new, default 16).
        d_conv        : Mamba depthwise conv kernel size (new, default 4).
        expand        : Mamba expansion factor (new, default 2).
        bidirectional : Use bidirectional CrossMamba (new, default True).
        fusion_type   : Fusion strategy (new, default "gate").
        args          : Unused; kept for API compatibility.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        num_layers,
        num_f_maps,
        time_emb_dim,
        kernel_size,
        dropout_rate,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
        fusion_type="gate",
        args=None,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_in = nn.ModuleList([
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
        ])

        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)

        self.module = MultiScaleCrossMambaModule(
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            input_dim_cross=input_dim,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            time_emb_dim=time_emb_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=bidirectional,
            fusion_type=fusion_type,
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, t, event):
        """
        Args:
            x     : [B, F, T]  encoder backbone features
            t     : [B]        diffusion timestep indices
            event : [B, C, T]  noisy action probability map  [0, 1]
        Returns:
            event_out : [B, C, T]  predicted clean action map
        """
        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_in[0](time_emb)
        time_emb = swish(time_emb)
        time_emb = self.time_in[1](time_emb)

        fra       = self.conv_in(event)
        fra       = self.module(fra, x, time_emb)
        event_out = self.conv_out(fra)
        return event_out
