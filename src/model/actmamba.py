"""
actmamba.py — ActMamba V3

Key changes from V2:
  1. Masking reverted to PRE-ENCODER (same as V1 — experimentally confirmed best).
  2. Dual learnable mask tokens:
       - ant_mask_token   : used exclusively when cond_type == 'ant'
       - other_mask_token : used for all other masking strategies
                           (boundary, segment, random_patch, full, zero, …)
     Both tokens have shape [1, input_dim, 1] and are xavier-initialized.
     Rationale: anticipation masks out a contiguous future region; other masks
     remove scattered/boundary frames.  Separate tokens let the encoder learn
     distinct "I don't know the future" vs "I don't know this frame" signals.

All diffusion logic (DDIM, q_sample) is unchanged from ActFusion / V1 / V2.
"""

import math
import torch
import random
import numpy as np
import time as Time
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from .backbone import EncoderModel, DecoderModel


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_timestep_embedding(timesteps, embedding_dim):
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


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def normalize(x, scale):
    return (x * 2 - 1.) * scale


def denormalize(x, scale):
    return ((x / scale) + 1) / 2


# ─────────────────────────────────────────────────────────────────────────────
# ActMamba V3 Model
# ─────────────────────────────────────────────────────────────────────────────

class ActMamba(nn.Module):
    """
    ActMamba V3 — Pre-encoder masking with dual learnable mask tokens.

    Mask token design
    -----------------
    ant_mask_token   [1, input_dim, 1]  — anticipation conditioning
                     Replaces the future (right) portion of the sequence.
    other_mask_token [1, input_dim, 1]  — all other masking strategies
                     Replaces scattered, boundary or segment-based positions.

    Selection logic inside mask():
        cond_type == 'ant'  →  ant_mask_token
        everything else     →  other_mask_token

    Masking position: BEFORE encoder (pre-encoder), same as V1.
    """

    def __init__(self, encoder_params, decoder_params, diffusion_params,
                 num_classes, device, args=None):
        super(ActMamba, self).__init__()

        self.device      = device
        self.num_classes = num_classes

        # ── Diffusion Schedule ────────────────────────────────────────
        timesteps           = diffusion_params['timesteps']
        betas               = cosine_beta_schedule(timesteps)
        alphas              = 1. - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps,          = betas.shape
        self.num_timesteps  = int(timesteps)

        self.sampling_timesteps = diffusion_params['sampling_timesteps']
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta  = diffusion_params['ddim_sampling_eta']
        self.scale              = diffusion_params['snr_scale']

        self.register_buffer('betas',                         betas)
        self.register_buffer('alphas_cumprod',                alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',           alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',           torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',  torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',     torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',   torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance',            posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # ── Model Config ──────────────────────────────────────────────
        self.detach_decoder = diffusion_params['detach_decoder']
        self.cond_types      = diffusion_params['cond_types']

        self.use_instance_norm = encoder_params['use_instance_norm']
        if self.use_instance_norm:
            self.ins_norm = nn.InstanceNorm1d(encoder_params['input_dim'],
                                              track_running_stats=False)

        # Save input_dim before mutating encoder_params
        input_dim_for_mask = encoder_params['input_dim']

        # Compute decoder input_dim from feature_layer_indices
        decoder_params['input_dim'] = len(
            [i for i in encoder_params['feature_layer_indices'] if i not in [-1, -2]]
        ) * encoder_params['num_f_maps']
        if -1 in encoder_params['feature_layer_indices']:
            decoder_params['input_dim'] += encoder_params['input_dim']
        if -2 in encoder_params['feature_layer_indices']:
            decoder_params['input_dim'] += self.num_classes

        decoder_params['num_classes'] = num_classes
        encoder_params['num_classes'] = num_classes
        encoder_params_clean = {k: v for k, v in encoder_params.items()
                                if k != 'use_instance_norm'}
        encoder_params_clean['args'] = args
        decoder_params['args'] = args

        self.encoder = EncoderModel(**encoder_params_clean)
        self.decoder = DecoderModel(**decoder_params)

        # ── Dual Learnable Mask Tokens ────────────────────────────────
        # ant_mask_token   : anticipation — masks contiguous future region
        # other_mask_token : boundary/segment/random — masks scattered frames
        # Both live in input feature space [1, input_dim, 1].
        self.ant_mask_token   = nn.Parameter(torch.zeros(1, input_dim_for_mask, 1))
        self.other_mask_token = nn.Parameter(torch.zeros(1, input_dim_for_mask, 1))
        nn.init.xavier_uniform_(self.ant_mask_token)
        nn.init.xavier_uniform_(self.other_mask_token)

        self.args = args

    # ── Diffusion Helpers ─────────────────────────────────────────────

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def model_predictions(self, backbone_feats, x, t):
        x_m    = torch.clamp(x, min=-self.scale, max=self.scale)
        x_m    = denormalize(x_m, self.scale)
        assert x_m.max() <= 1 and x_m.min() >= 0
        x_start = self.decoder(backbone_feats, t, x_m.float())
        x_start = F.softmax(x_start, 1)
        assert x_start.max() <= 1 and x_start.min() >= 0
        x_start    = normalize(x_start, self.scale)
        x_start    = torch.clamp(x_start, min=-self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def prepare_targets(self, event_gt):
        assert event_gt.max() <= 1 and event_gt.min() >= 0
        t     = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(size=event_gt.shape, device=self.device)
        x_start        = (event_gt * 2. - 1.) * self.scale
        x              = self.q_sample(x_start=x_start, t=t, noise=noise)
        x              = torch.clamp(x, min=-self.scale, max=self.scale)
        event_diffused = ((x / self.scale) + 1) / 2.
        return event_diffused, noise, t

    # ── Masking / Conditioning ────────────────────────────────────────

    def mask(self, x, event_gt=None, boundary_gt=None, cond_type='full', args=None):
        """
        Apply a conditioning mask to input features x [B, D, T].

        Token selection (V3 dual-token design):
            cond_type == 'ant'  →  self.ant_mask_token
            anything else       →  self.other_mask_token

        Supported cond_types:
            'zero', 'full', 'boundary05-', 'boundary03-',
            'segment=1', 'segment=2', 'ant',
            'random_num_patch', 'random_patch'
        """
        # ── Select the appropriate learnable mask token ───────────────
        tok = self.ant_mask_token if cond_type == 'ant' else self.other_mask_token

        if cond_type == 'zero':
            mask_tokens  = tok.repeat(x.shape[0], 1, x.shape[2]).to(self.device)
            feature_mask = torch.zeros((x.shape[0], 1, x.shape[2])).to(self.device)
            return mask_tokens, feature_mask

        elif cond_type == 'full':
            feature_mask = torch.ones((x.shape[0], 1, x.shape[2]))

        elif cond_type == 'boundary05-':
            feature_mask = (boundary_gt < 0.5).float()

        elif cond_type == 'boundary03-':
            feature_mask = (boundary_gt < 0.3).float()

        elif cond_type == 'segment=1':
            event_gt     = torch.argmax(event_gt, dim=1, keepdim=True).long()
            events       = torch.unique(event_gt)
            random_event = np.random.choice(events.cpu().numpy())
            feature_mask = (event_gt != random_event).float()

        elif cond_type == 'segment=2':
            event_gt     = torch.argmax(event_gt, dim=1, keepdim=True).long()
            events       = torch.unique(event_gt)
            e1           = np.random.choice(events.cpu().numpy())
            e2           = np.random.choice(events.cpu().numpy())
            feature_mask = (event_gt != e1).float() * (event_gt != e2).float()

        elif cond_type == 'ant':
            obs_ps       = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            obs_p        = random.choice(obs_ps)
            obs_len      = int(x.shape[2] * obs_p)
            feature_mask = torch.ones((x.shape[0], 1, x.shape[2]))
            feature_mask[:, :, obs_len:] = 0   # future is masked

        elif cond_type == 'random_num_patch':
            patch_size     = getattr(args, 'patch_size', 10)
            n_mask_default = getattr(args, 'n_mask', 10)
            max_idx        = int(event_gt.shape[2] / patch_size)
            n_mask         = min(random.randint(5, n_mask_default),
                                 max(max_idx, max_idx - n_mask_default))
            start_idx      = random.sample(range(0, max_idx), n_mask)
            feature_mask   = torch.ones((x.shape[0], 1, x.shape[2]))
            for idx in start_idx:
                feature_mask[:, :, idx * patch_size: idx * patch_size + patch_size] = 0

        elif cond_type == 'random_patch':
            patch_size   = getattr(args, 'patch_size', 10)
            max_idx      = int(event_gt.shape[2] / patch_size)
            n_mask       = min(getattr(args, 'n_mask', 10), max(max_idx, max_idx - 10))
            start_idx    = random.sample(range(0, max_idx), n_mask)
            feature_mask = torch.ones((x.shape[0], 1, x.shape[2]))
            for idx in start_idx:
                feature_mask[:, :, idx * patch_size: idx * patch_size + patch_size] = 0

        else:
            raise ValueError(f"Unknown cond_type: '{cond_type}'")

        feature_mask = feature_mask.to(self.device)
        sorted_idx   = torch.argsort(feature_mask, dim=-1, descending=True)

        # reshape(-1) is safe for any batch size / temporal length
        selected_idx = (feature_mask.reshape(-1) == 1).nonzero(as_tuple=False).reshape(-1)
        x            = torch.index_select(x, 2, selected_idx)

        n_masked     = sorted_idx.shape[2] - x.shape[2]
        mask_tokens  = tok.repeat(x.shape[0], 1, n_masked).to(self.device)
        restored_idx = torch.argsort(sorted_idx, dim=2)
        x_           = torch.cat([x, mask_tokens], dim=2)
        x            = torch.gather(x_, dim=2,
                                    index=restored_idx.repeat(1, x.shape[1], 1))
        return x, feature_mask

    # ── Training ──────────────────────────────────────────────────────

    def get_training_loss(
        self, video_feats, event_gt, boundary_gt,
        encoder_ce_criterion, encoder_mse_criterion, encoder_boundary_criterion,
        decoder_ce_criterion, decoder_mse_criterion, decoder_boundary_criterion,
        soft_label, args=None,
    ):
        if self.use_instance_norm:
            video_feats = self.ins_norm(video_feats)

        input_feats = video_feats

        # ── V3 Pre-Encoder Masking ────────────────────────────────────────────
        # Mask is applied to raw input features BEFORE the encoder.
        # ant_mask_token / other_mask_token selected automatically inside mask().
        cond_type   = random.choice(self.cond_types)
        input_feats, feature_mask = self.mask(
            input_feats, event_gt, boundary_gt, cond_type, args
        )
        encoder_out, backbone_feats = self.encoder(
            input_feats, get_features=True, mask=feature_mask
        )

        # ── Encoder Loss ──────────────────────────────────────────────
        if soft_label is None:
            encoder_ce_loss = encoder_ce_criterion(
                encoder_out.transpose(2, 1).contiguous().view(-1, self.num_classes),
                torch.argmax(event_gt, dim=1).view(-1).long(),
            )
        else:
            soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
            for i in range(soft_event_gt.shape[1]):
                soft_event_gt[0, i] = gaussian_filter1d(soft_event_gt[0, i], soft_label)
            soft_event_gt   = torch.from_numpy(soft_event_gt).to(self.device)
            encoder_ce_loss = -(soft_event_gt * F.log_softmax(encoder_out, 1)).sum(0).sum(0)

        encoder_mse_loss = torch.clamp(
            encoder_mse_criterion(
                F.log_softmax(encoder_out[:, :, 1:],  dim=1),
                F.log_softmax(encoder_out.detach()[:, :, :-1], dim=1),
            ), min=0, max=16,
        ).mean()

        encoder_boundary_loss = torch.tensor(0).to(self.device)
        encoder_ce_loss       = encoder_ce_loss.mean()

        # ── Diffusion Targets ─────────────────────────────────────────
        event_diffused, noise, t = self.prepare_targets(event_gt)
        event_out = self.decoder(backbone_feats, t, event_diffused.float())

        # ── Decoder Loss ──────────────────────────────────────────────
        decoder_boundary = 1 - torch.einsum(
            'bicl,bcjl->bijl',
            F.softmax(event_out[:, None, :, 1:],          2),
            F.softmax(event_out[:, :, None, :-1].detach(), 1),
        ).squeeze(1)

        if soft_label is None:
            decoder_ce_loss = decoder_ce_criterion(
                event_out.transpose(2, 1).contiguous().view(-1, self.num_classes),
                torch.argmax(event_gt, dim=1).view(-1).long(),
            )
        else:
            soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
            for i in range(soft_event_gt.shape[1]):
                soft_event_gt[0, i] = gaussian_filter1d(soft_event_gt[0, i], soft_label)
            soft_event_gt   = torch.from_numpy(soft_event_gt).to(self.device)
            decoder_ce_loss = -(soft_event_gt * F.log_softmax(event_out, 1)).sum(0).sum(0)

        decoder_mse_loss = torch.clamp(
            decoder_mse_criterion(
                F.log_softmax(event_out[:, :, 1:],  dim=1),
                F.log_softmax(event_out.detach()[:, :, :-1], dim=1),
            ), min=0, max=16,
        ).mean()

        decoder_boundary_loss = decoder_boundary_criterion(
            decoder_boundary, boundary_gt[:, :, 1:]
        ).mean()
        decoder_ce_loss = decoder_ce_loss.mean()

        return {
            'encoder_ce_loss':       encoder_ce_loss,
            'encoder_mse_loss':      encoder_mse_loss,
            'encoder_boundary_loss': encoder_boundary_loss,
            'decoder_ce_loss':       decoder_ce_loss,
            'decoder_mse_loss':      decoder_mse_loss,
            'decoder_boundary_loss': decoder_boundary_loss,
        }

    # ── DDIM Sampling (Inference) ─────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self, video_feats, seed=None, full_len=None, args=None):
        """
        DDIM reverse-diffusion sampling.

        TAS  : full_len=None  — processes the complete video
        LTA  : full_len=T     — video_feats covers observed portion only;
               the future is padded with ant_mask_token (consistent with training).
        """
        if self.use_instance_norm:
            video_feats = self.ins_norm(video_feats)

        if full_len is not None:
            # ── LTA: pad future with ant_mask_token ──────────────────
            mask_len    = full_len - video_feats.size(2)
            mask_tokens = self.ant_mask_token.repeat(1, 1, mask_len).to(self.device)
            video_feats = torch.cat((video_feats, mask_tokens), dim=-1)

            mask              = torch.ones((1, 1, full_len))
            vis_len           = full_len - mask_len
            mask[:, :, vis_len:] = 0
        else:
            # ── TAS: no padding needed ───────────────────────────────
            mask = torch.ones((1, 1, video_feats.size(2)))

        encoder_out, backbone_feats = self.encoder(
            video_feats, get_features=True, mask=mask.to(self.device)
        )

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if full_len is not None:
            shape          = (video_feats.shape[0], self.num_classes, full_len)
            backbone_feats = backbone_feats[:, :, :video_feats.shape[-1]]
            obs_len        = backbone_feats.size(-1)
            new_feats      = torch.zeros(
                (1, backbone_feats.size(1), full_len), device=self.device
            )
            new_feats[:, :, :obs_len] = backbone_feats
            input_feats    = new_feats
        else:
            shape       = (video_feats.shape[0], self.num_classes, video_feats.shape[2])
            input_feats = backbone_feats

        total_timesteps    = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta                = self.ddim_sampling_eta

        times      = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times      = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_time  = torch.randn(shape, device=self.device)
        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((1,), time, device=self.device, dtype=torch.long)

            if full_len is not None:
                pred_noise, x_start = self.model_predictions(input_feats,    x_time, time_cond)
            else:
                pred_noise, x_start = self.model_predictions(backbone_feats, x_time, time_cond)

            x_return = torch.clone(x_start)

            if time_next < 0:
                x_time = x_start
                continue

            alpha      = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma      = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c          = (1 - alpha_next - sigma ** 2).sqrt()
            noise      = torch.randn_like(x_time)
            x_time     = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        x_return = denormalize(x_return, self.scale)

        if seed is not None:
            t = int(1000 * Time.time()) % 2**16
            random.seed(t)
            torch.manual_seed(t)
            torch.cuda.manual_seed_all(t)

        return x_return
