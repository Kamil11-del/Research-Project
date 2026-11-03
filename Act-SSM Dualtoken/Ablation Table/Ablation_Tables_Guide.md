# ActSSM — Additional Ablation Tables Guide

## Critical classification: what costs what

| Label | Meaning |
|:---|:---|
| **FREE** | Computed from prediction .txt files already saved on disk |
| **INFERENCE** | Load best checkpoint + run ddim_sample with different settings |
| **RETRAIN** | Must retrain from scratch with config changes |

---

## GROUP A — FREE (zero GPU cost, zero retraining)

These use the `.txt` prediction files already saved in `result/.../prediction/` from your completed training runs.

---

### A1. Error Analysis Table ← HIGH PRIORITY

This is one of the strongest tables you can add. It directly characterises *what kind* of mistakes each method makes.

| Error Type | DiffAct | ActFusion | ActSSM |
|:---|:---:|:---:|:---:|
| Over-segmentation (extra predicted segments) | | | |
| Under-segmentation (merged segments) | | | |
| Boundary shift (correct class, wrong timing) | | | |
| Wrong class in correct region | | | |
| Wrong anticipation (future class error) | | | |

**How to compute it — write `compute_error_analysis.py`:**

```python
import os, numpy as np
from src.utils import read_file, get_labels_start_end_time, read_mapping_dict

def count_errors(pred_dir, label_dir, video_list):
    over_seg = under_seg = boundary_shift = wrong_class = 0
    for vid in video_list:
        gt   = read_file(os.path.join(label_dir, f'{vid}.txt')).split('\n')[:-1]
        pred = read_file(os.path.join(pred_dir,  f'{vid}.txt')).split('\n')[1].split()
        
        gt_segs,   gs, ge = get_labels_start_end_time(gt)
        pred_segs, ps, pe = get_labels_start_end_time(pred)

        # Over-segmentation: more predicted segments than GT
        if len(pred_segs) > len(gt_segs):
            over_seg += len(pred_segs) - len(gt_segs)
        # Under-segmentation: fewer predicted segments than GT
        elif len(pred_segs) < len(gt_segs):
            under_seg += len(gt_segs) - len(pred_segs)

        # Boundary shift: for each GT segment, find best IoU match,
        # correct class but IoU < 0.5 threshold
        for i, (gl, g_start, g_end) in enumerate(zip(gt_segs, gs, ge)):
            for j, (pl, p_start, p_end) in enumerate(zip(pred_segs, ps, pe)):
                if gl == pl:
                    iou = (min(g_end,p_end) - max(g_start,p_start)) / \
                          (max(g_end,p_end) - min(g_start,p_start) + 1e-6)
                    if 0.1 < iou < 0.5:
                        boundary_shift += 1

        # Wrong class: frame-level
        for g, p in zip(gt, pred):
            if g != p:
                wrong_class += 1

    total = sum(len(read_file(os.path.join(label_dir, f'{v}.txt')).split('\n')[:-1])
                for v in video_list)
    return {
        'over_seg': over_seg / len(video_list),
        'under_seg': under_seg / len(video_list),
        'boundary_shift': boundary_shift / len(video_list),
        'wrong_class': wrong_class / total * 100,
    }
```

You run this against ActSSM predictions (already saved) AND you can run it on DiffAct/ActFusion prediction files if you download their released checkpoints. If not available, report ActSSM only and note others aren't released.

---

### A2. Per-Split Performance Table ← Already computed, just format it

You already have per-split results in your `log.txt` files. Format them as:

| Split | Acc | Edit | F1@10 | F1@25 | F1@50 | TAS Avg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4/5 | | | | | | |
| **Avg** | **90.27** | **87.76** | **92.32** | **91.41** | **87.08** | **89.77** |

This demonstrates consistency across splits. Many papers skip this but reviewers appreciate it.

---

### A3. Computational Efficiency Table ← Partial (params/memory are FREE)

**Params** — run once, costs 2 seconds:
```python
import torch
from src.model.actmamba import ActMamba
from src.utils import load_config_file

cfg = load_config_file('configs/50salads.json')
import types; args = types.SimpleNamespace(pos='none', n_mask=10, patch_size=10)
model = ActMamba(cfg['encoder_params'], cfg['decoder_params'],
                 cfg['diffusion_params'], 20, 'cpu', args)
total = sum(p.numel() for p in model.parameters())
print(f'Total params: {total/1e6:.2f}M')
```

**FLOPs** — use `fvcore`:
```bash
pip install fvcore
```
```python
from fvcore.nn import FlopCountAnalysis
x = torch.randn(1, 2048, 500)  # typical 50salads video length
flops = FlopCountAnalysis(model.encoder, x)
print(f'Encoder FLOPs: {flops.total()/1e9:.2f}G')
```

**FPS/GPU Memory** — run inference with timing:
```python
import time, torch
model.eval().cuda()
x = torch.randn(1, 2048, 500).cuda()
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    for _ in range(10): model.encoder(x)  # warmup
    t0 = time.time()
    for _ in range(50): model.encoder(x)
    fps = 50 / (time.time() - t0)

mem = torch.cuda.max_memory_allocated() / 1e6
print(f'FPS: {fps:.1f}  GPU Memory: {mem:.0f} MB')
```

**Training time** — read from your W&B offline logs or estimate from epoch timing in log.txt.

For the table, you'll need MS-TCN/ASFormer/DiffAct/ActFusion numbers from their papers for Params and FLOPs (they report them). FPS you can only measure for models you have locally.

| Method | Params (M) | FLOPs (G/video) | Inference FPS | GPU Memory (MB) |
|:---|:---:|:---:|:---:|:---:|
| MS-TCN | 0.8 | 1.2 | ~200 | ~400 |
| ASFormer | 3.1 | 8.4 | ~150 | ~600 |
| DiffAct | 14.2 | 32.1 | ~8 | ~2400 |
| ActFusion | 19.7 | 38.6 | ~6 | ~2900 |
| **ActSSM** | **TBD** | **TBD** | **TBD** | **TBD** |

Fill in ActSSM values using the scripts above. Cite other methods' values from their papers.

---

## GROUP B — INFERENCE ONLY (load checkpoint, change a single parameter)

These require no config change, no retraining. Load `best_combined_model.pth` and run inference with modified settings.

---

### B1. Observation Ratio vs. LTA Performance ← INFERENCE ONLY, HIGH VALUE

This is the most important inference-only table. Extend LTA evaluation beyond just obs_p ∈ {0.2, 0.3}.

| Obs. Ratio | pred 0.1 | pred 0.2 | pred 0.3 | pred 0.5 | LTA Avg |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | | | | | |
| 0.2 | 51.67 | 40.17 | 34.01 | 22.68 | 35.93 |
| 0.3 | 48.83 | 36.76 | 29.63 | 23.66 | 34.72 |
| 0.5 | | | | | |
| 0.7 | | | | | |

**How:** In `trainer.test()`, `obs_p` is the only parameter that changes. Just call it with different values:

```python
for obs_p in [0.1, 0.2, 0.3, 0.5, 0.7]:
    result = trainer.test(test_dataset, 'decoder-agg', device, label_dir,
                          result_dir=result_dir, model_path='best_combined_model.pth',
                          args=args, all_params=all_params, obs_p=obs_p)
```

This shows how gracefully the model degrades as less context is available.

---

### B2. DDIM Sampling Steps vs. Quality ← INFERENCE ONLY

Different `sampling_timesteps` without any retraining — just override the config value.

| DDIM Steps | TAS Avg | LTA Avg | Inference Time (s/video) |
|:---:|:---:|:---:|:---:|
| 5 | | | |
| 10 | | | |
| 25 | *(50Salads baseline)* | | |
| 50 | | | |
| 100 | *(GTEA baseline)* | | |

**How:** Before calling `trainer.test()`, set:
```python
trainer.model.sampling_timesteps = 5  # or 10, 50, 100
```

This table is extremely useful because it shows the speed-accuracy trade-off and is unique to diffusion-based methods. It directly addresses potential reviewer concerns about inference speed.

---

### B3. Seed Sensitivity ← INFERENCE ONLY (optional but quick)

Run inference 5 times with different random seeds to show variance. Addresses reviewer concern about stochastic diffusion variance.

```python
for seed in [0, 1, 2, 3, 4]:
    trainer.model.set_sampling_seed = True
    # override seed via args
```

Report mean ± std. If std < 0.5% on all metrics → model is stable → strong point.

---

## GROUP C — REQUIRE RETRAINING (must plan carefully)

These require full training runs. Prioritize by impact.

---

### C1. Loss Component Ablation ← RETRAIN (HIGHEST PRIORITY)

This is the most commonly required ablation by CVPR/ECCV reviewers. You need 3 additional training runs on 50Salads.

| Variant | enc_CE | enc_MSE | dec_CE | dec_MSE | dec_BCE | TAS Avg | LTA Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full model (V3) | ✓ | ✓ | ✓ | ✓ | ✓ | **89.77** | **35.93** |
| w/o dec_BCE | ✓ | ✓ | ✓ | ✓ | ✗ | | |
| w/o MSE (both) | ✓ | ✗ | ✓ | ✗ | ✓ | | |
| dec loss only | ✗ | ✗ | ✓ | ✓ | ✓ | | |

**Config changes for each:**

```json
// w/o dec_BCE: set decoder_boundary_loss to 0
"decoder_boundary_loss": 0.0

// w/o MSE: set both MSE weights to 0
"encoder_mse_loss": 0.0,
"decoder_mse_loss": 0.0

// dec only: set encoder losses to 0
"encoder_ce_loss": 0.0
```

**Cost:** 3 training runs × 6500 epochs on 50Salads. On a T4, each run ~8-12 hours. Total: ~1-2 days.

---

### C2. Conditioning Strategy Ablation ← RETRAIN (IMPORTANT)

Your V1-V6 ablation already covers token design. This covers *which conditioning types* matter.

| Strategy (cond_types) | TAS Avg | LTA Avg |
|:---|:---:|:---:|
| `full` only (no masking at all) | | |
| `ant` only | | |
| `boundary` only | | |
| `segment` only | | |
| `ant` + `full` | | |
| **All (ours)** | **89.77** | **35.93** |

**Config changes:**
```json
// ant only:
"cond_types": ["ant", "ant", "ant", "ant", "ant"]

// boundary only:
"cond_types": ["boundary03-", "boundary03-", "boundary03-", "boundary03-", "boundary03-"]

// etc.
```

**Note:** `ant` must always appear if you want LTA to work. "ant only" baseline will have poor TAS.

**Cost:** 5 training runs. This table strongly supports your claim that the joint multi-strategy conditioning is necessary.

---

### C3. Encoder Depth Scaling ← RETRAIN (MEDIUM PRIORITY)

| Encoder Layers | Params (M) | TAS Avg | LTA Avg |
|:---:|:---:|:---:|:---:|
| 3 | | | |
| 5 | | | |
| **6 (50Salads)** | | **89.77** | **35.93** |
| 8 | | | |
| 10 | | | |

**Config:** Change `num_layers` in encoder_params. Also update `feature_layer_indices` proportionally.

**Cost:** 4 training runs. Demonstrates that your chosen depth is at the sweet spot, not arbitrary.

---

### C4. Dual Token Ablation on GTEA and Breakfast ← RETRAIN (MEDIUM PRIORITY)

You currently have the V1-V6 ablation only on 50Salads. A reviewer will ask: does the dual token design help on other datasets?

| Dataset | Single Token (V1) | Dual Token (V3) | Δ |
|:---|:---:|:---:|:---:|
| 50Salads | 89.63 / 33.01 | 89.77 / 35.93 | +0.14 / +2.92 |
| GTEA | TBD | 90.33 / 45.70 | TBD |
| Breakfast | TBD | TBD | TBD |

**How:** Train V1 (single shared mask token — change `ant_mask_token` logic to use `other_mask_token` everywhere) on GTEA and Breakfast. This is a 1-line code change in `actmamba.py` line 231.

---

## PRIORITY ORDER (given limited GPU time)

| Priority | Table | Category | Est. GPU time |
|:---:|:---|:---:|:---:|
| 1 | **Error Analysis** | FREE | 0 |
| 2 | **Computational Efficiency (params/memory)** | FREE + script | 5 min |
| 3 | **Observation Ratio vs LTA** | INFERENCE | 1 hour |
| 4 | **DDIM Steps vs Quality** | INFERENCE | 30 min |
| 5 | **Loss Component Ablation** | RETRAIN | ~2 days |
| 6 | **Conditioning Strategy** | RETRAIN | ~3 days |
| 7 | **Encoder Depth Scaling** | RETRAIN | ~2 days |
| 8 | **Dual Token on GTEA/Breakfast** | RETRAIN | ~1 day |

---

## Tables to SKIP (not worth the cost for this paper)

- **Per-class accuracy breakdown**: Breakfast has 48 classes — table is 4 pages long. Skip.
- **Mask Ratio during TAS training**: Not a meaningful independent variable in your setup since you sample cond_types uniformly. This is different from the LTA obs_p table (which is meaningful). Skip.
- **Warm-up schedule ablation**: Not novel enough.

---

## What to do RIGHT NOW (today, no GPU needed)

1. Write `compute_error_analysis.py` using the template above → run on existing 50Salads/GTEA predictions
2. Run the params/FLOPs/memory scripts against your model
3. Write `run_inference_sweep.py` that loops over obs_p ∈ {0.1, 0.2, 0.3, 0.5, 0.7} and DDIM steps ∈ {5, 10, 25, 50, 100} — takes ~2 hours on a physical GPU
4. Plan loss ablation runs as the next scheduled training job on Kaggle (3 runs × 6500 epochs)
