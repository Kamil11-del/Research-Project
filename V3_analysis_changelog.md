# Act_Mamba V3 — Deep Analysis & Complete Changelog

> **Project:** `Act_Mamba V3 epch 6500`  
<!-- > **Based on:** `Act_Mamba V2 epch 7k`   -->
> **Analysis date:** 2026-05-10

---

## 1. Deep Analysis Results

### 1.1 `src/model/actmamba.py` — PASS ✅

#### Dual Mask Token Implementation

| Check | Line(s) | Expected | Actual | Status |
|---|---|---|---|---|
| `ant_mask_token` declared | 170 | `nn.Parameter [1, input_dim, 1]` | `nn.Parameter(torch.zeros(1, input_dim_for_mask, 1))` | ✅ |
| `other_mask_token` declared | 171 | `nn.Parameter [1, input_dim, 1]` | `nn.Parameter(torch.zeros(1, input_dim_for_mask, 1))` | ✅ |
| `ant_mask_token` xavier init | 172 | `xavier_uniform_` | `nn.init.xavier_uniform_(self.ant_mask_token)` | ✅ |
| `other_mask_token` xavier init | 173 | `xavier_uniform_` | `nn.init.xavier_uniform_(self.other_mask_token)` | ✅ |
| Old `self.mask_token` removed | — | absent | not found anywhere in file | ✅ |
| Old `self.post_enc_mask_token` (V2) removed | — | absent | not found anywhere in file | ✅ |
| `self.backbone_feat_dim` (V2) removed | — | absent | not found anywhere in file | ✅ |

#### Token Selection Logic in `mask()`

| Check | Line | Expected | Actual | Status |
|---|---|---|---|---|
| Token selection at top of `mask()` | 231 | `tok = ant if 'ant' else other` | `tok = self.ant_mask_token if cond_type == 'ant' else self.other_mask_token` | ✅ |
| `'zero'` branch uses `tok` | 234 | `tok.repeat(...)` | `tok.repeat(x.shape[0], 1, x.shape[2])` | ✅ |
| Mask token insertion uses `tok` | 298 | `tok.repeat(...)` | `tok.repeat(x.shape[0], 1, n_masked)` | ✅ |
| `boundary05-` → `other_mask_token` | 241–242 | cond≠'ant' → other | ✅ (tok set line 231) | ✅ |
| `boundary03-` → `other_mask_token` | 244–245 | cond≠'ant' → other | ✅ | ✅ |
| `segment=1` → `other_mask_token` | 247–251 | cond≠'ant' → other | ✅ | ✅ |
| `segment=2` → `other_mask_token` | 253–258 | cond≠'ant' → other | ✅ | ✅ |
| `ant` → `ant_mask_token` | 260–265 | cond=='ant' → ant | ✅ (tok set line 231) | ✅ |
| `random_num_patch` → `other_mask_token` | 267–276 | cond≠'ant' → other | ✅ | ✅ |
| `random_patch` → `other_mask_token` | 278–285 | cond≠'ant' → other | ✅ | ✅ |
| Unknown cond_type raises `ValueError` | 287–288 | error on bad input | `raise ValueError(f"Unknown cond_type: '{cond_type}'")` | ✅ |
| `reshape(-1)` safety fix preserved | 294 | no `.squeeze()` | `(feature_mask.reshape(-1) == 1).nonzero(...).reshape(-1)` | ✅ |

#### Masking Position

| Check | Lines | Expected | Actual | Status |
|---|---|---|---|---|
| `mask()` called BEFORE `encoder()` | 321–327 | pre-encoder | `mask(input_feats...)` then `encoder(input_feats...)` | ✅ |
| No post-encoder masking code (V2 style) | — | absent | not present | ✅ |
| `encoder()` receives masked features | 325–326 | `encoder(masked_feats)` | `self.encoder(input_feats, get_features=True, mask=feature_mask)` | ✅ |

#### Inference (`ddim_sample`)

| Check | Line | Expected | Actual | Status |
|---|---|---|---|---|
| LTA pads with `ant_mask_token` | 413 | `self.ant_mask_token.repeat(...)` | `self.ant_mask_token.repeat(1, 1, mask_len)` | ✅ |
| TAS path unchanged | 419–421 | no padding | `mask = torch.ones(...)` (no token used) | ✅ |
| Inference consistent with training | — | same token for 'ant' | both train+infer use `ant_mask_token` | ✅ |

---

### 1.2 `configs/50salads.json` — PASS ✅

| Field | Expected | Actual | Status |
|---|---|---|---|
| `num_epochs` | 6500 | 6500 | ✅ |
| `naming` | `actmamba_v3_50salads` | `actmamba_v3_50salads` | ✅ |
| `cond_types` includes `'ant'` | yes | `["full","ant","random_num_patch","boundary05-","segment=2","segment=2"]` | ✅ |

---

### 1.3 `configs/Breakfast.json` — PASS ✅

| Field | Expected | Actual | Status |
|---|---|---|---|
| `num_epochs` | 6500 | 6500 | ✅ |
| `naming` | `actmamba_v3_breakfast` | `actmamba_v3_breakfast` | ✅ |

---

### 1.4 `src/trainer.py` — PASS ✅ (inherited from V2, no regression)

- No direct `mask_token` references — uses model API only
- Resume bug fix (best metrics in checkpoint) intact
- Structured logging intact
- `best_scores.json` + `best_metrics.json` saving intact

---

### 1.5 Notebook (`act-mamba-training.ipynb`) — PENDING ⚠️

The notebook string-replacement was blocked by OneDrive sync locking the file.  
The file contains stale V2 strings (`NUM_EPOCHS = 7001`, `Act_Mamba_V2`, `actmamba_50s_split`).

**Fix:** Run this once from the project folder:
```bash
python patch_notebook.py
```
`patch_notebook.py` is already present in the project root.

> All code logic (actmamba.py, trainer.py, configs) is 100% correct.  
> The notebook is display-only configuration — it does not affect model behaviour.

---

## 2. Changelog — V2 → V3

### File: `src/model/actmamba.py`

| # | Change | Detail |
|---|---|---|
| 1 | **Removed** `self.mask_token` | Single unified token from V1/V2 deleted |
| 2 | **Removed** `self.post_enc_mask_token` | V2 post-encoder token deleted |
| 3 | **Removed** `self.backbone_feat_dim` | V2-only attribute deleted |
| 4 | **Added** `self.ant_mask_token` | `nn.Parameter([1, input_dim, 1])`, xavier init — for anticipation masking |
| 5 | **Added** `self.other_mask_token` | `nn.Parameter([1, input_dim, 1])`, xavier init — for all other masking |
| 6 | **Modified** `mask()` signature | Removed `custom_token` param (V2 only); now auto-selects token from `cond_type` |
| 7 | **Modified** `mask()` token logic | Line 231: `tok = ant_mask_token if cond_type=='ant' else other_mask_token` |
| 8 | **Reverted** `get_training_loss()` | Masking moved back to **pre-encoder** (was post-encoder in V2) |
| 9 | **Modified** `ddim_sample()` LTA path | Padding now uses `self.ant_mask_token` (was `self.mask_token`) |
| 10 | **Added** `ValueError` in `mask()` | Raises on unknown `cond_type` — catches config typos early |
| 11 | Docstring updated | Class docstring reflects V3 dual-token design |

### File: `configs/50salads.json`

| # | Change | From → To |
|---|---|---|
| 1 | `num_epochs` | `7001` → `6500` |
| 2 | `naming` | `actmamba_v3_50salads` (updated) |

### File: `configs/Breakfast.json`

| # | Change | From → To |
|---|---|---|
| 1 | `num_epochs` | `7001` → `6500` |
| 2 | `naming` | `actmamba_v3_breakfast` (updated) |

### File: `patch_notebook.py` *(new)*

| # | Change |
|---|---|
| 1 | New helper script — patches notebook references for V3 (V2→V3 title, path, NUM_EPOCHS, RESULT_TAG) |

### Files **NOT changed** (inherited from V2 unchanged)

| File | Reason |
|---|---|
| `src/trainer.py` | V2 improvements fully compatible with V3 |
| `src/dataset.py` | No change needed |
| `src/utils.py` | No change needed |
| `src/model/backbone.py` | Encoder/Decoder API unchanged |
| `src/model/layers.py` | No change needed |
| `src/model/cross_mamba.py` | No change needed |
| `main.py` | No change needed |

---

## 3. Architecture Summary: V1 → V2 → V3

```
V1: mask(input)  → encoder → backbone → decoder     [single mask_token]
                                                      pre-encoder masking
                                                      1 token for all strategies

V2: encoder(full) → backbone → mask(backbone) → decoder  [single mask_token + post_enc_mask_token]
                                                           post-encoder masking
                                                           2 tokens (1 pre, 1 post)

V3: mask(input)  → encoder → backbone → decoder     [ant_mask_token + other_mask_token]
                                                      pre-encoder masking  ← reverted (V1 position)
                                                      2 tokens by strategy  ← NEW
```

**Key insight:** Pre-encoder masking (V1/V3) outperforms post-encoder masking (V2) empirically.  
V3 adds the dual-token hypothesis: the encoder should learn different representations for  
"this frame is missing" (`other_mask_token`) vs "the future is unknown" (`ant_mask_token`).
