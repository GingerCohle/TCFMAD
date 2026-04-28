# TCFMAD_visapcb: Mean-Feature + Consistency-Loss Change Summary

## Method Used

This summary was produced from the current working tree using:

- `git rev-parse --is-inside-work-tree` -> `true`
- `git status --short`
- `git diff --name-only`
- `git diff` on the feature-relevant files
- `git log -n 5 --oneline`
- `rg` for:
  - `training.multi_layer_mean`
  - `training.consistency_loss`
  - `training.cons_lambda`
  - `testing.multi_layer_mean`
  - `get_intermediate_layers`
  - `CONS-MEAN`
  - helper names such as `multi_layer_features`, `target_multi_layer_features`, `mean_feature`

Notes:

- The worktree is a Git repo with unstaged changes and no staged changes.
- `git diff --name-only` also shows unrelated dirty files such as deleted `assets/*`, plus edits in `tcfmad/main.py` and `tcfmad/src/datasets/dataset.py`.
- This report only covers files that actually implement or expose the mean-feature + consistency-loss behavior, verified by code search and direct inspection.

Recent commits at analysis time:

- `658aa9e modify training config`
- `c3327b9 update ack`
- `671e2c6 update arxiv`
- `57d62e3 fix bug on datasets`
- `0e7a9ad update gitignore`

## Feature-Relevant Changed Files

### Tracked modified files in `git diff`

- `tcfmad/configs/config.yaml`
  - Adds the new runtime flags under `training.*` and `testing.*`.
- `tcfmad/src/tcfmad.py`
  - Adds the helper API for resolving three layers, extracting them, and averaging them into a single mean feature.
- `tcfmad/src/train.py`
  - Gates the new train-time behavior with `training.multi_layer_mean`.
  - Adds optional consistency loss controlled by `training.consistency_loss` and `training.cons_lambda`.
- `tcfmad/src/AD.py`
  - Gates the new eval/demo behavior with `testing.multi_layer_mean`.
  - Replaces the single-layer encoder feature with `h_mean` when enabled.

### Untracked feature-related files present in the worktree

- `docs/multilayer_consistent_mean.md`
  - Human-facing usage notes and command examples for the feature.
- `scripts/test_visa_tmp_2shot_fast.sh`
  - Wrapper script updated to pass `testing.multi_layer_mean` and optional `testing.layers`.

### Relevant but not feature-modified context files

- `tcfmad/main.py`
  - Already provides the train/eval entry routing used by this feature.
  - No mean-feature-specific keywords or logic were found here.
- `tcfmad/src/datasets/dataset.py`
  - Still provides the existing train/test data loading and preprocessing.
  - No mean-feature-specific keywords or logic were found here.
- `tcfmad/configs/app/train_dinov3.yaml`
  - Not changed for this feature, but still defines the anchor layer `meta.n_layer: 3`.

## Exact Implementation Locations

### 1. Configuration flags

File: `tcfmad/configs/config.yaml`

- `training.multi_layer_mean: false` at lines 73-77
- `training.consistency_loss: false` at lines 73-77
- `training.cons_lambda: 0.1` at lines 73-77
- `training.layers: null` at lines 73-77
- `testing.multi_layer_mean: false` at lines 81-90
- `testing.layers: null` at lines 81-90

Anchor config:

- `tcfmad/configs/app/train_dinov3.yaml:9`
  - `meta.n_layer: 3`

Default resolved layer triplet:

- `tcfmad/src/tcfmad.py:52-55`
  - `resolve_feature_layers(...)` maps `n_layer=3` to `[2, 3, 4]`, clamped at `>= 1`.

### 2. Multi-layer extraction and fusion helper

File: `tcfmad/src/tcfmad.py`

- `VisionModule.resolve_feature_layers(...)` at lines 52-55
  - Resolves either an explicit `layers` override or the default `[n_layer-1, n_layer, n_layer+1]`.
- `VisionModule.multi_layer_features(...)` at lines 57-59
  - Public helper to extract multiple layers for the context branch.
- `VisionModule.target_multi_layer_features(...)` at lines 61-63
  - No-grad version for the target branch.
- `VisionModule.mean_feature(...)` at lines 65-67
  - Performs the actual fusion:
    - `torch.stack(features, dim=0).mean(dim=0)`
- `VisionModule._extract_layers(...)` at lines 101-158
  - Uses one `get_intermediate_layers(...)` call for `dinov2` / `dinov3` / `dino` when possible.
  - For DINO-family models it converts "last-k layer" notation into absolute block ids.
- `VisionModule._extract_single(...)` at lines 160-196
  - Preserves the original single-layer behavior for the OFF path.

Relevant encoder API:

- `tcfmad/src/dinov2/models/vision_transformer.py:375-399`
  - `get_intermediate_layers(...)` accepts either:
    - an integer `n` = last `n` layers, or
    - a sequence of absolute block ids.

### 3. Train-time gate and loss implementation

File: `tcfmad/src/train.py`

- Config parsing at lines 65-76
  - Reads `training.multi_layer_mean`, `training.consistency_loss`, `training.cons_lambda`, `training.layers`.
- Enable log at lines 70-76
  - Emits `[CONS-MEAN][train] enabled layers=...` only when the feature is on.
- Consistency-loss helper at lines 162-164
  - `_consistency_loss_fn(layer_preds, mean_pred)`
- Profile branch mean-feature path at lines 243-293
  - Multi-layer extraction, mean-feature computation, predictor call, main loss, optional consistency loss.
- Non-profile branch mean-feature path at lines 304-327
  - Same logic without the profiling wrappers.

### 4. Eval / demo gate

File: `tcfmad/src/AD.py`

- Eval config parsing at lines 74-103
  - Reads `testing.multi_layer_mean` and `testing.layers`.
  - Emits `[CONS-MEAN][eval] enabled layers=...` only when the feature is on.
- Eval encoder replacement at lines 167-171
  - `enc = mean_feature(target_multi_layer_features(...))` when enabled.
- Scoring path at lines 177-189
  - Unchanged scoring after `enc` is formed:
    - predictor output
    - MSE token loss
    - Top-K image score
    - bilinear heatmap upsampling
- Demo path at lines 368-372
  - Mirrors eval: `enc = mean_feature(...)` when enabled.

## Train-Time Forward Path and Loss

### Entrypoint and data loading

Train entrypoint is unchanged:

1. `tcfmad/main.py:56-65`
   - `mode=train` routes to `src.train.main(args=params)`.
2. `tcfmad/src/train.py:81-93`
   - `build_dataloader(mode="train", ...)`
3. `tcfmad/src/datasets/dataset.py:162-188`
   - `TrainDataset` loads the few-shot images from `<train_root>/train/*`.
   - Existing transforms remain unchanged.

### Mean-feature ON path

For `training.multi_layer_mean=true`, the train loop does the following:

1. Extract target-side multi-layer clean features
   - `h_layers = target_multi_layer_features(imgs, paths, n_layer, layers)`
   - Code: `train.py:252-254` and `train.py:312-314`
2. Extract context-side multi-layer features from either:
   - clean `imgs`, or
   - cut-paste augmented `imgs_abn`
   - Code: `train.py:255-262` and `train.py:311-317`
3. Fuse layers by mean
   - `h_mean = mean_feature(h_layers)`
   - `z_mean = mean_feature(z_layers)`
   - Code: `train.py:287`, `train.py:318-319`
4. Run the existing predictor path on the mean context feature
   - `p_mean = predictor(dropout(z_mean))`
   - Code: `train.py:271-272` or `train.py:320`
5. Compute the main loss against the clean target mean feature
   - `L_main = loss_fn(h_mean, p_mean)`
   - Code: `train.py:289` or `train.py:321`

### Consistency loss

When `training.consistency_loss=true`, the code also computes:

- per-layer predictor outputs:
  - `p_l = predictor(dropout(z_l))`
  - Code: `train.py:275-278` or `train.py:323-325`
- consistency loss:
  - `L_cons = mean_l MSE(p_l, p_mean)`
  - Code: `train.py:162-164`, `train.py:292`, `train.py:327`

Total loss:

```text
L_total = L_main + cons_lambda * L_cons
```

Important implementation detail:

- `p_mean` is **not detached** in `L_cons`.
- There is no `.detach()` and no `torch.no_grad()` around `p_mean` in `train.py:287-293` or `train.py:318-327`.
- Therefore `L_cons` backpropagates through both:
  - the per-layer prediction branches `p_l`, and
  - the mean-feature prediction branch `p_mean`.

Important model-path detail:

- For the default `dinov3` path, this feature uses the existing **predictor** only.
- The optional `projector` still exists only for special backbones such as `dinosiglip` in `tcfmad/src/tcfmad.py:71-96`.
- There is no new train-time projector call introduced by the mean-feature feature for `dinov3`.

## Test / Eval Forward Path and How It Matches Train

### Entrypoint

Eval entrypoint is unchanged:

1. `tcfmad/main.py:66-76`
   - `mode=AD` loads the saved train `params.yaml`
   - restores `meta`
   - resolves `train-step<ckpt_step>.pth.tar`
   - routes into `src.AD.main(args=params)`
2. `tcfmad/src/AD.py:51-261`
   - `_evaluate_single_ckpt(...)`

### Mean-feature ON path

For `testing.multi_layer_mean=true`, eval does:

1. Extract target-side multi-layer features from the input image
   - `enc_layers = target_multi_layer_features(img, paths, n_layer, layers)`
   - Code: `AD.py:167-168`
2. Fuse them by mean
   - `enc = mean_feature(enc_layers)`
   - Code: `AD.py:169`
3. Keep the original downstream scoring unchanged
   - `pred = model.predict(enc)` at `AD.py:177`
   - `l = mse(enc, pred).mean(dim=2)` at `AD.py:183`
   - `topk = torch.topk(l, K, dim=1)...` at `AD.py:185`
   - heatmap resize at `AD.py:187-189`

The demo path mirrors this exactly:

- `AD.py:368-372`

### Match to train

Train and test now match at the representation level when the flags are ON:

- Train main branch target:
  - `h_mean = mean([h2, h3, h4])`
- Test encoder feature:
  - `enc = mean([h2, h3, h4])`

In both cases the same predictor is applied to the mean feature representation rather than to a post-hoc fusion of multiple anomaly maps.

### No `mean_std` fusion in this consistent mode

Direct code evidence:

- `rg -n "mean_std" tcfmad scripts docs` returned no hits in this repo for the current implementation.
- In `AD.py`, the ON path is only:
  - multi-layer extraction -> `mean_feature(...)` -> predictor -> MSE -> Top-K / heatmap.
- No alternate `mean_std` fusion branch exists in `tcfmad/src/AD.py`.

## What Remains Unchanged When the Flags Are OFF

### Train OFF path

When `training.multi_layer_mean=false`, the original single-layer logic remains:

- `h = target_features(imgs, paths, n_layer=self.n_layer)`
- `p` comes from `context_features(...)` on either clean or augmented input
- loss is the existing `MSE` or `smooth_l1`
- Code: `train.py:244-250` and `train.py:304-309`

### Eval OFF path

When `testing.multi_layer_mean=false`, eval remains single-layer:

- `enc = target_features(img, paths, n_layer=n_layer)`
- then unchanged predictor/MSE/Top-K/heatmap/metrics
- Code: `AD.py:170-171`

### Other unchanged pieces

- `main.py` routing remains the same.
- Dataset loading and image preprocessing remain the same.
- `meta.n_layer` remains the existing anchor key; the new mode builds on it instead of replacing it.
- Old scoring and metrics stay the same after `enc` is formed.
- Extra `[CONS-MEAN]` logging is emitted only when the feature is enabled.

## Quick Sanity Evidence

The working tree contains one successful eval log at `logs/ad_visa_tmp_2shot_fast.log`.
Relevant lines:

```text
Testing multi-layer mean: true
INFO:evaluator:[CONS-MEAN][eval] enabled layers=[2, 3, 4]
INFO:evaluator:Mean | AUROC_i 0.9428 | AUPR_i 0.9398 | AUROC_p 0.9974 | PRO-AUC 0.9838
```

This is consistent with the code path in `scripts/test_visa_tmp_2shot_fast.sh` and `tcfmad/src/AD.py`.

