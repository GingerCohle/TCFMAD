# Multilayer Consistent Mean Mode

## Overview

This repo routes `mode=train` and `mode=AD` from `tcfmad/main.py`.
The relevant implementation points for this change are:

- `tcfmad/src/tcfmad.py`
  - `VisionModule._extract(...)`
  - `VisionModule.target_features(...)`
  - new multilayer helper methods for extracting `[n_layer-1, n_layer, n_layer+1]` and computing the mean feature
- `tcfmad/src/train.py`
  - `Trainer.train(...)`
  - `Trainer._loss_fn(...)`
- `tcfmad/src/AD.py`
  - `_evaluate_single_ckpt(...)`
  - `_demo(...)`
- `tcfmad/configs/app/train_dinov3.yaml`
  - `meta.n_layer` is the existing base layer key used by both training and testing

## What Changed

An opt-in "consistent mean-feature" mode was added.
Default behavior is unchanged when the new switches stay `false`.

When enabled:

1. Three layers are resolved from `meta.n_layer`.
2. Default layer list is `[n_layer-1, n_layer, n_layer+1]` with lower clamp at `1`.
3. Training uses `h_mean = mean([h_l])` as the main branch feature.
4. Optional consistency loss is added on predictor outputs:

```text
L_cons = mean_l || p(h_l) - p(h_mean) ||^2
L = L_main(h_mean) + cons_lambda * L_cons
```

5. Evaluation and demo heatmaps also use `h_mean`.
6. No extra logs are printed unless the new mode is enabled.

## Config Switches

Top-level config keys in `tcfmad/configs/config.yaml`:

```yaml
training:
  multi_layer_mean: false
  consistency_loss: false
  cons_lambda: 0.1
  layers: null

testing:
  multi_layer_mean: false
  layers: null
```

Notes:

- `training.layers=null` and `testing.layers=null` both fall back to `[n_layer-1, n_layer, n_layer+1]`.
- `meta.n_layer` is still the anchor layer.
- `training.consistency_loss` only takes effect when `training.multi_layer_mean=true`.

## Usage

Baseline behavior remains:

```bash
python tcfmad/main.py \
    mode=train \
    app=train_dinov3 \
    data.dataset=visa \
    data.data_name=visa_tmp \
    data.data_path=/media/disk/kejunjie_only/dino_anomaly/compare_visa/TCFMAD_visapcb \
    diy_name=_2shot
```

Retrain with consistent mean-feature mode:

```bash
python tcfmad/main.py \
    mode=train \
    app=train_dinov3 \
    data.dataset=visa \
    data.data_name=visa_tmp \
    data.data_path=/media/disk/kejunjie_only/dino_anomaly/compare_visa/TCFMAD_visapcb \
    diy_name=_2shot \
    training.multi_layer_mean=true \
    training.consistency_loss=true \
    training.cons_lambda=0.1
```

Optional explicit layer override:

```bash
python tcfmad/main.py \
    mode=train \
    app=train_dinov3 \
    data.dataset=visa \
    data.data_name=visa_tmp \
    data.data_path=/media/disk/kejunjie_only/dino_anomaly/compare_visa/TCFMAD_visapcb \
    diy_name=_2shot \
    training.multi_layer_mean=true \
    training.consistency_loss=true \
    training.layers='[2,3,4]'
```

Standard eval command in this repo:

```bash
python tcfmad/main.py \
    mode=AD \
    data.dataset=visa \
    data.data_name=visa_tmp \
    data.test_root=/media/disk/kejunjie_only/dino_anomaly/compare_visa/TCFMAD/visa \
    diy_name=_2shot \
    app=test \
    app.ckpt_step=100 \
    testing.multi_layer_mean=true
```

Optional explicit layer override for eval:

```bash
python tcfmad/main.py \
    mode=AD \
    data.dataset=visa \
    data.data_name=visa_tmp \
    data.test_root=/media/disk/kejunjie_only/dino_anomaly/compare_visa/TCFMAD/visa \
    diy_name=_2shot \
    app=test \
    app.ckpt_step=100 \
    testing.multi_layer_mean=true \
    testing.layers='[2,3,4]'
```

If you want the exact same layer triplet at train/eval time, set both:

```bash
training.layers='[2,3,4]' testing.layers='[2,3,4]'
```
