# TCFMAD

Cleaned, shareable copy of a TCFMAD DINOv3 anomaly detection project for VisA-style few-shot anomaly detection experiments.

This copy keeps the core code, configs, scripts, and notes needed to understand the project, while removing datasets, checkpoints, logs, and other local artifacts that are not suitable for a public repository snapshot.

## Core Ideas

This variant centers on two project-specific additions beyond the upstream baseline.

### 1. Defect-Patch Synthesis

The training pipeline can replace the default `CutPasteUnion` augmentation with `synthesis.mode=segpatch_folder`.

In this mode, the trainer:
- reads same-class defect patches from `train/<class>/seg_patch/`
- pastes one sampled defect patch onto a clean training image to build `imgs_abn`
- keeps the target branch on the clean image and uses the abnormal image only for the context branch
- supports configurable patch area, feathered edges, and optional color matching

Relevant implementation entrypoints:
- `tcfmad/src/utils/synthesis.py`
- `tcfmad/src/train.py`
- `tcfmad/configs/config.yaml`

### 2. Multi-Layer Consistency Learning

The training pipeline can enable a multi-layer branch with:
- `training.multi_layer_mean=true`
- `training.consistency_loss=true`

The configuration name keeps the legacy `multi_layer_mean` label, but the current implementation uses per-dimension learned fusion over three adjacent encoder layers. In practice, the model:
- extracts three neighboring encoder features around `n_layer`
- fuses target features and context features with learnable per-dimension layer weights
- applies the predictor on the fused context features
- optimizes the main feature-matching loss on the fused target/context features
- optionally adds a consistency loss that aligns each per-layer prediction with the fused prediction

Relevant implementation entrypoints:
- `tcfmad/src/tcfmad.py`
- `tcfmad/src/train.py`
- `tcfmad/src/AD.py`
- `tcfmad/configs/config.yaml`

## Data Format

See `DATA_FORMAT.md` for the expected training, `seg_patch`, and test directory layouts.

## Repository Layout

- `tcfmad/`: main package, configs, model code, training and inference entrypoints
- `scripts/`: helper shell scripts kept from the working repo
- `tools/`: dataset and audit utilities kept from the working repo
- `docs/`: project notes and change summaries, plus clean-copy audit docs
- `assets/`: lightweight static assets

## What Is Intentionally Excluded

This clean copy excludes:
- local datasets and few-shot sample folders
- logs and outputs
- checkpoints and other large binary artifacts
- editor metadata and Python caches
- local command-notes files that were only useful on the original workstation

## Environment

The original project expects Python 3.10 and installation via:

```bash
pip install -r requirements.txt
pip install -e .
```

## Entry Point

The main entrypoint remains:

```bash
python tcfmad/main.py
```

Typical modes in the project are:
- `mode=train app=train_dinov3`
- `mode=AD app=test`
- `mode=demo app=test`

Some inherited scripts and historical docs still contain workstation-specific paths. Review `docs/forgithub_report.md` before publishing or sharing broadly.

## Refreshing This Clean Copy

To rebuild this clean copy from a sibling working repo:

```bash
bash scripts/refresh_from_source.sh
```

If the source repo is not located at `../TCFMAD_source`, set it explicitly:

```bash
SOURCE_ROOT=/path/to/TCFMAD_source bash scripts/refresh_from_source.sh
```
