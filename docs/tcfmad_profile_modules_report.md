# TCFMAD Per-module GPU Memory Profiling Report

## 1) Command used

```bash
CUDA_VISIBLE_DEVICES=1 python tcfmad/main.py \
    mode=train \
    app=train_dinov3 \
    data.dataset=visa \
    data.data_name=visa_tmp \
    data.data_path=/media/disk/kejunjie_only/dino_anomaly/TCFMAD \
    diy_name=_2shot \
    profile=true \
    profile_max_steps=20
```

Detailed log: `logs/tcfmad_profile_modules.log`

## 2) Environment snapshot (from log header)

- Timestamp: 2026-03-05T05:44:07-05:00
- GPU (nvidia-smi head): NVIDIA GeForce RTX 3090, Driver 550.120, NVIDIA-SMI CUDA 12.4
- Torch: `2.10.0+cu128`
- `torch.cuda.is_available()`: `True`
- `torch.version.cuda`: `12.8`
- `torch.cuda.device_count()`: `8`
- Profiling run configuration: `profile=true`, `profile_max_steps=20`, warmup skipped for first 3 steps
- Aggregated measured steps: `17` (steps 4..20)

## 3) Per-segment memory table (mean ± std, MB)

| Segment | delta_alloc_MB (mean ± std) | segment_peak_MB (mean ± std) |
|---|---:|---:|
| H2D | 24.00 ± 0.00 | 651.06 ± 0.00 |
| encoder_fwd | 48.23 ± 0.00 | 941.09 ± 0.00 |
| predictor_fwd | 3528.92 ± 0.12 | 4407.86 ± 0.12 |
| loss | 48.00 ± 0.00 | 4203.99 ± 0.12 |
| backward | -3483.45 ± 0.23 | 5158.65 ± 0.06 |
| optimizer_step | 0.00 ± 0.00 | 715.78 ± 0.12 |
| zero_grad | -45.47 ± 0.12 | 672.54 ± 0.12 |

## 4) Overall run peaks

- `max_memory_allocated`: **5158.72 MB**
- `max_memory_reserved`: **5582.00 MB**
- Activation-related peak (max over encoder/predictor/loss segment peaks): **4407.99 MB**

## 5) Bottleneck conclusions

- **Dominant segment by per-segment peak:** `backward` (segment peak mean **5158.65 MB**).
- **Dominant positive persistent allocation:** `predictor_fwd` (delta_alloc mean **+3528.92 MB**).
- `encoder_fwd` and `loss` add comparatively small positive deltas (~+48 MB each).
- `backward` and `zero_grad` release memory (negative deltas), indicating most forward activations are freed during/after gradient computation.

## 6) Next optimization targets (no implementation in this baseline)

- Activation checkpointing / mixed precision for predictor path.
- Reduce token count (crop size / patch stride / fewer layers).
- Avoid unnecessary tensor copies and redundant intermediate allocations.
