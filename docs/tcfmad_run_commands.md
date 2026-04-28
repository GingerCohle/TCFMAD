# TCFMAD Run Commands (Absolute Local Paths)
Generated on: 2026-03-05

All commands assume:

```bash
cd /media/disk/kejunjie_only/dino_anomaly/TCFMAD
conda activate tcfmad
```

## Local dataset roots (verified)

- MVTec: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/mvtec`
- VisA: `/media/disk/kejunjie_only/dino_anomaly/TCFMAD/visa`

## A) MVTec

## A1. Run demo/inference with released pretrained projector (no training)

```bash
python tcfmad/main.py \
  mode=demo \
  app=test \
  data.dataset=mvtec \
  data.data_name=mvtec_1shot \
  data.test_root=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/mvtec \
  diy_name=_pretrained \
  testing.segmentation_vis=True
```

Expected outputs:
- checkpoint loaded from `logs/mvtec_1shot/dinov3_pretrained/pretrained.pth.tar`
- heatmaps under `logs/mvtec_1shot/dinov3_pretrained/demo/heatmaps/`

## A2. Build few-shot train subset (example: 1-shot)

```bash
python tcfmad/src/sample.py \
  source=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/mvtec \
  target=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/fewshot/mvtec_1shot \
  num_samples=1 \
  seed=42
```

Expected outputs:
- `fewshot/mvtec_1shot/train/<class_name>/*`

## A3. Train (few-shot)

```bash
python tcfmad/main.py \
  mode=train \
  app=train_dinov3 \
  data.dataset=mvtec \
  data.data_name=mvtec_1shot \
  data.data_path=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/fewshot \
  diy_name=_local
```

Expected outputs:
- `logs/mvtec_1shot/dinov3_local/params.yaml`
- `logs/mvtec_1shot/dinov3_local/train.csv`
- `logs/mvtec_1shot/dinov3_local/train-step*.pth.tar` (every 100 global steps)

## A4. Evaluate trained checkpoint (`mode=AD`)

```bash
python tcfmad/main.py \
  mode=AD \
  app=test \
  data.dataset=mvtec \
  data.data_name=mvtec_1shot \
  data.test_root=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/mvtec \
  diy_name=_local \
  app.ckpt_step=100 \
  testing.segmentation_vis=True
```

Replace `app.ckpt_step` with an existing step from `train-step*.pth.tar`.

Expected outputs:
- `logs/mvtec_1shot/dinov3_local/eval/100/AD_eval.csv`
- optional segmentation grids under `.../eval/100/segmentation/`

## B) VisA

## B1. Run demo/inference with released pretrained projector (no training)

```bash
python tcfmad/main.py \
  mode=demo \
  app=test \
  data.dataset=visa \
  data.data_name=visa_4shot \
  data.test_root=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/visa \
  diy_name=_pretrained \
  testing.segmentation_vis=True
```

Expected outputs:
- checkpoint loaded from `logs/visa_4shot/dinov3_pretrained/pretrained.pth.tar`
- heatmaps under `logs/visa_4shot/dinov3_pretrained/demo/heatmaps/`

## B2. Build few-shot train subset (example: 1-shot)

```bash
python tcfmad/src/sample.py \
  source=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/visa \
  target=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/fewshot/visa_1shot \
  num_samples=1 \
  seed=42
```

Expected outputs:
- `fewshot/visa_1shot/train/<class_name>/*`

## B3. Train (few-shot)

```bash
python tcfmad/main.py \
  mode=train \
  app=train_dinov3 \
  data.dataset=visa \
  data.data_name=visa_1shot \
  data.data_path=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/fewshot \
  diy_name=_local
```

Expected outputs:
- `logs/visa_1shot/dinov3_local/params.yaml`
- `logs/visa_1shot/dinov3_local/train.csv`
- `logs/visa_1shot/dinov3_local/train-step*.pth.tar`

## B4. Evaluate trained checkpoint (`mode=AD`)

```bash
python tcfmad/main.py \
  mode=AD \
  app=test \
  data.dataset=visa \
  data.data_name=visa_1shot \
  data.test_root=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/visa \
  diy_name=_local \
  app.ckpt_step=100 \
  testing.segmentation_vis=True
```

Expected outputs:
- `logs/visa_1shot/dinov3_local/eval/100/AD_eval.csv`
- optional segmentation grids under `.../eval/100/segmentation/`

## Released checkpoint download steps (documented, not executed)

README links:
- MVTec: `mvtec_1shot.zip`, `mvtec_2shot.zip`, `mvtec_4shot.zip`
- VisA: `visa_1shot.zip`, `visa_2shot.zip`, `visa_4shot.zip`

Example:

```bash
cd /media/disk/kejunjie_only/dino_anomaly/TCFMAD/logs
curl -L --fail -o mvtec_1shot.zip https://www.campar.in.tum.de/public_datasets/2025_tcfmad/mvtec_1shot.zip
unzip -o mvtec_1shot.zip
```

After extraction, expected layout for demo:
- `logs/<data_name>/dinov3_pretrained/params.yaml`
- `logs/<data_name>/dinov3_pretrained/pretrained.pth.tar`

## Important behavior constraints

- `mode=AD` loads `train-step<app.ckpt_step>.pth.tar` (not `pretrained.pth.tar`).
- `mode=demo` loads `pretrained.pth.tar` and writes visualization heatmaps.
- `data.test_root` must contain the dataset token (`mvtec` or `visa`) in the path string due assertions in code.
