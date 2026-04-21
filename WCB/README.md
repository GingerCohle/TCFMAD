# SAM Fine-Tune Pipeline

This folder contains a minimal two-script pipeline for adapting SAM to industrial defect segmentation and then using the fine-tuned model for box-guided inference.

Files:
- `sam_finetune.py`: lightweight SAM fine-tuning with pixel-level defect masks
- `sam_infer_from_box.py`: box-guided inference with the fine-tuned SAM

## 0. Setup in a New Environment

If you are not using `foundad`, create a fresh environment first. A minimal dependency set for these two scripts is:

- Python 3.9 or 3.10
- PyTorch
- torchvision
- numpy
- pillow
- opencv-python
- segment-anything

A typical setup flow is:

```bash
conda create -n sam_ft python=3.10 -y
conda activate sam_ft
```

Install PyTorch first. Choose the command that matches your CUDA version from the official PyTorch site. For example, with CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining packages:

```bash
pip install numpy pillow opencv-python
pip install git+https://github.com/facebookresearch/segment-anything.git
```

If you already have a local `segment_anything` source tree, you can also install it in editable mode instead of pulling from GitHub:

```bash
pip install -e /abs/path/to/segment-anything
```

Quick import check:

```python
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("opencv:", cv2.__version__)
```

If this import check passes, the two scripts in this folder should run in that environment.

## 1. Minimal Commands

Fine-tune:

```bash
CUDA_VISIBLE_DEVICES=6 python tools/sam_ft_pipeline/sam_finetune.py \
  --train_root /data/my_dataset/paired_train \
  --sam_ckpt /data/checkpoints/sam_vit_b_01ec64.pth
```

Inference:

```bash
CUDA_VISIBLE_DEVICES=6 python tools/sam_ft_pipeline/sam_infer_from_box.py \
  --dataset_root /data/my_dataset/weak_box_data \
  --ft_ckpt /data/outputs/sam_ft/ckpts/best.pth \
  --sam_ckpt /data/checkpoints/sam_vit_b_01ec64.pth
```

`sam_model_type` can now be inferred automatically from the SAM checkpoint filename when it contains `vit_b`, `vit_l`, or `vit_h`.

## 2. Fine-Tuning Data Format

```text
train_root/
  img/
  seg_mask/
```

Rules:
- pair by filename stem
- masks must be binary `.png`
- image and mask size must match

## 3. Inference Data Format

```text
dataset_root/
  class_a/
    test/
      ko/
    ground_truth/
      ko/
```

Rules:
- `test/ko/<stem>.*` is the image
- `ground_truth/ko/<stem>.png` is the box mask
- output is `ground_truth/ko/<stem>_seg.png`

## 4. Main Outputs

Fine-tune default output root:
- `outputs/sam_ft/ckpts`
- `outputs/sam_ft/vis_train`

Inference default visualization dir:
- `outputs/sam_ft/vis_gen`

You can override these with `--output_root`. Advanced options like `--save_dir` and `--vis_dir` are still supported for manual control.
