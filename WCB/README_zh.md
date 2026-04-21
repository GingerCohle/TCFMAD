# SAM 微调与推理说明

这个目录只保留两份核心脚本：

- `sam_finetune.py`：用少量像素级缺陷 mask 对 SAM 做轻量微调
- `sam_infer_from_box.py`：加载微调后的权重，基于 box mask 做推理并生成精细分割

## 0. 如果不在 `foundad` 环境中

如果你是在其他机器或其他环境里使用这两个脚本，先单独创建一个新环境。这个最小流水线实际依赖的包只有：

- Python 3.9 或 3.10
- PyTorch
- torchvision
- numpy
- pillow
- opencv-python
- segment-anything

建议安装顺序是：先装 PyTorch，再装其他包。

### 0.1 新建环境

```bash
conda create -n sam_ft python=3.10 -y
conda activate sam_ft
```

### 0.2 安装 PyTorch

这一步要根据你的 CUDA 版本选择官方命令。比如 CUDA 11.8 可以用：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

如果你是 CPU 环境，或者 CUDA 版本不同，就去 PyTorch 官网换成对应命令，不要直接照抄。

### 0.3 安装其他依赖

```bash
pip install numpy pillow opencv-python
pip install git+https://github.com/facebookresearch/segment-anything.git
```

如果你本地已经有 `segment_anything` 源码，也可以改成：

```bash
pip install -e /绝对路径/segment-anything
```

### 0.4 自检命令

安装完后，先跑一个最小导入检查：

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

如果这一步能正常输出版本号，说明这个环境已经能支撑本目录下的两个脚本。

## 1. 现在的参数更精简了

常用情况下，只需要记住下面三类输入：

1. 数据根目录
2. 原始 SAM 预训练权重 `sam_vit_*.pth`
3. 微调后的 `best.pth`

另外，`sam_model_type` 现在可以根据权重文件名自动推断：

- 文件名里包含 `vit_b`，自动用 `vit_b`
- 文件名里包含 `vit_l`，自动用 `vit_l`
- 文件名里包含 `vit_h`，自动用 `vit_h`

如果你的权重文件名不规范，再手动传 `--sam_model_type`。

## 2. 微调数据格式

微调脚本要求数据组织成：

```text
train_root/
  img/
    xxx.jpg
    yyy.png
  seg_mask/
    xxx.png
    yyy.png
```
mask就是目标区域是白色，背景为黑（seg分割的）
要求：

- `img/` 和 `seg_mask/` 通过文件名 stem 配对
- mask 必须是二值图，非零即前景
- 图像和 mask 尺寸必须一致
- `seg_mask/` 必须是 `.png`

## 3. 最简微调命令

```bash
CUDA_VISIBLE_DEVICES=6 python tools/sam_ft_pipeline/sam_finetune.py \
  --train_root /data/my_dataset/paired_train \
  --sam_ckpt /data/checkpoints/sam_vit_b_01ec64.pth
```

这条命令默认会：

- 自动从 `sam_ckpt` 推断 `sam_model_type`
- 默认冻结 `image_encoder`
- 默认冻结 `prompt_encoder`
- 默认只训练 `mask_decoder`
- 默认输出到 `outputs/sam_ft/`

默认输出目录：

```text
outputs/sam_ft/
  ckpts/
    best.pth
    last.pth
  vis_train/
```

## 4. 常用微调参数

如果你需要更常见的控制项，通常只改这几个：

- `--epochs`
- `--batch_size`
- `--lr`
- `--output_root`

示例：

```bash
CUDA_VISIBLE_DEVICES=6 python tools/sam_ft_pipeline/sam_finetune.py \
  --train_root /data/my_dataset/paired_train \
  --sam_ckpt /data/checkpoints/sam_vit_b_01ec64.pth \
  --epochs 50 \
  --batch_size 2 \
  --lr 1e-4 \
  --output_root /data/outputs/sam_ft
```

## 5. 微调时模型实际做了什么

微调不是只用 box 做监督。当前流程是：

1. 用真实 `seg_mask` 作为监督真值
2. 从 `seg_mask` 在线提取 tight box
3. 从缺陷区域内部采样正点
4. 从 box 内但 GT 外部采样负点
5. 用 `box + 点 prompt` 让 SAM 预测 mask
6. 用 GT seg mask 做 `BCE + Dice` 监督

所以这是一种：

- 少量像素级标注监督
- prompt 驱动的轻量 SAM 领域适配

## 6. 推理数据格式

推理脚本要求目标数据是类似下面的结构：

```text
dataset_root/
  class_a/
    test/
      ko/
        a.jpg
        b.png
    ground_truth/
      ko/
        a.png
        b.png
```

含义：

- `test/ko/<stem>.*`：待推理图像
- `ground_truth/ko/<stem>.png`：对应的 box mask
- 输出文件：`ground_truth/ko/<stem>_seg.png`

原始 box mask 不会被覆盖。

## 7. 最简推理命令

```bash
CUDA_VISIBLE_DEVICES=6 python tools/sam_ft_pipeline/sam_infer_from_box.py \
  --dataset_root /data/my_dataset/weak_box_data \
  --ft_ckpt /data/outputs/sam_ft/ckpts/best.pth \
  --sam_ckpt /data/checkpoints/sam_vit_b_01ec64.pth
```

这里：

- `--dataset_root` 是 `--visa_root` 的通用别名
- `--ft_ckpt` 是 `--ckpt` 的更直观别名
- `sam_model_type` 默认自动推断

默认可视化输出目录：

```text
outputs/sam_ft/vis_gen/
```

## 8. 常用推理参数

通常只需要关心这几个：

- `--max_ratio`
- `--tighten_iters`
- `--neg_points`
- `--neg_margin_px`
- `--output_root`
- `--max_images_per_class`

示例：

```bash
CUDA_VISIBLE_DEVICES=6 python tools/sam_ft_pipeline/sam_infer_from_box.py \
  --dataset_root /data/my_dataset/weak_box_data \
  --ft_ckpt /data/outputs/sam_ft/ckpts/best.pth \
  --sam_ckpt /data/checkpoints/sam_vit_b_01ec64.pth \
  --max_ratio 0.85 \
  --tighten_iters 2 \
  --neg_points 4 \
  --neg_margin_px 8 \
  --output_root /data/outputs/sam_ft
```

## 9. 推理阶段的硬约束

生成的 `_seg.png` 会强制满足：

- 最终 mask 一定在输入 box mask 内
- 最终 mask 一定单独保存为 `_seg.png`
- 最终 mask 面积不超过 `max_ratio * area(box)`

如果第一次预测过大，脚本会自动：

1. 增强负提示重新收紧
2. 仍然过大时退化为 box 内概率 top-k 选择

## 10. 迁移到其他数据集时怎么做

### 情况 A：你有像素级缺陷 mask

直接整理成：

```text
train_root/img
train_root/seg_mask
```

然后运行 `sam_finetune.py`。

### 情况 B：你有 box mask，但没有精细 seg mask

把数据整理成：

```text
dataset_root/<class>/test/ko
dataset_root/<class>/ground_truth/ko
```

然后运行 `sam_infer_from_box.py`。

### 情况 C：你的目录结构不是这种形式

最稳妥的做法不是直接改核心逻辑，而是先把数据重组到脚本要求的目录结构，再运行。这样出错更少。

## 11. 建议的实际使用顺序

1. 先用少量像素级 mask 微调 SAM
2. 检查 `vis_train/` 里的训练可视化
3. 再对 box 数据做推理生成 `_seg.png`
4. 检查 `vis_gen/` 的结果是否过大或过小
5. 最后再批量跑全数据集

## 12. 对新数据集的建议

第一次迁移时建议：

- 先用 `vit_b`
- 先跑 smoke test：`--epochs 1 --batch_size 1`
- 推理时先加 `--max_images_per_class 5`
- 先看可视化，再决定是否全量运行

如果结果偏大：

- 减小 `--max_ratio`
- 增大 `--tighten_iters`
- 增大 `--neg_margin_px`

如果结果偏小：

- 适当增大 `--max_ratio`
- 适当减小 `--neg_margin_px`

## 13. 一句话总结

这套流程本质上就是：

- 用少量 `seg_mask` 对 SAM 做轻量微调
- 再用微调后的 SAM，把大量 box mask 自动转成更精细的 `_seg.png`

而且现在常用命令已经缩减到最核心的几个输入，拿去适配其他数据集会更直接。
