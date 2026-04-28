# SegPatch Folder Synthesis

## Overview

新增了一个默认关闭的训练时异常合成模式：

- `synthesis.mode=cutpaste`
  - 原有 `CutPasteUnion`
- `synthesis.mode=segpatch_folder`
  - 从同类的 `seg_patch/` 文件夹里取缺陷 patch
  - 每张 OK 图只粘贴 1 个 patch
  - 使用 feathered alpha blending
  - 可选做简单的颜色统计匹配

这个改动只影响 `imgs_abn` 的生成方式。
其余训练逻辑保持不变：

- target 分支始终使用干净图 `imgs`
- context 分支仍然是原来的 50/50 策略
  - `imgs`
  - `imgs_abn`
- 多层 learned fusion / consistency loss 完全不变

## Expected Folder Layout

few-shot 训练目录需要满足：

```text
<DATA_PATH>/<DATA_NAME>/train/<class>/
<DATA_PATH>/<DATA_NAME>/train/<class>/seg_patch/*.png
```

例如：

```text
visa_tmp/train/candle/
visa_tmp/train/candle/seg_patch/000.png
```

说明：

- `<class>/` 下原有 OK 训练图保持不动
- `seg_patch/` 下放矩形 defect patch
- patch 会从当前样本所属类别的 `seg_patch/` 中采样

## Config

默认值在 `tcfmad/configs/config.yaml`：

```yaml
synthesis:
  mode: cutpaste
  seed: 0
  area_ratio: [0.005, 0.05]
  feather_px: 2
  color_match: true
  max_tries: 20
  seg_patch_dirname: seg_patch
  paste_k: 1
```

含义：

- `synthesis.mode`
  - `cutpaste` 或 `segpatch_folder`
- `synthesis.seed`
  - 控制 patch 采样、缩放和粘贴位置
- `synthesis.area_ratio`
  - patch 面积相对整张图面积的比例范围
- `synthesis.feather_px`
  - 矩形边缘 feather 宽度
- `synthesis.color_match`
  - 是否对 patch 和目标区域做简单通道统计匹配
- `synthesis.max_tries`
  - 尝试粘贴的位置采样次数
- `synthesis.seg_patch_dirname`
  - patch 子目录名
- `synthesis.paste_k`
  - 当前固定要求 `1`

## Synthesis Logic

对每张 OK 图：

1. 从同类 `seg_patch/` 中随机采样 1 个 patch
2. 用与训练图一致的 ImageNet mean/std 做归一化
3. 按 `area_ratio` 缩放 patch
4. 在图上随机采样粘贴位置
5. 如果开启 `color_match`，对 patch 和目标区域做简单通道 mean/std 匹配
6. 构造 feather alpha mask
7. 用 alpha blending 生成 `imgs_abn`

## How Class Is Resolved

训练图路径形如：

```text
.../<data_name>/train/<class>/<filename>
```

代码会优先按 `train_root/train` 的相对路径提取 `<class>`，
如果相对路径解析失败，则回退到父目录名。

## How To Enable

在你现有训练命令上只加：

```bash
synthesis.mode=segpatch_folder \
synthesis.seed=0
```

完整示例：

```bash
python tcfmad/main.py \
  mode=train \
  app=train_dinov3 \
  data.dataset=visa \
  data.data_name=visa_tmp \
  data.data_path=/media/disk/kejunjie_only/dino_anomaly/compare_visa/TCFMAD_visapcb \
  diy_name=_2shot_04 \
  training.multi_layer_mean=true \
  training.consistency_loss=true \
  training.cons_lambda=0.4 \
  synthesis.mode=segpatch_folder \
  synthesis.seed=0 \
  dist.master_port=30777 \
  devices=[cuda:0]
```

## Smoke Run

短跑可以直接用：

```bash
python tcfmad/main.py \
  mode=train \
  app=train_dinov3 \
  data.dataset=visa \
  data.data_name=visa_tmp \
  data.data_path=/media/disk/kejunjie_only/dino_anomaly/compare_visa/TCFMAD_visapcb \
  diy_name=_2shot_04 \
  training.multi_layer_mean=true \
  training.consistency_loss=true \
  training.cons_lambda=0.4 \
  synthesis.mode=segpatch_folder \
  synthesis.seed=0 \
  trainer.max_steps=5 \
  dist.master_port=30777 \
  devices=[cuda:0]
```

## Notes

- `segpatch_folder` 默认关闭，不影响旧训练命令
- 不会修改或覆盖任何原始 OK 训练图
- `paste_k` 当前固定要求为 `1`
- 如果某个类没有 `seg_patch/`，该类样本会回退为不粘贴
