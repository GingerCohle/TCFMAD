# Per-Dimension Learned Fusion

## Overview

当前仓库的多层分支只保留一种融合方式：

- 单层模式：
  - `training.multi_layer_mean=false`
  - `testing.multi_layer_mean=false`
  - 行为与原始单层 `n_layer` 路径完全一致
- 多层模式：
  - `training.multi_layer_mean=true`
  - `testing.multi_layer_mean=true`
  - 只使用 per-dimension learned fusion

这里的配置名虽然仍然叫 `multi_layer_mean`，但开启后执行的不是 mean fusion，而是按维度学习的加权融合。

## Fusion Definition

对 3 层 token feature：

```text
h_layers = [h2, h3, h4]
z_layers = [z2, z3, z4]
```

分别学习两组参数：

```text
W_h in R^(3 x D)
W_z in R^(3 x D)
```

其中 `D` 是 embedding 维度。

对于每个维度 `d`，在 3 个层上做 softmax：

```text
alpha_h[:, d] = softmax(W_h[:, d])
alpha_z[:, d] = softmax(W_z[:, d])
```

最终融合结果是按 token index 对齐、按 embedding 维逐维加权：

```text
h_fused[b, p, d] = sum_l alpha_h[l, d] * h_layers[l][b, p, d]
z_fused[b, p, d] = sum_l alpha_z[l, d] * z_layers[l][b, p, d]
```

因此：

- 每个维度都有自己独立的 3 层权重
- 不再是全维共享的标量 3 权重
- 不再有 mean fusion / mean_std fusion / 旧 scalar fusion 分支

## Implementation

实现位于：

- `tcfmad/src/tcfmad.py`

核心模块是：

- `LayerDimWeightFusion`

模型中挂了两组融合器：

- `self.fuse_h`
- `self.fuse_z`

以及对应 helper：

- `resolve_feature_layers(...)`
- `multi_layer_features(...)`
- `target_multi_layer_features(...)`
- `fuse_h_layers(...)`
- `fuse_z_layers(...)`
- `fusion_alpha_means(...)`

## Initialization

`W_h` 和 `W_z` 都初始化为全 0：

```text
W_h = 0
W_z = 0
```

因此初始时每个维度上的 softmax 都是均匀分布：

```text
alpha_h[:, d] = [1/3, 1/3, 1/3]
alpha_z[:, d] = [1/3, 1/3, 1/3]
```

启动日志里打印的是按维度平均后的层权重：

```text
alpha_h_mean = mean_d alpha_h[:, d]
alpha_z_mean = mean_d alpha_z[:, d]
```

在零初始化下：

```text
alpha_h_mean = alpha_z_mean = [1/3, 1/3, 1/3]
```

## Fixed 3-Layer Rule

多层融合固定为 3 层。

默认层号：

```text
[n_layer - 1, n_layer, n_layer + 1]
```

默认 `n_layer=3` 时就是：

```text
[2, 3, 4]
```

如果手动设置 `training.layers` 或 `testing.layers`，代码会强制：

```text
len(layers) == 3
```

## Train Path

当 `training.multi_layer_mean=true` 时，训练主路径为：

1. 抽取 3 层 target feature：

```text
h_layers = [h2, h3, h4]
```

2. 抽取 3 层 context feature：

```text
z_layers = [z2, z3, z4]
```

3. 分别做 per-dimension fusion：

```text
h_fused, alpha_h = fuse_h_layers(h_layers)
z_fused, alpha_z = fuse_z_layers(z_layers)
```

4. predictor 只作用在 `z_fused` 上：

```text
p_fused = predictor(dropout(z_fused))
```

5. 主损失：

```text
L_main = loss_fn(h_fused, p_fused)
```

6. 一致性损失保持原逻辑：

```text
layer_preds = [predictor(dropout(z_l)) for z_l in z_layers]
L_cons = mean_l MSE(layer_preds[l], p_fused)
L_total = L_main + cons_lambda * L_cons
```

这里 `p_fused` 不做 `detach`。

训练启动时会打印一次：

```text
[LEARNED-FUSION][train] enabled layers=[2, 3, 4] alpha_h_mean=[...] alpha_z_mean=[...]
```

## Eval Path

当 `testing.multi_layer_mean=true` 时，eval / demo 逻辑为：

1. 抽取 3 层 encoder target feature：

```text
enc_layers = [h2, h3, h4]
```

2. 只对 `enc_layers` 使用 `fuse_h`：

```text
enc_fused, alpha_h = fuse_h_layers(enc_layers)
```

3. 后续评分流程保持不变，只是输入从单层 `enc` 换成 `enc_fused`：

```text
pred = predictor(enc_fused)
l = mse(enc_fused, pred)
top-k image score
pixel heatmap
```

eval 启动时会打印一次：

```text
learned fusion enabled | layers=[2, 3, 4] | alpha_h_mean=[...]
```

## Consistency Between Train and Eval

当前 train / eval 的一致性体现在表示层：

- train 主分支使用 `h_fused`
- eval 打分使用 `enc_fused`
- 两边都来自相同的 3 层抽取规则
- 两边都使用相同类型的 per-dimension fusion

因此不再存在：

- 训练一种融合
- 测试换另一种融合

## Checkpoint Compatibility

新的 checkpoint 会保存：

- `predictor`
- `projector`（如果存在）
- `fuse_h`
- `fuse_z`

其中 `fuse_h` / `fuse_z` 保存的是 per-dimension 参数矩阵。

### Migration Note

旧 checkpoint 可能出现两种情况：

1. 没有 `fuse_h` / `fuse_z`
2. 有旧融合字段，但 shape 与当前 `3 x D` 不兼容

当前 eval 侧会：

- 用 `strict=False` 尝试加载
- 对缺失或 shape 不兼容的融合参数跳过加载
- 保留默认零初始化
- 打印 warning 一次

因此旧 checkpoint 仍然可以运行，并自动回退到均匀初始化：

```text
alpha_h_mean = alpha_z_mean = [1/3, 1/3, 1/3]
```

## Config Semantics

当前只保留旧开关名，不引入新的 fusion mode 选择器。

语义如下：

```text
training.multi_layer_mean=false -> 原始单层路径
training.multi_layer_mean=true  -> 3 层 per-dim learned fusion

testing.multi_layer_mean=false  -> 原始单层路径
testing.multi_layer_mean=true   -> 3 层 per-dim learned fusion
```

## Commands

训练：

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

测试：

```bash
python tcfmad/main.py \
    mode=AD \
    data.dataset=visa \
    data.data_name=visa_tmp \
    data.test_root=/media/disk/kejunjie_only/dino_anomaly/TCFMAD/visa \
    diy_name=_2shot \
    app=test \
    app.ckpt_step=6000 \
    testing.multi_layer_mean=true \
    testing.layers='[2,3,4]'
```

范围测试：

```bash
DEVICE_LIST="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7" \
START_STEP=4000 \
END_STEP=6000 \
TEST_BATCH_SIZE=128 \
TEST_NUM_WORKERS=8 \
SEGMENTATION_VIS=false \
TESTING_MULTI_LAYER_MEAN=true \
TESTING_LAYERS='[2,3,4]' \
bash scripts/test_visa_tmp_2shot_range.sh
```
