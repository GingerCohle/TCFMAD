# 当前 Paste 预处理总结

本文总结当前仓库里 `synthesis.mode=segpatch_folder` 时，训练阶段如何对 defect patch 做预处理并粘贴到正常图上。

## 入口

训练初始化位置：`tcfmad/src/train.py:99-117`

当前逻辑是：

- `synthesis.mode=cutpaste` 时，使用 `CutPasteUnion`
- `synthesis.mode=segpatch_folder` 时，使用 `SegPatchFolderPaste`

对应代码：

```python
synthesis_cfg = args.get("synthesis", {})
self.synthesis_mode = synthesis_cfg.get("mode", "cutpaste")

if self.synthesis_mode == "cutpaste":
    self.synth = CutPasteUnion(colorJitter=0.5)
elif self.synthesis_mode == "segpatch_folder":
    train_image_root = Path(dcfg["train_root"]) / "train"
    self.synth = SegPatchFolderPaste(args, train_image_root, self.synthesis_seed)
```

训练时生成异常图的位置：`tcfmad/src/train.py:183-188`

```python
if self.synthesis_mode == "segpatch_folder":
    _, imgs_abn = self.synth(imgs, paths)
else:
    _, imgs_abn = self.synth(imgs, labels)
```

也就是说，`segpatch_folder` 生成的是训练用的 `imgs_abn`，不会改动原始 `imgs`。

## 当前 paste 总流程

实现类：`tcfmad/src/utils/synthesis.py:280`

对每张正常图 `img_ok`，当前流程是：

1. 根据训练图路径推断类别名
2. 从同类 `train/<class>/seg_patch/` 中随机选 1 个 patch
3. 读取 patch，并转换到与训练图一致的归一化空间
4. 按目标面积比例缩放 patch
5. 在整张图上随机采样一个粘贴位置
6. 可选做 patch 和目标区域的通道 mean/std 匹配
7. 生成矩形 feather alpha mask
8. 用 alpha blending 得到 `img_abn`

## 1. patch 索引

代码位置：`tcfmad/src/utils/synthesis.py:298-312`

当前会在：

- `train/<class>/seg_patch/*.png`
- `train/<class>/seg_patch/*.jpg`
- `train/<class>/seg_patch/*.jpeg`

里建立索引，形成：

- `class_name -> patch_path_list`

注意：

- 只会从“同类”的 `seg_patch/` 里取 patch
- 如果某个类没有 patch，则该样本直接回退为不粘贴

## 2. 类别解析

代码位置：`tcfmad/src/utils/synthesis.py:320-330`

当前优先从训练图相对 `train_root` 的路径解析类别：

- `.../train/<class>/<image>` -> `<class>`

如果相对路径解析失败，则回退到父目录名。

## 3. patch 读取与归一化

代码位置：`tcfmad/src/utils/synthesis.py:332-339`

当前实现：

```python
patch = Image.open(patch_path).convert("RGB")
cached = self.normalize(TF.to_tensor(patch))
```

这意味着：

- patch 会被强制转成 `RGB`
- 当前不会读取 PNG alpha 通道
- patch 会先 `ToTensor()`，再按 ImageNet mean/std 归一化
- patch 缓存在 `patch_cache` 里，避免重复解码

所以，当前 patch 和训练图是在同一个“归一化后”的数值空间里做后续处理。

## 4. patch 缩放

代码位置：`tcfmad/src/utils/synthesis.py:341-361`

当前缩放逻辑：

- 从 `synthesis.area_ratio=[min,max]` 里随机采样一个目标面积比例
- 目标面积 = `ratio * H * W`
- 根据原 patch 宽高比计算缩放系数
- 使用 bilinear resize 得到新 patch

关键点：

- 默认面积范围是 `[0.005, 0.05]`
- 这是相对整张训练图面积的比例
- patch 只做缩放，不做旋转

## 5. 颜色匹配

代码位置：`tcfmad/src/utils/synthesis.py:363-368`

如果 `synthesis.color_match=true`，当前会对 patch 和目标区域做简单通道统计匹配：

```python
(patch - patch_mean) * (region_std / patch_std) + region_mean
```

也就是：

- 逐通道减去 patch 自身均值
- 按目标区域标准差重新缩放
- 再加上目标区域均值

注意：

- 这是在归一化后的张量空间里做的
- 当前不是更复杂的颜色迁移，只是 mean/std 对齐

## 6. alpha mask

代码位置：`tcfmad/src/utils/synthesis.py:370-387`

当前 alpha 是矩形软边，不是基于缺陷真实 mask：

- 先生成全 1 的矩形 alpha
- 在四周边缘按 `feather_px` 做线性衰减
- 最终通过 `outer(alpha_y, alpha_x)` 得到 2D alpha

这意味着：

- 当前贴的是“整个矩形 patch”
- 即使 patch PNG 原本带透明背景，也不会被单独利用
- feather 只是在矩形边框处做平滑

## 7. paste 与 blending

代码位置：`tcfmad/src/utils/synthesis.py:389-420`

当前每张图只粘贴 1 个 patch：

- `paste_k` 固定要求为 `1`

当前粘贴位置采样方式：

- 在整张图范围内随机选 `top,left`
- 最多尝试 `max_tries` 次
- 目前没有前景 mask 约束

最终融合公式：

```python
img_abn[:, top:top+patch_h, left:left+patch_w] = region * (1 - alpha) + patch_to_paste * alpha
```

也就是标准 alpha blending。

## 8. 训练里如何使用 paste 结果

训练循环里：

- target 分支始终用干净图 `imgs`
- context 分支仍是原来的 50/50 策略
  - 一半概率用 `imgs`
  - 一半概率用 `imgs_abn`

因此当前 `paste` 预处理只影响：

- `imgs_abn` 的生成方式

不会影响：

- 原始正常图读取
- target 分支
- learned fusion
- consistency loss

## 当前实现的几个关键特征

现在这套 paste 预处理有 4 个很重要的特点：

1. patch 在归一化空间里参与缩放、颜色匹配和融合
2. patch 来源于同类 `seg_patch/` 文件夹
3. 当前贴的是矩形 patch，不是 defect mask 形状
4. 当前粘贴位置是全图随机，不受前景区域约束

## 当前实现的局限

从代码上看，当前版本有几个明显限制：

- 不读取 PNG alpha，透明背景信息会丢失
- 不做真实 mask paste，只做矩形 paste
- 不做前景约束，patch 可能贴到背景
- 默认 `area_ratio` 偏小，异常可能比较弱
- `color_match + feather` 可能进一步削弱异常边界

## 简单结论

当前的 paste 预处理不是“直接把 patch 生硬盖上去”，而是：

- 先同类采样
- 再归一化
- 再按面积缩放
- 再可选做 mean/std 匹配
- 再用矩形软边 alpha 混合

但它目前仍然是一个比较简化的 paste 实现，核心限制是：

- 没有利用 patch 的真实前景/alpha mask
- 没有前景区域约束
- 仍然是矩形 paste
