# CutPaste 修改总结

本文总结当前仓库里和 `CutPaste` 相关的实际改动，重点说明：

- `CutPaste` 本体改了什么
- 训练入口怎么接入 `CutPaste`
- 新增了什么替代合成路径
- 默认行为是否变化

## 结论

当前仓库里，`CutPaste` 的核心算法没有被重写，`CutPasteNormal`、`CutPasteScar`、`CutPasteUnion` 仍然保留，默认训练仍然走 `CutPasteUnion`。

真正的变化主要有两类：

1. 训练侧从“固定只用 `CutPasteUnion`”改成了“通过 `synthesis.mode` 选择合成器”。
2. 新增了一条并行的异常合成路径 `segpatch_folder`，用于替代 `CutPaste` 生成 `imgs_abn`。

## 1. `CutPaste` 本体没有被改坏

文件：`tcfmad/src/utils/synthesis.py`

当前仍然保留以下类：

- `CutPaste`
- `CutPasteNormal`
- `CutPasteScar`
- `CutPasteUnion`

其中 `CutPasteUnion` 依然是：

- 50% 概率调用 `CutPasteNormal`
- 50% 概率调用 `CutPasteScar`

对应代码位置：

- `CutPasteNormal`: `tcfmad/src/utils/synthesis.py:112`
- `CutPasteScar`: `tcfmad/src/utils/synthesis.py:173`
- `CutPasteUnion`: `tcfmad/src/utils/synthesis.py:259`

也就是说，原来的 `CutPaste` 逻辑没有被删除，也没有被替换成别的实现。

## 2. 训练入口的改动

文件：`tcfmad/src/train.py`

以前训练初始化时是固定写死：

- `self.cutpaste = CutPasteUnion(...)`

现在改成：

- 从配置读取 `synthesis.mode`
- 当 `synthesis.mode=cutpaste` 时，使用 `CutPasteUnion`
- 当 `synthesis.mode=segpatch_folder` 时，使用 `SegPatchFolderPaste`

关键代码位置：

- `tcfmad/src/train.py:99-117`

当前逻辑等价于：

```python
synthesis_cfg = args.get("synthesis", {})
self.synthesis_mode = synthesis_cfg.get("mode", "cutpaste")

if self.synthesis_mode == "cutpaste":
    self.synth = CutPasteUnion(colorJitter=0.5)
elif self.synthesis_mode == "segpatch_folder":
    self.synth = SegPatchFolderPaste(args, train_image_root, self.synthesis_seed)
```

这说明：

- `CutPaste` 不再是唯一合成方式
- 但它仍然是默认方式

## 3. 训练时生成 `imgs_abn` 的接口被统一了

文件：`tcfmad/src/train.py`

新增了一个统一 helper：

- `Trainer._make_abnormal_images(...)`

对应位置：

- `tcfmad/src/train.py:183-188`

逻辑是：

- 如果当前模式是 `segpatch_folder`，就调用 `self.synth(imgs, paths)`
- 否则调用 `self.synth(imgs, labels)`

原因是两种合成器的输入不同：

- `CutPasteUnion` 需要类名 `labels`
- `SegPatchFolderPaste` 需要图像路径 `paths`，用来定位同类 `seg_patch/`

训练循环里不再直接写死 `self.cutpaste(...)`，而是统一走 `_make_abnormal_images(...)`。

对应位置：

- `tcfmad/src/train.py:277`
- `tcfmad/src/train.py:341`

## 4. 新增了 `segpatch_folder`，但它不是对 `CutPaste` 的内部修改

文件：`tcfmad/src/utils/synthesis.py`

新增类：

- `SegPatchFolderPaste`

起始位置：

- `tcfmad/src/utils/synthesis.py:280`

这个类的行为是：

- 从 `train/<class>/seg_patch/` 中读取 patch
- 每张正常图贴 1 个同类 patch
- 支持缩放、随机位置、feather alpha blending、可选 color match

要点是：

- 它是新增的一条替代合成路径
- 不是把 `CutPasteUnion` 改成了另一套算法

因此，从代码结构上看，现在是：

- `cutpaste` 仍然存在
- `segpatch_folder` 是新加的并行选项

## 5. 新增的配置项

文件：`tcfmad/configs/config.yaml`

新增配置块：

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

对应位置：

- `tcfmad/configs/config.yaml:79-87`

其中最关键的是：

- `synthesis.mode: cutpaste`

这说明默认行为仍然是 `CutPasteUnion`。

也就是说，如果训练命令里不额外指定：

```bash
synthesis.mode=segpatch_folder
```

那么训练仍然会走老的 `CutPaste` 路径。

## 6. 什么没有变

以下行为在 `synthesis.mode=cutpaste` 时保持原样：

- target 分支仍然输入干净图 `imgs`
- context 分支仍然是 50% 干净图、50% 异常图
- 多层 learned fusion / consistency loss 与合成方式解耦
- `imgs_abn` 仍然只是训练时的上下文异常输入

也就是说，当前对 `CutPaste` 的改动主要是“外层接线方式”发生了变化，而不是“`CutPaste` 算法本身”被重写。

## 7. 简单结论

如果只看 `CutPaste`：

- `CutPasteNormal` 没被改
- `CutPasteScar` 没被改
- `CutPasteUnion` 没被改

真正的工程改动是：

- 增加了 `synthesis.mode`
- 用统一的 `self.synth` / `_make_abnormal_images(...)` 接口替代原来写死的 `CutPasteUnion`
- 新增了 `segpatch_folder` 作为可选异常合成器

因此，当前仓库不是“修改了 `CutPaste` 算法”，而是“把训练侧的异常合成器做成了可切换架构，默认仍是 `CutPaste`”。
