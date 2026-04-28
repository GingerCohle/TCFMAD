# TCFMAD 变更审查总结

## 背景

- 用户要求对当前变更做一次 change review。
- 指定的自定义 skill `code_review_change` 不在本次会话可用列表中，因此采用标准代码审查流程执行。
- 顶层目录 `/media/disk/kejunjie_only/dino_anomaly` 不是 Git 仓库，实际 Git 仓库位于 `TCFMAD/`。

## 已完成工作

### 1. 定位审查对象

- 确认实际仓库路径为 `TCFMAD/`。
- 检查当前变更范围，识别到以下已修改文件：
  - `tcfmad/configs/config.yaml`
  - `tcfmad/main.py`
  - `tcfmad/src/AD.py`
  - `tcfmad/src/train.py`

### 2. 阅读并分析变更

- 查看了上述 4 个文件的 diff。
- 为确认行为影响范围，继续阅读了以下相关代码：
  - `tcfmad/src/tcfmad.py`
  - `tcfmad/src/datasets/dataset.py`
  - `tcfmad/src/utils/metrics.py`
  - `tcfmad/src/utils/logging.py`
  - `tcfmad/src/helper.py`
  - `tcfmad/src/utils/synthesis.py`

### 3. 基础验证

- 执行了语法编译检查：

```bash
python -m py_compile tcfmad/main.py tcfmad/src/AD.py tcfmad/src/train.py
```

- 执行了 diff 格式检查：

```bash
git diff --check
```

- 两项检查均通过。

### 4. 额外验证

- 为确认评估 profiling 路径的失败模式，直接验证了 `compute_imagewise_retrieval_metrics(...)` 在只包含单一类别标签时的行为。
- 结果确认：当 `y_true` 只有一个类别时，`roc_auc_score` 会抛出 `ValueError`。

## 主要结论

### Finding 1: 评估 profiling 路径可能报错或输出失真

严重性：High

涉及文件：

- `tcfmad/src/AD.py`
- `tcfmad/src/utils/metrics.py`
- `tcfmad/src/datasets/dataset.py`

问题说明：

- `AD.py` 新增了 profiling 模式，并允许通过 `profile_max_images` 提前截断测试样本。
- 但是截断后，代码仍然继续按完整评估流程计算 AUROC / AUPR / PRO。
- 当前 metrics 实现要求标签中同时存在正样本和负样本，否则 `roc_auc_score` 不可定义。
- `TestDataset` 的测试集遍历顺序按 anomaly 名字排序，`mvtec` 下通常 `"good"` 会排在前面，`visa` 下通常 `"ok"` 会排在前面。
- 因此当 `profile_max_images` 较小，profiling 很容易只采到正常样本，随后在图像级指标计算时报错，或者得到没有代表性的部分指标。

结论：

- profiling 模式下当前评估逻辑不稳健，存在真实运行失败风险。

### Finding 2: 默认非 profiling 训练路径引入无意义的内存增长

严重性：Medium

涉及文件：

- `tcfmad/src/train.py`

问题说明：

- `train.py` 中新增了 `step_times_ms` 列表，并在每个 step 后无条件执行 `step_times_ms.append(t)`。
- 但该列表只在 `self.profile` 为 `true` 的汇总分支中被使用。
- 当前默认配置 `profile: false`，因此正常长时训练也会持续累积这个列表，却不会消费它。
- 这会带来线性增长的 host memory 占用，尤其在大量 step 的训练场景下没有必要。

结论：

- 该问题不会立刻打断训练，但会在默认路径上引入不必要的资源消耗。

## 当前没有做的事情

- 还没有修改代码。
- 还没有提交修复补丁。
- 还没有执行完整训练或完整评估回归。
- 没有发现现成的自动化测试覆盖这次改动路径。

## 建议的下一步

1. 修复 `AD.py` 中 profiling 截断后的指标计算逻辑。
2. 修复 `train.py` 中 `step_times_ms` 在 `profile=false` 下的无意义累积。
3. 如果需要保留 profiling 指标，建议增加最小可评估样本约束，或者在 profiling 模式下仅输出性能统计、不输出不完整的质量指标。

## 可直接转述给 ChatGPT 的一句话总结

这次 review 已完成静态审查和基础验证，确认了 2 个主要问题：一是 `AD.py` 的 profiling 评估会因为截断测试集而导致 AUROC/AUPR 计算报错或失真；二是 `train.py` 在默认非 profiling 模式下仍持续累积 step timing，带来不必要的内存增长。目前还没有开始修复代码。
