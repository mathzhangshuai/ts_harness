# tszoo

支持通过 YAML 按字段名选择七类时序变量，默认仍为仅销量输入。参见[变量配置](docs/m5-features.md)。

M5 预测：Chronos-2-small、Chronos-2 仅目标变量零样本基线，以及支持可选协变量的 Small 全参数微调。没有 `src/` 嵌套或通用数据集框架。

## 目录

```text
tszoo/
  run.py                   命令入口
  config.py                配置解析与校验
  configs/                 baseline.yaml、train_small.yaml
  data/                    M5 下载、准备、窗口与 manifest.json
  models/chronos2/         主干、目标模型、网络层、许可证
  training/trainer.py      全参数微调与 dry-run
  evaluation/              预测、MAE/1-WAPE、WRMSSE、评测流程
  tests/                   无参数更新的回归测试
  docs/                    运行说明与当前基线
  datasets/                本地原始及预处理数据
  .model-cache/            本地预训练权重
  runs/                    本地训练与评测产物
```

## 使用

Python 3.11；按 [云端环境说明](docs/github-workflow.md) 安装依赖。在本目录运行：

```bash
python run.py download
python run.py prepare
python run.py train --dry-run
python run.py train
python run.py evaluate --output runs/m5-baseline-rerun
```

下载默认使用 `configs/baseline.yaml`，可加 `--models chronos2_small` 只下载 Small。准备命令默认生成训练、评测两个独立存储；已有完整存储复用，不覆盖。

训练默认使用 `configs/train_small.yaml`：只输入销量，512 天历史、28 天标签，训练截止 d_1913。执行训练前请阅读 [训练流程](docs/m5-small-training.md)。

评测原始权重与微调模型使用同一个入口：

```bash
python run.py evaluate --checkpoint runs/m5-small-target-train --output runs/m5-small-target-eval
```

只输出 1-WAPE、MAE、WRMSSE。评测输出目录不能已存在。`baseline` 保留为 `evaluate` 的命令别名；独立 `wrmsse` 命令仍能读取既有预测产物。

## 兼容与验证

- 原始权重、M5 文件和当前基线结果保留原位置。
- 旧的仅目标变量 memmap 和 `model.pt` 检查点可继续读取。
- 多数据集、协变量 CLI、旧 `config/` 路径和 `utils/` 导入不再支持；使用本页新入口。
- `python run.py test` 只执行数据、配置、推理、评分和 dry-run 检查，不执行优化器更新。
- 可选的原版数值对照测试使用外层参考源码；没有参考源码时跳过该项。

[当前基线](docs/m5-zero-shot.md) · [指标口径](docs/m5-wrmsse.md) · [术语](CONTEXT.md)
