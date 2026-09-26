# Chronos-2-small 微调

本文命令默认仅使用销量；可在 `data.features` 中按字段名启用协变量，详见[七类变量配置](m5-features.md)。启用后请使用新的输出目录，并在评测配置中选择相同字段。

以下训练命令在云端运行。本轮本地仅验证数据、推理和 dry-run，不执行训练。

## 云端命令

安装适配 GPU 的 Python 3.11 依赖后，在仓库根目录执行：

```bash
git pull --ff-only
cd tszoo
python run.py download --models chronos2_small
python run.py prepare
python run.py train --dry-run
python run.py train
```

`prepare` 默认创建训练和评测存储，也可用 `--split train` 或 `--split evaluation` 单独准备。重复运行复用完整存储，不覆盖。

## 配置

配置位于 [train_small.yaml](../configs/train_small.yaml)，相对路径按 YAML 所在目录解析；CLI 路径按当前工作目录解析。

- 原始训练数据为 `sales_train_validation.csv`，只保存 d_1–d_1913 的销量。
- 每个窗口输入 512 天，标签 28 天；训练标签不进入 d_1914–d_1941。
- 默认不启用商品、门店、价格、节日等协变量；只有在 `data.features` 中选择的字段才进入模型。
- Small 原始权重来自下载配置固定的 revision 和 SHA-256。
- 默认全参数微调，1,000 步、batch size 8、学习率 1e-5、AdamW、梯度范数上限 1.0、seed 0、FP32。
- 从合法窗口中有放回抽样；优化原模型的分位数损失，保存最后一步模型。
- 不另设验证集，不早停，不用评测区间选择检查点。超参数是起点，未作本地训练调优。

CLI 可覆盖 `--steps 2000 --batch-size 4 --lr 0.000005 --output runs/another-run`。

`--dry-run` 在 CPU 加载模型并检查窗口，不执行前向、反向、优化器或模型保存；它不验证 GPU 显存需求。

## 训练后评测

```bash
python run.py evaluate --checkpoint runs/m5-small-target-train --output runs/m5-small-target-eval
```

这与零样本模型共用相同的 28 天留出区间、q0.5 点预测以及三个评分指标。已有输出目录不会被覆盖。

## 产物

训练输出为 `model.pt`、`config.json`、`resolved_config.yaml`、`run.json`，保存在 `runs/m5-small-target-train/`，不上传 Git。

训练器仅在结束时保存，不支持中断续训。逐步 loss 输出到终端，完整 loss 列表在成功结束后写入 `run.json`。
