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
- 默认全参数微调，`epochs: 1`、batch size 8、学习率 1e-5、AdamW、梯度范数上限 1.0、seed 0、FP32。
- 使用 Lightning 2.4 管理训练循环、自动优化、梯度裁剪、进度条和 DDP。DataLoader 每轮打乱、无放回抽取窗口，`drop_last=True`，丢弃不足一个 batch 的尾部样本。
- 多卡使用 `DistributedSampler(drop_last=True)`，不通过重复样本补齐各卡；Lightning 每轮更新 sampler 的 epoch。只有 rank 0 保存最终模型和报告。
- 不另设验证集，不早停，不用评测区间选择检查点。超参数是起点，未作本地训练调优。

CLI 可覆盖 `--epochs 3 --batch-size 4 --lr 0.000005 --output runs/another-run`。旧 `steps` 配置和 `--steps` 参数不再接受。

默认窗口设置共生成 41,893,260 个训练样本；单卡 batch size 8 时，每轮有 5,236,657 个完整 batch，丢弃 4 个尾部样本。`--dry-run` 显示单卡批次数作为参考，不启动 GPU 或 DDP。多卡每轮每卡批次数为 `floor(floor(样本数 / 卡数) / batch_size)`；batch size 是每卡大小。

`training.device: cuda`、`devices: auto`、`strategy: auto` 默认使用所有可见 GPU，由 Lightning 选择单卡或 DDP。可设置 `devices: 2`、`strategy: ddp`，或通过 `--devices 2 --strategy ddp` 覆盖。CPU 使用 `device: cpu`。仍使用本地改编的 Chronos-2 主干，而非官方 Pipeline.fit。

训练损失保持为归一化目标上的分位数损失：`2 * max(q * error, (q - 1) * error)`，对所有预测分位数求和，再除以有效目标点数。目标复用历史窗口的归一化统计量，并按模型配置进行 arcsinh 变换。缺失标签不计入损失。MAE、1-WAPE、WRMSSE 仅用于评测。

`--dry-run` 在 CPU 加载模型并检查窗口，不执行前向、反向、优化器或模型保存；它不验证 GPU 显存需求。

## 训练后评测

```bash
python run.py evaluate --checkpoint runs/m5-small-target-train --output runs/m5-small-target-eval
```

这与零样本模型共用相同的 28 天留出区间、q0.5 点预测以及三个评分指标。已有输出目录不会被覆盖。评测推理不丢弃尾部批次，默认 `configs/baseline.yaml` 中 `batch_size: 80`，为默认训练批次 8 的十倍；两者独立配置，可用 `evaluate --batch-size 80` 覆盖，并按云端显存调整。

## 产物

训练输出为 `model.pt`、`config.json`、`resolved_config.yaml`、`run.json`，保存在 `runs/m5-small-target-train/`，不上传 Git。

训练器仅在结束时保存，不支持中断续训。Lightning 进度条显示训练损失，最终聚合损失、卡数、每轮批次数及总更新次数在成功结束后写入 `run.json`，不在内存中累积数百万条逐步 loss。
