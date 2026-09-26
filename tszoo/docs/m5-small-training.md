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
- `data.train_days: 180` 限制训练使用截止 `train_end: 1913` 的最后 180 天，即 d_1734–d_1913；每个窗口输入 `context: 49` 天，标签 `horizon: 7` 天。历史和标签均在这 180 天内，不进入留出期。复用原 memmap 文件，按索引限制读取范围，无需重新 prepare。
- 默认不启用商品、门店、价格、节日等协变量；只有在 `data.features` 中选择的字段才进入模型。
- Small 原始权重来自下载配置固定的 revision 和 SHA-256。
- 默认全参数微调，`epochs: 6`、每卡 batch size 256、每卡 12 个 DataLoader worker、学习率 1e-5、AdamW、梯度范数上限 1.0、seed 0。多卡全局 batch size 为 `256 * 卡数`，worker 总数为 `12 * 卡数`。
- `training.precision: bf16-mixed` 由 Lightning 管理混合精度；GPU 不支持 BF16 时使用 `--precision 16-mixed`，需关闭混合精度时使用 `--precision 32-true`。构造 Trainer 前设置 `torch.set_float32_matmul_precision("medium")`；这是 FP32 矩阵乘法的内部精度设置，不等同于混合精度。GPU 的实际加速取决于硬件支持。
- 使用 Lightning 2.4 管理训练循环、自动优化、梯度裁剪、进度条和 DDP。DataLoader 每轮打乱、无放回抽取窗口，`drop_last=True`，丢弃不足一个 batch 的尾部样本。
- Lightning DataModule 提供 DataLoader，使用继承 `DistributedSampler` 的 `BlockShuffleSampler`；Lightning 识别并保留它，每轮更新 epoch。采样流按位置分给各卡，不重复补齐，各卡批次数一致。只有 rank 0 保存最终模型和报告。
- `training.shuffle_block_size: 65536` 控制索引块大小，也可用 `--shuffle-block-size` 覆盖。每轮打乱块顺序，再按需打乱当前块；默认 M5 仅保存 59 个块编号和至多 65,536 个 int64 窗口索引，约 513 KiB。Dataset 按索引读取当前窗口，DataLoader 仅组装当前批次及有限的 worker 预取批次。分块打乱不是全局均匀随机排列，相邻批次的数据混合程度较低；块越大，混合范围和索引内存越大。
- 不另设验证集，不早停，不用评测区间选择检查点。超参数是起点，未作本地训练调优。

CLI 可覆盖 `--epochs 3 --batch-size 4 --lr 0.000005 --output runs/another-run`。旧 `steps` 配置和 `--steps` 参数不再接受。

默认每条序列有 `180 - 49 - 7 + 1 = 125` 个窗口，共 3,811,250 个训练样本；单卡 batch size 256 时，每轮有 14,887 个完整 batch，丢弃 178 个尾部样本，6 轮共 89,322 步。`--dry-run` 显示单卡批次数作为参考，不启动 GPU 或 DDP。多卡每轮每卡批次数为 `floor(floor(样本数 / 卡数) / batch_size)`；batch size 是每卡大小。可用 `--train-days` 覆盖训练跨度，需满足 `context + horizon <= train_days <= train_end`。

`training.device: cuda`、`devices: auto`、`strategy: auto` 默认使用所有可见 GPU，由 Lightning 选择单卡或 DDP。可设置 `devices: 2`、`strategy: ddp`，或通过 `--devices 2 --strategy ddp` 覆盖。CPU 使用 `device: cpu`。仍使用本地改编的 Chronos-2 主干，而非官方 Pipeline.fit。

训练损失使用参考实现的逐点分位数公式：`2 * abs(error * (I[target <= prediction] - q))`，在当前 batch 中对所有有效目标点、所有分位数求和，再除以有效目标点数。补齐位置和缺失标签既不贡献损失，也不计入分母；单条完整 7 天标签的分母为 7，而不是补齐后的 16。全 batch 没有有效标签时报错，避免无监督信号的优化步骤。这是明确区别于参考实现的归约方式。目标复用历史窗口的归一化统计量，并沿用权重配置中的 arcsinh 变换。MAE、1-WAPE、WRMSSE 仅用于评测。

`model.attention` 配置变量注意力：`variate_attention: grouped` 为按组计算，`global_masked` 为完整矩阵加分组掩码；两者保持相同的组隔离语义。`variate_attention_policy: target_aware` 禁止协变量读取目标，`bidirectional` 允许双向读取。`variate_grouping: series` 隔离不同序列，`batch` 允许同批序列互相读取。默认启用 grouped、target_aware、series。保存并恢复这些设置；推理不会再强制改为双向。详见[参考实现核查](m5-chronos-audit.md)。

`--dry-run` 在 CPU 加载模型并检查窗口，不执行前向、反向、优化器或模型保存；它不验证 GPU 显存需求。

## 训练后评测

```bash
python run.py evaluate --checkpoint runs/m5-small-target-train --context 49 --horizon 7 --output runs/m5-small-target-eval-7d
```

训练后评测和零样本基线统一使用 d_1914–d_1920 的 7 天留出区间、49 天历史和 q0.5 点预测，输出三个评分指标。WRMSSE 使用 d_1–d_1913 的缩放历史和末 28 天销售额权重；销售额窗口与预测长度是不同概念。已有输出目录不会被覆盖。评测推理不丢弃尾部批次，默认 `batch_size: 80`；评测与训练批次独立配置，可用 `evaluate --batch-size` 覆盖，并按云端显存调整。

## 产物

训练输出为 `model.pt`、`config.json`、`resolved_config.yaml`、`run.json`，保存在 `runs/m5-small-target-train/`，不上传 Git。

训练器仅在结束时保存，不支持中断续训。Lightning 进度条显示训练损失，最终聚合损失、卡数、每轮批次数及总更新次数在成功结束后写入 `run.json`，不在内存中累积数百万条逐步 loss。
