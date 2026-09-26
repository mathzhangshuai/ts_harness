# Chronos-2-small 仅目标变量微调

本流程在云端运行。本地仅检查代码和数据边界，不执行训练。

## 云端命令

在仓库根目录拉取更新，然后进入 `tszoo/`。使用 Python 3.11，并按 [环境说明](github-workflow.md) 安装适配 GPU 的依赖。

```bash
git pull --ff-only
cd tszoo

python run.py download --config config/m5-baseline.yaml --models chronos2_small
python run.py prepare --dataset m5 --source datasets/m5 --output datasets/processed/m5-train-target --m5-split validation --target-only
python run.py train --config config/m5-small-target-train.yaml --dry-run
python run.py train --config config/m5-small-target-train.yaml
```

数据准备和训练输出目录均不能已存在；准备过同一份数据后跳过 `prepare`。再次训练请用 `--output runs/新的目录名`。不覆盖旧检查点。

## 训练约定

- 只从 `sales_train_validation.csv` 读取销量，训练存储最多到 `d_1913`；不读取 evaluation 销量表，也不生成商品、门店、价格、日历等协变量。
- `item_id` 和起始日期仅为数据索引元信息，不编码为模型输入。
- `fields: [target]`；每个训练窗口输入 512 天，标签 28 天，所有标签均不晚于 `d_1913`。`d_1914–d_1941` 继续保留为评测区间。
- 使用固定版本的 `autogluon/chronos-2-small` 原始权重，全参数微调：`freeze_backbone: false`。不新增特征适配器。
- 默认 1,000 个优化步，batch size 8，学习率 1e-5，AdamW，梯度范数上限 1.0，随机种子 0，FP32。参数是可调整的起点，未经本地训练调优，不承诺最优效果或显存需求。
- 从所有合法历史窗口中有放回抽样，`steps` 是优化步数，不是 epoch。使用原模型的分位数训练损失；最终评测口径仍为 q0.5 的 1-WAPE、MAE、WRMSSE。
- 沿用已接受的数据切分，不另设验证集、不做早停、不根据评测区间选检查点；保存最后一步模型。

可通过 CLI 覆盖，例如 `--batch-size 4 --steps 2000 --lr 0.000005`。

`--dry-run` 在 CPU 加载权重、检查窗口并打印字段、样本数、可训练参数量及超参数；不进行前向、反向、优化器更新或保存模型，也不验证 GPU 训练显存。

## 输出

`runs/m5-small-target-train/` 中包含 `model.pt`、`config.json`、`resolved_config.yaml` 和 `run.json`。运行时逐步输出 loss；`run.json` 在训练成功结束后保存完整 loss 列表。

这是项目的 `SplitChronos2` 检查点格式，通过 `SplitChronos2.from_local(...)` 加载，不是 Hugging Face 原始 safetensors 格式。当前通用零样本入口不能直接接收此检查点；训练后评测需使用该加载接口，具体评测入口后续衔接。

当前训练器只在结束时保存，不支持中断续训；请保留云端日志并确保任务持续运行。输出由 Git 忽略，不上传数据或权重。
