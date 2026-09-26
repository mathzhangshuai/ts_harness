# M5 7 天零样本基线

当前仅比较 Chronos-2-small 与 Chronos-2，报告 1-WAPE、MAE 和 WRMSSE。

在 `tszoo/` 下执行 `python run.py evaluate`，配置为 [baseline.yaml](../configs/baseline.yaml)。`baseline` 是同一命令的别名。输出目录必须不存在。

- 全量 30,490 条商品与门店序列；只输入销量，不训练、不微调。
- 输入 d_1865–d_1913 共 49 天，预测 d_1914–d_1920 共 7 天，每模型 213,430 个预测点。
- 点预测取 q0.5，不裁剪、不取整。模型使用配置固定的 revision 和 SHA-256。
- `1-WAPE = 1 - sum(abs(prediction - target)) / sum(abs(target))`，越高越好，可为负数。
- `MAE = sum(abs(prediction - target)) / 213430`，越低越好。
- WRMSSE 使用 [12 层评分口径](m5-wrmsse.md)，越低越好。
- 零样本推理保持 FP32；训练的混合精度配置不影响本基线。

结果来自 `runs/m5-baseline/metrics.json`，预测、标签、序列和配置位于 `predictions/`，评分元数据位于 `scoring/`。运行产物不上传 GitHub。

## 当前结果

2026-09-26 在本地完成全量推理，每个模型 213,430 个预测点。

| 模型 | 1-WAPE（越高越好） | MAE（越低越好） | WRMSSE（越低越好） |
| --- | ---: | ---: | ---: |
| Chronos-2-small | 31.86083508% | 0.92525343 | 1.54565487 |
| Chronos-2 | 31.77862901% | 0.92636970 | 1.48094918 |

该窗口曾用于历史实验探索。本轮只重新推理，不进行模型选择或训练。
