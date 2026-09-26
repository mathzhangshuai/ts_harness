# M5 零样本基线

此前实验输出已按用户要求清理。当前基线仅比较 Chronos-2-small 与 Chronos-2，报告 1-WAPE、MAE 和 WRMSSE。

在 `tszoo/` 下执行 `python run.py baseline`，配置为 [m5-baseline.yaml](../config/m5-baseline.yaml)。输出目录必须不存在，防止覆盖已有结果。

- 全量 30,490 条商品与门店序列；只输入销量历史，不训练、不微调。
- 输入为 d_1402 至 d_1913，共 512 天；标签为 d_1914 至 d_1941，共 28 天。
- 点预测取 q0.5，不裁剪、不取整。两个模型均使用配置固定的 revision 和 SHA-256。
- `1-WAPE = 1 - sum(abs(prediction - target)) / sum(abs(target))`，全量汇总，越高越好；不表示分类准确率，可为负数。
- `MAE = sum(abs(prediction - target)) / 853720`，越低越好。
- WRMSSE 使用 [12 层评分口径](m5-wrmsse.md)，越低越好。

结果来自本地 `runs/m5-baseline/metrics.json`，预测、标签、行号与实际配置保存在同目录的 `predictions/`，评分口径元数据保存在 `scoring/`。这些运行产物不上传 GitHub。

## 当前结果

2026-09-26 重新执行全量推理，853,720 个预测点/模型；只保留本次双模型基线。

| 模型 | 1-WAPE（越高越好） | MAE（越低越好） | WRMSSE（越低越好） |
| --- | ---: | ---: | ---: |
| Chronos-2-small | 32.90627605% | 0.96804144 | 1.98955761 |
| Chronos-2 | 33.15672702% | 0.96442788 | 1.91238462 |

## 季节朴素预测的代码

通用评测入口在 `utils/evaluate_m5.py` 使用 `np.resize(window["target"][0, -7:].numpy(), horizon)`：将最后 7 天销量按原顺序循环复制，28 天就是重复 4 次。比如历史末周为 `[1, 2, 3, 4, 5, 6, 7]`，未来四周均预测为这组值。它不训练、不求均值，也不读取未来标签。

当前 `baseline` 入口传入 `point_only=True`，不会生成或评分季节朴素预测；仅保留两个指定模型的结果。

该窗口曾用于历史实验探索；重新推理不改变这一事实。本轮不进行模型选择或训练。
