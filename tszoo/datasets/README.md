# 本地 M5 数据

原始文件位于 `m5/`，不上传 Git。来源与 SHA-256 见 [下载清单](../data/manifest.json)。

| 文件 | 用途 |
| --- | --- |
| sales_train_validation.csv | d_1–d_1913，训练历史 |
| sales_train_evaluation.csv | d_1–d_1941，其中最后 28 天作评测标签 |
| calendar.csv | 日期索引；WRMSSE 的周价格对齐 |
| sell_prices.csv | WRMSSE 历史销售额权重，不作为模型输入 |
| sample_submission.csv | 官方提交模板，不是未来销量 |

两个销量表历史重叠，不拼接为独立样本。

`python run.py prepare` 在项目目录下生成：
- `datasets/processed/m5-train-target/`：只来自 validation 销量表。
- `datasets/processed/m5-zero-shot/`：evaluation 销量表，用于历史输入和独立评测标签。

每个存储只有销量二进制、索引清单及准备记录，不需要 schema 或词表。
历史下载的其他本地数据保留在磁盘上，但当前代码不使用、不上传。
