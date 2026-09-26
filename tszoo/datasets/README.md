# 实验数据集

数据下载于 2026-09-25。原始文件共 882,756,697 字节（约 841.86 MiB），保留发布格式，未做清洗、重采样或划分调整。

## 目录

```text
datasets/
├── README.md
├── download.py
├── manifest.json
├── freshretailnet-50k/
│   └── data/
│       ├── train.parquet
│       └── eval.parquet
├── freshretailnet-lt/
│   └── data/
│       ├── train.parquet
│       └── eval.parquet
└── m5/
    ├── calendar.csv
    ├── sell_prices.csv
    ├── sales_train_validation.csv
    ├── sales_train_evaluation.csv
    └── sample_submission.csv
```

原始数据由 tszoo/.gitignore 忽略；说明和校验清单由 Git 管理。

## FreshRetailNet-50K

- 来源：[Dingdong-Inc 官方 Hugging Face 数据集](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)。
- 固定版本：`08c1fab7f9257bc73679d415d65d644165d351d4`。
- 按官方数据卡：train 为 4,500,000 行，eval 为 350,000 行。每行包含门店、商品、日期、日销量、小时销量、缺货状态及其他协变量。
- `sale_amount` 与 `hours_sale` 是经过归一化的销量；建模时不要直接解释为原始销售件数。字段定义参见官方数据卡。
- 许可：CC BY 4.0，实验成果应注明数据来源。
- 原始文件已校验大小和 SHA-256；本地 Parquet 元数据与分批解码校验结果见 `parquet_validation.json`。

读取示例（从tszoo 目录运行）：

```python
import pandas as pd

train = pd.read_parquet(
    "datasets/freshretailnet-50k/data/train.parquet",
    columns=["store_id", "product_id", "dt", "sale_amount"],
)
evaluation = pd.read_parquet(
    "datasets/freshretailnet-50k/data/eval.parquet",
    columns=["store_id", "product_id", "dt", "sale_amount"],
)
```

当前 `ts_harness_env` 已安装 pandas 和 PyArrow，可运行上述示例。大文件建议使用下面的分批读取方式。

## FreshRetailNet-LT

- 来源：[Dingdong-Inc 官方数据集](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-LT)，许可为 CC BY 4.0。
- 固定版本：`8a9543bee53de0e2edc90881c2bbd2bfb711cc51`。
- 保存位置：`freshretailnet-lt/data/train.parquet` 和 `freshretailnet-lt/data/eval.parquet`。
- 两个原始文件共 317,408,002 字节；train 为 7,869,549 行，eval 为 70,000 行。
- 文件大小和 SHA-256 与官方清单核对；数据保持发布格式，未自行合并或重划分。

### 大型 Parquet 分批读取

`ts_harness_env` 已安装 `pyarrow==25.0.1`，并在tszoo/requirements.txt 固定版本。使用 `iter_batches()` 按列、按批处理，避免一次读入完整文件：

```python
import pyarrow.parquet as pq

parquet = pq.ParquetFile(
    "datasets/freshretailnet-lt/data/train.parquet"
)
print("总行数：", parquet.metadata.num_rows)

total_rows = 0
for batch in parquet.iter_batches(
    batch_size=65_536,
    columns=["store_id", "product_id", "dt", "sale_amount"],
):
    frame = batch.to_pandas()
    # 在这里处理当前批次，例如特征统计或分批落盘。
    total_rows += len(frame)

print("已处理行数：", total_rows)
```

不要将全部批次收集进列表后再拼接，否则仍会占用全量内存。批次是读取边界，不保证每批都包含完整的门店—商品时间序列；需要完整历史窗口的模型应另行按序列组织输入。

## M5 零售数据集

- 原始来源：[Kaggle M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)。
- 实际下载来源：[Zenodo 公开副本，记录 10203108](https://zenodo.org/records/10203108)，DOI：10.5281/zenodo.10203108。该记录注明文件来自上述 Kaggle 竞赛。
- 下载的是原始五个 CSV，未转换为第三方预处理格式。文件大小及 MD5 均与 Zenodo 发布清单一致；这不等同于另行从 Kaggle 下载并逐字节比较。
- 使用条件以原始竞赛条款为准。

| 文件 | 用途 |
| --- | --- |
| calendar.csv | 日期、事件及 SNAP 等日历特征 |
| sell_prices.csv | 门店、商品、周粒度价格 |
| sales_train_validation.csv | 历史销量，d_1 至 d_1913 |
| sales_train_evaluation.csv | 扩展历史销量，d_1 至 d_1941 |
| sample_submission.csv | 提交格式模板，不是真实未来销量 |

两份销量表含有重叠历史，**不要直接拼接为两份独立样本**。例如可用 d_1 至 d_1913 训练，以 evaluation 表中的 d_1914 至 d_1941 做 28 天回测；更早历史可另作滚动验证。

```python
import pandas as pd

calendar = pd.read_csv("datasets/m5/calendar.csv")
prices = pd.read_csv("datasets/m5/sell_prices.csv")
sales = pd.read_csv("datasets/m5/sales_train_evaluation.csv")
```

## 重新下载与完整性检查

在tszoo 目录、已安装 requests 的环境中执行：

```powershell
conda activate ts_harness_env
python run.py download --config config/m5-zero-shot.yaml
```

当前下载入口仅检查并补齐 M5 和 YAML 选择的模型权重；FreshRetailNet 文件随项目迁移保留，不自动下载。M5 的来源、大小和 SHA-256 固定在 [下载清单](../data/m5_manifest.json)。完整文件通过校验后复用，部分文件保留为 `.part` 并尝试断点续传；服务器不支持 Range 时重新下载。已有正式文件校验失败则停止，不覆盖原文件。

`manifest.json` 记录来源 URL、FreshRetailNet 版本、文件大小、发布方校验值、本地 SHA-256 和完成时间。FreshRetailNet 使用发布方 SHA-256，M5 使用 Zenodo MD5，同时为所有文件记录本地 SHA-256。
