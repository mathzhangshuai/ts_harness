# tszoo

线上环境通过 GitHub 克隆，参见 [GitHub 与线上环境](docs/github-workflow.md)。数据、权重及运行产物不上传；具体训练方案后续确定。

模型探索实现集中在以下五类目录：

```text
tszoo/
  utils/              # 训练、推理、评分、指标与结果文件
  data/               # 字段契约、内存映射、数据转换
  config/             # 配置解析、YAML、字段映射 JSON
  layers/             # 通用 Attention、RMSNorm、MLP、ResidualBlock
  models/
    chronos2/         # 模型配置、编码器、预测头、特征适配与许可证
```

通用层接受显式维度和超参数，不依赖具体模型配置。时序分组、目标注意力策略、
patch 编码及分位数预测头保留在模型目录。参数名称与计算顺序保持不变。

`utils/metrics.py` 和 `utils/io.py` 可独立导入，不加载 PyTorch；
配置解析在 `config/m5.py` 和 `config/training.py`。

从 `tszoo` 目录执行：

```powershell
python run.py baseline
python run.py test
```

YAML 中的相对路径以配置文件所在目录为基准。数据、权重、运行产物、文档、测试及依赖声明均在本目录内。
配置详情见 [config/README.md](config/README.md)，结果见 [文档索引](docs/README.md)。

旧兼容入口和重复实现已删除。`run.py` 支持目录直接运行，自动加载本目录的 `.experiment-deps`；不依赖外层项目，也不安装 Chronos 包。

新环境使用 Python 3.11，并执行 `python -m pip install -r requirements.txt`。当前 requirements 固定 CUDA 12.1 的 PyTorch；其他硬件需选择相应 PyTorch 构建。也可使用 `python -m pip install -e .` 安装本项目。

`download` 校验并补齐 M5 原始文件及 YAML 指定的权重；权重固定从 `https://hf-mirror.com` 下载，并按固定 revision 和 SHA-256 校验。`evaluate` 同样检查资源，并在缺少目标变量存储时自动生成 memmap。已有损坏文件会报错，不自动覆盖。`--models chronos2_small` 可只选择一个权重。

训练入口为 `python run.py train --help`，需要预先准备对应字段存储；自动数据准备当前仅支持 M5 目标变量。可选的 `python run.py reference` 需要外部参考源码和额外依赖；正常训练和推理不依赖 `reference_models`。

## 验证记录

2026-09-26：`python run.py test` 共 51 项通过，包含脱离父目录后的复制运行、下载校验与续传、字段隔离、注意力和评分测试。基线相关代码的 Ruff 检查与格式检查通过。

历史实验输出已清理。当前入口 `python run.py baseline` 使用 `config/m5-baseline.yaml`，仅重新评测 Chronos-2-small 与 Chronos-2，输出 1-WAPE、MAE、WRMSSE；结果保存在 `runs/m5-baseline/`，不执行训练。旧的通用评测入口仍可使用。
