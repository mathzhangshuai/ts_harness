# GitHub 与线上环境

项目通过 GitHub 同步源码，不使用迁移压缩包。仓库根目录保留 `tszoo/`，进入该目录后运行命令。

```bash
git clone https://github.com/mathzhangshuai/ts_harness.git
cd ts_harness/tszoo
```

私有仓库需要先在目标机器完成 GitHub 身份验证。

## 环境与资源

使用 Python 3.11。当前 `requirements.txt` 固定 CUDA 12.1 的 PyTorch，需先核对线上 GPU 和驱动是否兼容，再安装依赖。线上硬件适配和训练方案尚未确定。

```bash
python -m pip install -r requirements.txt
python run.py --help
python run.py test
```

M5 原始数据位于 `datasets/m5/`，预训练权重位于 `.model-cache/`。它们不在 Git 仓库内。
执行 `python run.py download --config config/m5-baseline.yaml` 可按固定 revision 和 SHA-256 获取数据与两个模型权重。也可以单独传输已下载的资源并保持配置中的相对路径。

`python run.py baseline` 自动校验资源、准备缺失的目标变量 memmap，再推理和评分；输出至本地 `runs/m5-baseline/`。该目录必须不存在，以防覆盖结果。此命令不训练模型。

## Git 范围

- 上传源码、配置、测试、文档、模型许可证，以及数据来源和校验清单。
- 不上传 `datasets/` 中的原始或派生数据、`.model-cache/`、`weights/`、`checkpoints/`、`runs/`、`.experiment-deps/`、缓存、虚拟环境及 `dist/`。
- `datasets/` 只放行 `README.md`、`manifest.json` 和 `parquet_validation.json` 三个说明或校验文件。
- 外层 `reference_models/` 是独立参考源码仓库，不随本仓库上传；正常训练和推理不依赖它。参考实现对照需另行准备源码和依赖。

后续提交前先检查 `git status` 和 `git diff --cached --stat`。新的数据集目录默认被忽略；不要强制添加被忽略的资源。
