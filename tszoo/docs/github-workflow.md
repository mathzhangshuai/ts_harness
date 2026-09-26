# GitHub 与云端环境

源码通过 GitHub 同步，不使用文件包：

```bash
git clone https://github.com/mathzhangshuai/ts_harness.git
cd ts_harness/tszoo
```

私有仓库需要先在目标机器完成 GitHub 身份验证。

## 环境

使用 Python 3.11。先确认云端 GPU 驱动与 PyTorch 构建兼容；当前 `requirements.txt` 固定 CUDA 12.1 的 PyTorch。

```bash
python -m pip install -r requirements.txt
python run.py --help
python run.py test
```

测试没有训练参数更新；缺少外部参考源码时跳过原版数值对照。

## 资源

`python run.py download` 下载并校验 M5 与两个基线权重。只训练 Small 时加 `--models chronos2_small`。也可单独传输已下载资源，保持配置中的相对路径。

`python run.py prepare` 创建训练和评测存储。后续步骤见 [训练流程](m5-small-training.md) 与 [零样本基线](m5-zero-shot.md)。

## Git 范围

只提交源码、配置、测试、说明、许可证与 `data/manifest.json` 下载清单。

`datasets/` 仅允许提交 README；原始和派生数据、预训练权重、训练检查点、运行输出、本地依赖、缓存与参考仓库均忽略。不强制添加这些资源。
