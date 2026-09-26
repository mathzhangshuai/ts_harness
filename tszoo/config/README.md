# 配置

当前基线入口为 `python run.py baseline`，使用 `m5-baseline.yaml`，仅运行 Chronos-2-small、Chronos-2，保存 1-WAPE、MAE、WRMSSE。以下其他配置保留为可选实验配置，不属于当前基线。

从独立项目 `tszoo` 目录执行：

```powershell
python run.py download --config config/m5-zero-shot.yaml --models chronos2_small
python run.py evaluate --config config/m5-zero-shot.yaml --models chronos2_small --output runs/m5-small-rerun
python run.py wrmsse --run runs/m5-small-rerun --output runs/m5-small-rerun-wrmsse
```

当前基线为 `context=512`、`origin=1913`、`horizon=28`，仅 `target`，`grouped + bidirectional`。YAML 相对路径以文件所在目录为基准，CLI 路径以当前目录为基准；已有输出目录不会覆盖。

## 数据与权重

`dataset` 使用 `name: m5` 和 `path: ../datasets/m5`。下载清单固定在 [m5_manifest.json](../data/m5_manifest.json)，目前自动下载仅支持 M5。

`models` 是模型名称到下载配置的映射，每项包含：

- `path`：本地权重目录。
- `repo_id`：镜像中的模型仓库，例如 `amazon/chronos-2`。
- `revision`：完整 40 位 commit。
- `files`：`config.json`、`model.safetensors` 各自的 64 位 SHA-256。

完整可用示例见 [m5-zero-shot.yaml](m5-zero-shot.yaml)。权重从 `https://hf-mirror.com` 下载，不安装 Chronos 包。也可将模型值设为本地路径字符串；此模式不自动下载，缺失时明确报错。自动下载暂不支持分片权重。

`--models` 只加载和下载所选权重；`--device cpu`、`--batch-size 8` 可覆盖配置。`--max-series 32` 仅供工程试跑，不能用于完整 M5 WRMSSE。

`download` 检查原始文件与权重；`evaluate` 还会在缺失目标变量存储时自动准备 memmap。协变量存储和训练数据需显式准备，训练入口使用 [chronos2.yaml](chronos2.yaml)。保存的 `resolved_config.yaml` 是运行审计记录，包含解析后路径与下载来源；复跑以原始配置为入口。

## 注意力

`variate_attention` 可取 `grouped` 或 `global_masked`；两者保持组间隔离语义，前者按组计算，避免完整跨组注意力矩阵。`variate_attention_policy` 可取 `bidirectional` 或 `target_aware`，后者禁止协变量状态读取目标。

结果与完整协议见 [零样本基线](../docs/m5-zero-shot.md) 和 [WRMSSE](../docs/m5-wrmsse.md)。
