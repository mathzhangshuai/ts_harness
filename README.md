# ts_harness

零售时序预测实验。主项目位于 [tszoo/](tszoo/README.md)，当前基线为 Chronos-2-small 与 Chronos-2 的 M5 零样本预测，仅报告 1-WAPE、MAE、WRMSSE。

- [当前基线与指标口径](tszoo/docs/m5-zero-shot.md)
- [Chronos-2-small 仅目标变量训练](tszoo/docs/m5-small-training.md)
- [GitHub 与线上环境](tszoo/docs/github-workflow.md)
- [项目目录与命令](tszoo/README.md)

Git 仅管理源码、配置、测试、文档及数据下载校验清单。原始数据、模型权重、预测数组、运行输出和本地依赖不上传。

```bash
cd tszoo
python run.py baseline
```

先按线上环境说明安装依赖并准备资源。此命令会校验和补齐配置指定的 M5 数据与两个模型权重，执行全量推理及评分；不训练模型。云端微调请使用上方的 Small 训练流程。
