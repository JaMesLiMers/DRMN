# 可用入口

- check_checkpoint.py：完整 ZIP/CRC 检查，不执行权重。
- check_sources.py：迁移与恢复文件来源、已审查版本哈希。
- prepare_data.py：验证 NPZ 或显式生成合成 fixture。
- encode_data.py：PNG/COCO 原始数据经冻结编码器导出特征，要求完好原权重。
- train.py：默认只前向/反向，不更新参数；可显式指定有限步数，当前任务不执行训练。
- evaluate.py：严格加载权重，评测所有输入特征样本，保存数据来源与各分组指标。
- predict.py：原图、文本与 WordPiece 短语编号到掩码。
- visualize.py：缓存特征到短语掩码 PNG。

在项目根目录执行 `python tools/<name>.py --help`。本次完整真实数据与原权重尚未通过验收。

- preprocess_annotations.py：移植原 PNG WordPiece/标签预处理，修复本地路径，不要求模型权重。

- train_epochs.py：epoch 训练、梯度累积、验证、保存和恢复，支持 torchrun；--validate-only 不更新参数。
- analyze.py：每轮概率图、top-k 注意力、可变形采样及数值导出。
- plot_recall.py：从实际评测 per_phrase 结果绘制五组 recall 曲线。
- encode_data.py / predict.py：支持 --fpn-weights 与 --bert-weights 独立冻结编码器权重。

完整参数示例和验证范围见 [workflows.md](../docs/workflows.md)。
