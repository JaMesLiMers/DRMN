# DRMN：论文代码整理与恢复

Context Does Matter: End-to-end Panoptic Narrative Grounding with Deformable Attention Refined Matching Network（2023）。

**当前为可运行的 CPU 验证版。** 原始图片/文本到掩码、特征输入到损失和反向、独立评测及掩码导出已接通。18 项测试通过；没有执行优化器更新或完整训练。原权重损坏、真实数据缺失，因此尚未确认原论文分割效果。

本轮按要求不准备真实数据、不提供 checkpoint。数据获取见 [数据下载说明](docs/数据下载说明.md)，代码审查结论及保留差异见 [最终一致性核对](docs/最终一致性核对.md)。这是不附权重的恢复代码版本，不声明论文指标已复现。

## 快速验证

克隆后在仓库根目录创建环境并安装依赖（Python 3.10）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/prepare_data.py --config configs/smoke.yaml --data artifacts/smoke_data --synthetic-smoke
python tools/train.py --config configs/smoke.yaml --data artifacts/smoke_data --output artifacts/runs/backward_smoke
```

最后一条默认 max-steps=0，仅计算一次前向和反向，不更新参数、不保存新模型。合成数据和随机权重只用于流程验证，不能作为论文结果。训练入口具备显式指定步数的能力，本次不运行正式训练。

新环境安装见 docs/环境与依赖.md；requirements.txt 固定了已验证版本。

## 使用完好的原权重

先检查完整性：

```bash
python tools/check_checkpoint.py /path/to/model_best.pth
```

现有 PPMN_PAPER.zip 中的权重校验失败，禁止跳过校验或以随机参数填补损坏项。所有推理/评测入口严格匹配参数，不使用 strict=False 掩盖问题。

先运行 `tools/preprocess_annotations.py` 生成 dataloader JSON（见数据下载说明）。已准备 PNG 数据时，导出冻结特征（以下路径需替换为真实资产）：

```bash
python tools/encode_data.py --config configs/drmn.yaml   --checkpoint /path/to/model_best.pth   --png-json /path/to/png_coco_val2017_dataloader.json   --panoptic-json /path/to/panoptic_val2017.json   --panoptic-masks /path/to/panoptic_val2017   --images /path/to/val2017 --output /path/to/prepared_val
python tools/evaluate.py --config configs/drmn.yaml   --checkpoint /path/to/model_best.pth --data /path/to/prepared_val   --output artifacts/runs/evaluation.json
```

原始图片预测：`python tools/predict.py --help`。输入 JSON 包含 image、caption、noun_ids；noun_ids 对应不含 CLS/SEP 的每个 WordPiece，非名词填 0、同一短语填相同正整数。例：caption 为 "a man with a dog"，noun_ids 为 [0,1,0,0,2]。其余权重与路径参数见 help。

缓存特征预测：`python tools/visualize.py --help`。原始几何掩码用于评测；恢复至原图尺寸的 PNG 仅用于可视化。

## 目录

- configs/：主模型、合成验证配置、原 BERT 配置和词表；ablations 尚待核对。
- drmn/：模型、算子、冻结编码器、数据和评测实现。
- tools/：数据编码、训练验证、独立评测、预测、校验工具。
- tests/：算子梯度、损失、标注对齐、参数映射及原始输入集成测试。
- docs/：中文方法对应、环境、数据、恢复记录和验收状态。
- artifacts/：运行日志和本地实验产物，默认不提交 Git；发布版不包含原始备份和恢复缓存。

## 与原论文的一致性边界

主模型 281 项参数名/形状匹配；图像编码器 538 项、BERT 465 项匹配。相同参数和相同重建算子下，迁移前后四阶段输出差异为 0。这些检查不证明损坏的原权重可用，也不证明已重现 62.9%。

deformable 算子与图像编码器适配器有重建部分；采样点数 100 由源码默认值和其他实验日志支持，最佳实验仍未完全锁定。论文 Dice 公式与原实现存在分母差异，当前保留原实现。详见 docs/方法对应.md、docs/验收目标.md。

保留仓库原有 MIT LICENSE。第三方代码保留其版权注释；Deformable-DETR 参考实现的 Apache 2.0 许可见 docs/LICENSE-Deformable-DETR。暂不提供 checkpoint。
