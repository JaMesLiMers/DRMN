# DRMN

**Context Does Matter: End-to-end Panoptic Narrative Grounding with Deformable Attention Refined Matching Network**

[论文](https://arxiv.org/abs/2310.16616) · [数据准备](docs/数据下载说明.md) · [实现说明](docs/方法对应.md) · [复现状态](docs/最终一致性核对.md)

DRMN 面向全景叙述定位（Panoptic Narrative Grounding）：给定一幅图像和描述其中场景的叙述文本，预测各名词短语对应的像素级分割掩码。模型通过多尺度可变形注意力引入视觉上下文，并在迭代匹配过程中细化与短语相关的图像特征。

## 方法概述

DRMN 的主要组成如下：

1. **图像与文本编码**：使用冻结的 ResNet101/FPN 和 BERT 提取多尺度视觉特征与文本表示。
2. **初始图文匹配**：通过多尺度可变形编码构建文本与图像像素的初始响应图。
3. **迭代特征细化**：选择与短语最相关的 top-k 像素，引入多尺度上下文，并将更新后的视觉特征聚合至文本表示。
4. **掩码预测**：输出各阶段的分割预测，以 BCE 与 Dice 损失进行中间监督。

评测采用 Average Recall，并分别报告单数／复数短语及 thing／stuff 类别的结果。

## 发布状态

当前版本提供模型实现、数据预处理、推理、评测及有限步数训练接口。Python 3.10 / CPU 环境下的 18 项单元与集成测试已通过。

- **预训练权重**：暂未提供；推理和真实数据评测需要兼容的 DRMN checkpoint。
- **运行环境**：当前验证基于纯 PyTorch 可变形注意力参考实现；GPU 运行及性能尚未验证。
- **复现范围**：已验证计算流程和参数结构兼容性，尚未完成原权重在真实数据集上的指标复核。

部分模块由归档源码和缓存恢复，部分依赖经过适配。论文公式、保存实现及实验配置之间的已知差异见[一致性核对报告](docs/最终一致性核对.md)。

## 安装

```bash
git clone https://github.com/JaMesLiMers/DRMN.git
cd DRMN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 固定了已验证的 CPU 环境依赖。环境配置说明见[环境与依赖](docs/环境与依赖.md)。

## 数据准备

本项目使用 **COCO 2017** 图像、panoptic 分割标注及 **Panoptic Narrative Grounding** 叙述标注。下载入口和目录约定见[数据下载说明](docs/数据下载说明.md)。

准备原始标注后，生成对应划分的 dataloader JSON：

```bash
python tools/preprocess_annotations.py \
  --data_dir /path/to/png \
  --splits val2017
```

使用兼容 checkpoint 提取冻结编码器特征：

```bash
python tools/encode_data.py \
  --config configs/drmn.yaml \
  --checkpoint /path/to/model_best.pth \
  --png-json /path/to/png/annotations/png_coco_val2017_dataloader.json \
  --panoptic-json /path/to/png/annotations/panoptic_val2017.json \
  --panoptic-masks /path/to/png/annotations/panoptic_segmentation/val2017 \
  --images /path/to/png/images/val2017 \
  --output /path/to/prepared_val
```

缓存格式及标注对齐约定见[数据格式说明](docs/数据准备.md)。

## 推理

创建输入文件 `input.json`：

```json
{
  "image": "/path/to/image.jpg",
  "caption": "a man with a dog",
  "noun_ids": [0, 1, 0, 0, 2]
}
```

`noun_ids` 与 BERT WordPiece token 逐项对应，不包含 `[CLS]` 和 `[SEP]`。非目标 token 标为 `0`，同一名词短语使用相同的正整数编号。

```bash
python tools/predict.py \
  --config configs/drmn.yaml \
  --checkpoint /path/to/model_best.pth \
  --input input.json \
  --output artifacts/predictions
```

输出包含模型尺度下的掩码、用于可视化的原图尺寸 PNG，以及预测记录。已有缓存特征时，可使用 `tools/visualize.py` 导出掩码。

## 评测

```bash
python tools/evaluate.py \
  --config configs/drmn.yaml \
  --checkpoint /path/to/model_best.pth \
  --data /path/to/prepared_val \
  --output artifacts/runs/evaluation.json
```

评测结果包含整体与各分组的 Average Recall、样本数量、配置及数据来源。权重加载前执行完整性检查，并严格匹配模型参数。也可单独检查 checkpoint：

```bash
python tools/check_checkpoint.py /path/to/model_best.pth
```

## 训练接口与流程验证

`tools/train.py` 接收已准备的冻结特征，支持阶段监督、有限步数参数更新及优化器状态恢复。该入口不包含原实验完整的多 GPU 和 epoch 调度流程。

默认 `--max-steps 0` 仅执行一次前向与反向传播，不更新参数。以下命令无需真实数据或预训练权重：

```bash
python tools/prepare_data.py \
  --config configs/smoke.yaml \
  --data artifacts/smoke_data \
  --synthetic-smoke

python tools/train.py \
  --config configs/smoke.yaml \
  --data artifacts/smoke_data \
  --output artifacts/runs/backward_smoke
```

以上使用合成样本与随机初始化，仅用于验证计算流程，不用于评估模型质量。`configs/smoke.yaml` 为测试配置；`configs/drmn.yaml` 的实验参数来源和待确认项见[实现说明](docs/方法对应.md)。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖采样与梯度、掩码损失、短语对齐、Average Recall、权重严格加载、保存／加载一致性，以及图像和文本到掩码的计算流程。

## 项目结构

```text
configs/        模型配置、测试配置与 BERT 词表
drmn/           模型、算子、编码器、数据处理与评测
tools/          预处理、特征提取、推理、评测与训练入口
scripts/        验证脚本
tests/          单元与集成测试
docs/           数据、环境、方法及复现说明
artifacts/      本地输出目录，生成内容默认不纳入版本管理
```

## 致谢与许可

本项目基于 [PPMN](https://github.com/dzh19990407/PPMN) 与 [Panoptic Narrative Grounding](https://github.com/BCV-Uniandes/PNG) 的研究工作，使用 [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) 的可变形注意力参考实现，并包含源自 Detectron2、BERT 和 OpenMMLab 的实现或适配代码。

项目沿用 [MIT License](LICENSE)。第三方代码保留原版权与许可声明；Deformable DETR 的 Apache 2.0 许可见[第三方许可文件](docs/LICENSE-Deformable-DETR)。
