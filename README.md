# DRMN

**Context Does Matter: End-to-end Panoptic Narrative Grounding with Deformable Attention Refined Matching Network**

[Paper](https://arxiv.org/abs/2310.16616) · [Workflows](docs/workflows.md) · [Dataset Preparation](docs/dataset-download.md) · [Implementation Notes](docs/method-mapping.md) · [Reproducibility Status](docs/consistency-audit.md)

DRMN addresses **Panoptic Narrative Grounding**: given an image and a narrative caption, the model predicts a pixel-level segmentation mask for each target noun phrase. It incorporates visual context through multi-scale deformable attention and iteratively refines the image features associated with each phrase.

## Method Overview

The model consists of four main components:

1. **Visual and textual encoding.** Frozen ResNet101/FPN and BERT encoders extract multi-scale image features and text representations.
2. **Initial matching.** Multi-scale deformable encoding produces an initial text-to-pixel response map.
3. **Iterative refinement.** The model selects the top-k relevant pixels, refines their representations using multi-scale context, and aggregates the resulting visual features into the text representations.
4. **Mask prediction.** Segmentation predictions are produced at each stage, with intermediate supervision using BCE and Dice losses.

Evaluation uses Average Recall, reported overall and separately for singular/plural phrases and thing/stuff categories.

## Release Status

This release includes the model, annotation preprocessing, independent pretrained encoder loading, epoch-based training with resume and distributed execution, inference, evaluation, and analysis exports. Validation coverage is documented in the [implementation notes](docs/consistency-audit.md).

- **Pretrained weights:** not currently distributed. Inference and evaluation on real data require a compatible DRMN checkpoint.
- **Runtime:** validated with the pure PyTorch reference implementation of deformable attention. GPU execution and performance have not been validated.
- **Reproducibility:** computation paths and parameter compatibility have been checked; the reported paper metrics have not been re-evaluated using the original weights and dataset.


## Installation

```bash
git clone https://github.com/JaMesLiMers/DRMN.git
cd DRMN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the dependencies used in the validated CPU environment. See [Environment Setup](docs/environment.md) for additional details.

## Dataset Preparation

The project uses **COCO 2017** images and panoptic segmentation annotations, together with **Panoptic Narrative Grounding** narrative annotations. Download sources and the expected directory layout are provided in the [dataset guide](docs/dataset-download.md).

After obtaining the raw annotations, generate the dataloader JSON for the desired split:

```bash
python tools/preprocess_annotations.py \
  --data_dir /path/to/png \
  --splits val2017
```

Extract frozen encoder features using separately downloaded pretrained encoders (see [weight sources](docs/workflows.md#pretrained-encoders)):

```bash
python tools/encode_data.py \
  --config configs/drmn.yaml \
  --fpn-weights /path/to/model_final_cafdb1.pkl \
  --bert-weights /path/to/pytorch_model.bin \
  --png-json /path/to/png/annotations/png_coco_val2017_dataloader.json \
  --panoptic-json /path/to/png/annotations/panoptic_val2017.json \
  --panoptic-masks /path/to/png/annotations/panoptic_segmentation/val2017 \
  --images /path/to/png/images/val2017 \
  --output /path/to/prepared_val
```

An intact original checkpoint containing both encoders can alternatively be supplied through `--checkpoint`. A trained DRMN head is not required for feature extraction.

See [Data Format](docs/data-format.md) for the cached feature format and annotation alignment requirements.

## Inference

Create an input file named `input.json`:

```json
{
  "image": "/path/to/image.jpg",
  "caption": "a man with a dog",
  "noun_ids": [0, 1, 0, 0, 2]
}
```

Each entry in `noun_ids` corresponds to a BERT WordPiece token, excluding `[CLS]` and `[SEP]`. Use `0` for non-target tokens and the same positive integer for tokens belonging to the same noun phrase.

```bash
python tools/predict.py \
  --config configs/drmn.yaml \
  --checkpoint /path/to/model_best.pth \
  --input input.json \
  --output artifacts/predictions
```

For a head-only checkpoint produced by `train_epochs.py`, also pass `--fpn-weights` and `--bert-weights` using the same encoders used to prepare its training features.

Outputs include masks at the model output resolution, PNG masks resized to the original image dimensions for visualization, and prediction metadata. To export masks from cached features, use `tools/visualize.py`.

## Evaluation

```bash
python tools/evaluate.py \
  --config configs/drmn.yaml \
  --checkpoint /path/to/model_best.pth \
  --data /path/to/prepared_val \
  --output artifacts/runs/evaluation.json
```

The output records overall and per-group Average Recall, sample counts, configuration, and dataset provenance. Checkpoint integrity is checked before loading, and model parameters are matched strictly. Checkpoints can also be inspected independently:

```bash
python tools/check_checkpoint.py /path/to/model_best.pth
```

## Training

Train on cached frozen features with a fixed Adam learning rate:

```bash
python tools/train_epochs.py \
  --config configs/drmn.yaml \
  --data /path/to/prepared_train --val-data /path/to/prepared_val \
  --output artifacts/runs/drmn --epochs 20
```

The runner supports gradient accumulation, epoch-boundary resume, validation, and `last.pth` / `best.pth` checkpoint saving. `torchrun` enables distributed execution. See the [workflow guide](docs/workflows.md) for commands, checkpoint requirements, and batching semantics. Training-protocol provenance and validation limits are documented in the [implementation notes](docs/consistency-audit.md).

## Smoke Test

`tools/train.py` provides a bounded-step diagnostic interface.

The default `--max-steps 0` performs one forward and backward pass without updating parameters. The following smoke test requires neither real data nor pretrained weights:

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

This test uses synthetic inputs and random initialization to validate the computation pipeline; it does not measure model quality. `configs/smoke.yaml` is a test configuration. The provenance and unresolved settings in `configs/drmn.yaml` are described in the [implementation notes](docs/method-mapping.md).

## Analysis and Ablations

`tools/analyze.py` exports stage-wise probability maps, top-k pixel attention, deformable sampling plots, and the underlying numeric arrays. `tools/plot_recall.py` plots overall and grouped recall curves from actual evaluation outputs. See [analysis commands and interpretation](docs/workflows.md#analysis).

[Structural variants](configs/ablations/README.md) cover projection-only image encoding, three image encoder layers, and initial matching without iterative refinement. These are executable analysis configurations; historical ablation results and exact paper figure selections are not claimed.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover sampling and gradients, mask losses, phrase alignment, Average Recall, strict checkpoint loading, prediction consistency after saving and loading, the image-and-text-to-mask computation pipeline, pretrained state loading, training/resume control, structural variants, and analysis exports.

## Repository Structure

```text
configs/        Model and test configurations; BERT vocabulary
drmn/           Models, operators, encoders, data processing, and evaluation
tools/          Preprocessing, feature extraction, inference, evaluation, and training
scripts/        Verification scripts
tests/          Unit and integration tests
docs/           Data, environment, implementation, and reproducibility documentation
artifacts/      Local outputs; generated files are excluded from version control
```

## Acknowledgements and License

This project builds on [PPMN](https://github.com/dzh19990407/PPMN) and [Panoptic Narrative Grounding](https://github.com/BCV-Uniandes/PNG), uses the deformable attention reference implementation from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and includes implementations or adaptations originating from Detectron2, BERT, and OpenMMLab.

The repository retains its [MIT License](LICENSE). Third-party code retains its original copyright and license notices. The Apache 2.0 license for the Deformable DETR reference implementation is included [here](docs/LICENSE-Deformable-DETR).
