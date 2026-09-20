# Dataset download and preparation

This guide describes how to obtain the assets required to use the repository. No real dataset was downloaded, extracted, or evaluated during release preparation. Model checkpoints are not included.

## Official sources

| Asset | Source | Required files |
|---|---|---|
| COCO 2017 images | [COCO downloads](https://cocodataset.org/#download) | train2017 and val2017 images; val2017 is sufficient for validation only |
| COCO panoptic annotations | [COCO downloads](https://cocodataset.org/#download) | 2017 panoptic JSON files and RGB segmentation PNGs |
| PNG narrative annotations | [PNG project](https://bcv-uniandes.github.io/panoptic-narrative-grounding/) | png_coco_train2017.json and png_coco_val2017.json |
| Asset layout and split definitions | [PNG repository](https://github.com/BCV-Uniandes/PNG#dataset-preparation) | Original dataset preparation instructions |
| One-stage preprocessing reference | [PPMN repository](https://github.com/dzh19990407/PPMN) | Dataloader JSON generation procedure |

Links were checked on September 19, 2026. The project page depends on JavaScript. If a download service is unavailable, consult the official repository or maintainers for an updated location.

The original PNG baseline's cached mask/semantic features are not interchangeable with this repository's four-level FPN and BERT NPZ format. Use the original images, panoptic annotations, and narrative annotations. Baseline proposal features are not required.

## Directory layout

```text
/path/to/png/
├── images/
│   ├── train2017/                # COCO JPG images
│   └── val2017/
└── annotations/
    ├── png_coco_train2017.json
    ├── png_coco_val2017.json
    ├── panoptic_train2017.json
    ├── panoptic_val2017.json
    └── panoptic_segmentation/
        ├── train2017/            # RGB segmentation PNGs for this split
        └── val2017/
```

Archives may contain an additional enclosing directory; adjust paths to match the extracted files. Filenames must agree with COCO image IDs and JSON `file_name` values. Instance/stuff annotations cannot replace panoptic annotations, and COCO 2014 cannot be substituted directly for COCO 2017.

## Generate dataloader annotations

From the repository root, after installing dependencies:

```bash
python tools/preprocess_annotations.py --data_dir /path/to/png --splits val2017
# To prepare training annotations:
python tools/preprocess_annotations.py --data_dir /path/to/png --splits train2017
```

The script uses the bundled BERT vocabulary and does not download a model. It reads existing annotations and writes `png_coco_*_dataloader.json` alongside them, without requiring a DRMN checkpoint. It preserves the archived WordPiece, boxes, noun-vector, and label algorithm, replaces the hard-coded server output path, and refuses to overwrite existing output. Records with token-alignment failures are excluded following the original algorithm; retain the counts reported in the terminal.

## Extract features and evaluate

Once the required assets are available, an intact original combined checkpoint can supply both encoders:

```bash
python tools/encode_data.py --config configs/drmn.yaml \
  --checkpoint /path/to/valid_drmn.pth \
  --png-json /path/to/png/annotations/png_coco_val2017_dataloader.json \
  --panoptic-json /path/to/png/annotations/panoptic_val2017.json \
  --panoptic-masks /path/to/png/annotations/panoptic_segmentation/val2017 \
  --images /path/to/png/images/val2017 --output /path/to/prepared_val
python tools/evaluate.py --config configs/drmn.yaml \
  --checkpoint /path/to/valid_drmn.pth --data /path/to/prepared_val
```

Alternatively, feature extraction accepts separate `--fpn-weights` and `--bert-weights` without a trained DRMN head. See the [workflow guide](workflows.md) for sources and commands. `train_epochs.py` consumes the resulting features. Evaluation still requires compatible trained head weights; public PNG/PPMN checkpoints or frozen encoder weights cannot replace them.

These real-data commands were not executed during release preparation. Datasets should be obtained from their official providers and are not repackaged here. Damaged local checkpoint backups are retained only as recovery evidence, not distributed as usable models.
