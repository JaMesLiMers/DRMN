# Command-line tools

Run `python tools/<name>.py --help` from the repository root for arguments.

| Tool | Purpose |
|---|---|
| `check_checkpoint.py` | Inspect ZIP/CRC integrity without executing checkpoint contents |
| `check_sources.py` | Verify archived source provenance and reviewed file hashes |
| `prepare_data.py` | Validate NPZ samples or explicitly generate synthetic fixtures |
| `preprocess_annotations.py` | Generate PNG dataloader annotations using the archived WordPiece/label algorithm and local paths; no model weights required |
| `encode_data.py` | Export frozen FPN/BERT features from raw PNG/COCO data using a combined checkpoint or separate pretrained encoder weights |
| `train.py` | Bounded-step diagnostics; defaults to forward/backward without parameter updates |
| `train_epochs.py` | Epoch training, gradient accumulation, validation, checkpoints, resume, and torchrun; `--validate-only` performs no updates |
| `evaluate.py` | Strictly load weights, evaluate all supplied feature samples, and record provenance and grouped metrics |
| `predict.py` | Produce masks from an image, text, and WordPiece phrase IDs; accepts separate encoder weights with head-only checkpoints |
| `visualize.py` | Export phrase-mask PNGs from cached features |
| `analyze.py` | Export stage-wise probability maps, top-k attention, deformable sampling plots, and numeric arrays |
| `plot_recall.py` | Plot five groups of recall curves from actual per-phrase evaluation results |

See [workflows](../docs/workflows.md) for complete examples and verification limits. Original-weight and full real-data acceptance remain outstanding.
