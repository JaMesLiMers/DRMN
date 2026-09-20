# Training and analysis workflows

## Pretrained encoders

Feature extraction accepts separate pretrained encoder files and does not require a trained DRMN head. Obtain the PanopticFPN ResNet-101 3x model (`model_final_cafdb1.pkl`) through the [official PNG preparation instructions](https://github.com/BCV-Uniandes/PNG), and the PyTorch `pytorch_model.bin` for [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main). Use the repository's matching uncased vocabulary and BERT configuration. The scripts do not download these files automatically.

Pass both `--fpn-weights` and `--bert-weights` to `tools/encode_data.py`, as shown in the root README. Alternatively, an intact original combined checkpoint can provide both encoder states through `--checkpoint`. Public PNG/PPMN head weights are not DRMN head weights.

All active backbone/FPN parameters and all active BERT text-path parameters must exist with compatible shapes. Missing active weights are rejected. Deterministic FPN preprocessing buffers are supplied when the public file omits them. Archived unused BERT visual/cross-modal branches remain initialized but are never called by the text forward path. Tests cover the mapping using synthetic states; actual public weight downloads and real feature extraction were not performed for this release.

`SOURCE.json` records encoder file hashes. Keep it with each feature split. Use identical encoder weights for training features, validation features, and subsequent image-caption inference. The runner hashes feature files for resume checks; it does not automatically prove that independently prepared splits used the same encoders.

## Epoch training

```bash
python tools/train_epochs.py --config configs/drmn.yaml \
  --data /path/to/prepared_train --val-data /path/to/prepared_val \
  --output artifacts/runs/drmn --epochs 20 --accumulation 1

python tools/train_epochs.py --config configs/drmn.yaml \
  --data /path/to/prepared_train --val-data /path/to/prepared_val \
  --output artifacts/runs/drmn --epochs 20 --accumulation 1 \
  --resume artifacts/runs/drmn/last.pth
```

`--epochs` is the total desired epoch count. The learning rate is fixed at the configured value (default Adam, 1e-4). `last.pth` is saved at each completed epoch. With validation, `best.pth` tracks the highest overall Average Recall. `history.jsonl` records epoch losses and metrics. Checkpoints contain the DRMN head, optimizer, epoch/step counters and per-rank Torch RNG states; frozen encoders remain separate.

Resume requires the same model/configuration, feature file contents and filenames, validation split, accumulation, device type, and process count. Resume occurs at epoch boundaries. It does not recover a partially completed epoch. The seed determines each epoch's shuffle. The current training path uses Torch randomness; Python/NumPy randomness is not used in sample loading or model forward.

Each process handles one image at a time because image and phrase dimensions vary. Losses are averaged across images within each accumulation group; the last group uses its actual length. Normal effective batch size is process count × accumulation. For equal rank lengths, distributed sampling repeats a few samples when the dataset size is not divisible by process count. This is an explicit reconstructed protocol, not evidence of the historical batch size or exact training trajectory.

A three-GPU launch can be configured as follows; GPU execution and full training remain unvalidated:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  tools/train_epochs.py --config configs/drmn.yaml \
  --data /path/to/prepared_train --val-data /path/to/prepared_val \
  --output artifacts/runs/drmn_ddp --device cuda --epochs 20
```

Validation runs on rank zero. Distributed execution uses NCCL on CUDA and Gloo on CPU. Performance relative to the original CUDA extension has not been benchmarked.

## Validation without training

```bash
python tools/prepare_data.py --config configs/smoke.yaml \
  --data artifacts/smoke_data --synthetic-smoke
python tools/train_epochs.py --config configs/smoke.yaml \
  --data artifacts/smoke_data --output artifacts/validation --validate-only
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/train_epochs.py --config configs/smoke.yaml \
  --data artifacts/smoke_data --output artifacts/validation_ddp --validate-only
```

These commands perform one accumulation group of forward/backward passes and write `validation.json`, without optimizer updates or model checkpoint saving. Both single-process and two-process CPU checks passed. Save/best/resume control is tested with a tiny model and mocked optimizer updates, not a full DRMN training run.

## Analysis

Evaluate a compatible head checkpoint on prepared validation features, then plot its actual phrase IoUs:

```bash
python tools/evaluate.py --config configs/drmn.yaml \
  --checkpoint /path/to/head.pth --data /path/to/prepared_val \
  --output artifacts/evaluation.json
python tools/plot_recall.py --results artifacts/evaluation.json \
  --labels DRMN --output artifacts/recall.png
python tools/analyze.py --config configs/drmn.yaml \
  --checkpoint /path/to/head.pth --sample /path/to/prepared_val/sample.npz \
  --phrase 0 --output artifacts/analysis
```

Multiple result files and labels may be supplied for comparisons. Curves show empirical recall versus IoU thresholds; the evaluator separately preserves the archived discrete AR calculation.

Analysis exports include initial and refinement probability maps, per-round top-k cross-attention, deformable sampling at four feature levels, `diagnostics.npz`, and selection metadata. Probability maps average tokens in the selected phrase; attention/sampling visualizations use its first token. Coordinates use normalized padded image geometry. The sampling plot selects the 50 largest deformable weights per level across pixels, heads and offsets. It uses probability-map backgrounds, not the original paper's image overlays. Full numeric coordinates, including out-of-bounds samples, are retained. The selection rule is independent of the paper's original figure selection.

## Reproducibility

See [implementation notes](consistency-audit.md) for checkpoint availability, validation coverage, and differences from the paper and historical training protocol.
