# Implementation consistency audit

Initial audit: September 19, 2026. Workflow verification update: September 20, 2026. Scope: compare the paper, archived source, bytecode recovery, and organized implementation using source inspection and synthetic checks. No real dataset preparation or full training was performed; release without checkpoints was permitted.

**The principal method and tested computation paths agree with the archived implementation. The evidence supports a recovered research-code release without weights, but does not establish identical paper formulas, final experiment settings, or measured performance.**

## Confirmed findings

| Item | Evidence | Result |
|---|---|---|
| Main architecture | Entry-by-entry checkpoint metadata comparison | 281 state names/shapes match; three refinement rounds and two image encoder layers have structural/source support |
| BERT | Archived model retained; strict state comparison | 465 names and shapes match |
| Image encoder | 538 mapped state entries; comparison against archived Detectron2 R101/FPN classes | Identical parameters on square and padded rectangular inputs produce zero differences at p2–p5 |
| DRMN core | Before/after execution with identical parameters, synthetic inputs, and reference operator | Zero maximum output difference at all four stages |
| Mask supervision | Archived losses and stage logic | BCE + Dice at every stage, both weighted by one, with padding excluded |
| Inference aggregation | Archived evaluation logic | Average sigmoid token probabilities within each phrase, then threshold at `>0.5` |
| Average Recall | Boundary and ordinary values compared against original meters | Original discrete threshold integration retained; numerical checks passed |
| Data geometry | Archived loader/training logic and reconstructed conversion | Resize, padding, quarter resolution, and `>0` threshold order agree; synthetic annotation checks passed |
| Checkpoint round trip | Strict loading and prediction tests | Test weights preserve predictions after reload; corrupt files and missing parameters are rejected |

## Corrections made during the audit

1. FrozenBN initially showed relative floating-point differences of approximately 10^-6 with non-default statistics. Matching the archived Detectron2 normalization arithmetic eliminated differences at all four feature levels on two input geometries. This used synthetic inputs and shared parameters, not original trained weights.
2. `tools/preprocess_annotations.py` generates dataloader annotations from official PNG JSON. It retains the original label algorithm, uses the local vocabulary, removes hard-coded output paths, and prevents overwrites. Small constructed annotations validate this path without downloading real data.

## Differences and unresolved evidence

### Dice formula

The local camera-ready paper uses the denominator Σp + Σy. The archived `dice_loss.py` uses Σp² + Σy² + 2ε, with ε=0.001. This release preserves the archived implementation; a test explicitly records the difference rather than treating the formulas as equivalent.

This affects the training objective. Dice is not computed during inference with fixed weights, so the discrepancy alone does not alter inference outputs. No retraining or silent substitution of the printed formula was performed.

### Non-parameter settings of the best experiment

Source defaults and two complete ablation logs support `num_points=100`. The beginning of the best-run log is damaged, so its setting cannot be established conclusively. Two ablation logs record `seed=3407` and `num_stages=1`, while their directory names mention three decoder layers. Command-line values and actual construction may differ; those logs must not override the main architecture blindly.

The three-round main architecture is supported by checkpoint parameters, not folder names. The current `seed=0` is for deterministic engineering validation, not a claim about the best run's seed. Configuration comments identify these evidence levels.

### Deformable attention operator

The original custom operator files are missing. This release uses the official Deformable DETR pure PyTorch reference kernel, checked against known sampling values and gradients. Bilinear sampling, multi-scale offsets, and attention have corresponding implementations, but these checks cannot exclude modifications in the missing original CUDA operator.

### Training entry point and real-data evaluation

The complete `head_example.py` source was not recovered. The reconstructed `train_epochs.py` adds epoch training, gradient accumulation, validation, last/best checkpoints, epoch-boundary resume, and a torchrun entry point. Single-process and two-process CPU forward/backward checks passed. Full training and GPU execution remain unvalidated. Batching and resume rules are documented engineering choices, not a recovered historical training trajectory. The older `train.py` remains available for bounded-step diagnostics.

Without an intact original checkpoint and real-data evaluation, neither the reported 62.9% nor real segmentation quality can be confirmed. The historical 62.942% log is retained only as a record of an earlier experiment.

## Verification and release status

As of September 20, 2026, 26 unit/integration tests passed, together with single-process and two-process CPU synthetic forward/backward checks, including distributed accumulation. No real model parameters were updated. Save/resume control tests use a tiny model and mock `optimizer.step`. No real data was downloaded and no weights are distributed. Independent FPN/BERT loading, analysis exports, and three structural variants are included; actual public pretrained files have not been downloaded and validated. See the [workflow guide](workflows.md).

The release includes executable code, pinned dependencies, configurations, tests, English documentation, and official data-source instructions. It is recovered research code, not a fully reproduced paper result. Effectiveness equivalence remains outside the demonstrated conclusions.
