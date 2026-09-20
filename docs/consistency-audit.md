# Implementation notes and reproducibility

This release adapts archived DRMN source, including modules recovered from bytecode, to the documented runtime. The checks below establish implementation compatibility within their stated scope. The paper's reported performance has not been independently reproduced with this release.

## Validation coverage

Verification recorded on September 20, 2026:

| Component | Check | Result |
|---|---|---|
| DRMN head | Names and shapes against archived checkpoint metadata | All 281 state entries match; three refinement rounds and two image encoder layers |
| Frozen encoders | State mapping against archived metadata | 538 image encoder and 465 BERT entries match |
| Image encoder | Identical parameters against archived Detectron2 on square and padded rectangular synthetic inputs, with non-default FrozenBN statistics | Zero output differences at p2–p5 |
| DRMN computation | Archived and adapted core using identical random parameters, synthetic inputs, and the same reference operator | Zero output differences at all four stages |
| Data and evaluation | Synthetic annotation conversion, phrase aggregation, and numerical comparison with archived AR calculations | Checks passed |
| Checkpoints | Prediction round trips, integrity checks, and strict parameter loading | Checks passed |
| Training and tools | 26 unit/integration tests; single-process and two-process CPU forward/backward, including accumulation | Checks passed without real model parameter updates |

Save/resume lifecycle tests use a small model with mocked optimizer updates. Public encoder loading is tested with synthetic states. Actual public pretrained files, full training, GPU execution, mixed precision, and real-data evaluation remain unvalidated. Tests and commands are described in the [test guide](../tests/README.md) and [workflows](workflows.md).

## Implementation differences

### Dice loss

The paper expresses the Dice denominator as Σp + Σy. The archived implementation uses Σp² + Σy² + 2ε, with ε=0.001. This release retains the archived implementation and tests the distinction. The difference affects the training objective; Dice is not evaluated during fixed-weight inference.

### Experiment configuration

The default `num_points=100` is supported by source defaults and available experiment logs, but has not been confirmed for the best reported run. The current `seed=0` supports deterministic engineering checks and is not established as the original experiment seed. Parameter shapes do not determine either setting.

The [structural variants](../configs/ablations/README.md) are provided for analysis. They are not verified historical ablation configurations or results.

### Attention operator

The original custom CUDA operator is unavailable. The implementation uses the official Deformable DETR pure PyTorch reference kernel, with known-value sampling and gradient checks. Numerical equivalence and performance relative to the original operator have not been established. See [operator provenance](operator-recovery.md).

### Training protocol

The epoch runner is reconstructed from available implementation evidence. Its batching, accumulation, and epoch-boundary resume behavior are specified in the [workflow guide](workflows.md). These rules do not establish an identical historical training trajectory.

## Checkpoints and reported results

No usable original DRMN checkpoint is distributed. The paper's overall 62.9% Average Recall and grouped results have not been re-evaluated with original weights and real data. Structural compatibility and synthetic tests do not establish equivalent segmentation accuracy.

## Source provenance metadata

[source_manifest.json](source_manifest.json) records source locations and file hashes. Each `historical_status` entry preserves recovery and adaptation notes, including labels such as `imports_pending` and `runtime_unverified`. These labels describe earlier processing stages, not the current validation status. The validation coverage above describes the current release.

`source_sha256` identifies archived source content; `sha256` identifies the reviewed destination content. `tools/check_sources.py` verifies destination hashes and syntax, and checks original source hashes when local archives are available. Hash verification does not establish runtime correctness.
