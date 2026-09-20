# Mapping the paper to the implementation

Paths below are relative to the repository root unless stated otherwise.

| Component | Implementation | Evidence and validation |
|---|---|---|
| Frozen ResNet101 + FPN | `drmn/models/frozen_encoders.py` | Reconstructed adapter: stride in 1×1 convolutions, FrozenBN, BGR normalization, sum FPN; 538 state entries map to archived metadata |
| Frozen BERT | `drmn/models/encoder_bert.py`, `modeling.py`, `FrozenTextEncoder` | Archived model retained; 465 state names and shapes match |
| Multi-scale deformable encoding | `drmn/models/head_model/deform_encoder_head.py`, `deform_img_encoder.py` | Encoder recovered from bytecode; operator uses the attributed PyTorch reference kernel |
| Initial matching | `drmn/models/head_model/kernel_head_new.py` | Archived computation retained |
| Top-k selection and iterative refinement | `kernel_update_head.py`, `deform_iter_head.py`, `deform_txt_decoder.py` under `drmn/models/head_model/` | Archived implementation plus recovered decoder; three refinement rounds match checkpoint structure |
| Intermediate supervision | `drmn.runtime.stage_loss`, `drmn/losses/` | BCE + Dice summed over stages; entry point reconstructed from saved training code |
| Phrase aggregation and evaluation | `drmn/evaluation/metrics.py` | Token probabilities averaged before thresholding; discrete Average Recall checked against archived meters |
| PNG preprocessing | `drmn/datasets/png.py`, `tools/encode_data.py` | Reconstructed from the original loader; synthetic annotation checks passed; full real-data validation remains outstanding |

All 281 default head state names and shapes match archived checkpoint metadata. Before/after comparisons using identical random parameters, synthetic inputs, and the same reconstructed operator produced zero differences at all four stages. This does not cover the missing original CUDA operator or establish reproduction of paper metrics.

The archived decoder had dictionary and list variants for debug output. The dictionary variant was selected to match the saved `kernel_update_head` caller. Declared but unused text branches in the original encoder remain unchanged.

The default `num_points=100` in `configs/drmn.yaml` comes from source defaults; it is not conclusively established for the best experiment. State names and shapes cannot prove this setting. The executable [structural variants](../configs/ablations/README.md) do not claim historical ablation configurations or results.

Image encoder outputs were also compared against archived Detectron2 with identical parameters, two synthetic image geometries, and non-default batch-normalization statistics. All four feature levels had zero differences. The paper's Dice denominator differs from the saved implementation; see the [consistency audit](consistency-audit.md). Agreement at the component level does not imply exact agreement with every printed formula.
