# Method and implementation

DRMN combines frozen image and text encoders with multi-scale deformable encoding and iterative phrase-conditioned mask refinement.

| Component | Entry point | Computation |
|---|---|---|
| Image encoding | [frozen_encoders.py](../drmn/models/frozen_encoders.py) | Frozen ResNet101/FPN; BGR normalization, FrozenBN, and additive pyramid fusion |
| Text encoding | [encoder_bert.py](../drmn/models/encoder_bert.py) | Frozen BERT WordPiece representations |
| Multi-scale encoding | [deform_encoder_head.py](../drmn/models/head_model/deform_encoder_head.py) | Deformable attention across image feature levels |
| Initial matching | [kernel_head_new.py](../drmn/models/head_model/kernel_head_new.py) | Initial text-to-pixel response maps |
| Iterative refinement | [kernel_update_head.py](../drmn/models/head_model/kernel_update_head.py) and [deform_iter_head.py](../drmn/models/head_model/deform_iter_head.py) | Top-k pixel selection, context aggregation, and three refinement rounds in the default model |
| Supervision | [runtime.py](../drmn/runtime.py) | BCE + Dice summed across stages for valid noun tokens; padding excluded |
| Phrase evaluation | [metrics.py](../drmn/evaluation/metrics.py) | Average sigmoid token probabilities within each phrase, threshold at `>0.5`, then compute discrete Average Recall |
| Data conversion | [png.py](../drmn/datasets/png.py) | Align narrative tokens, panoptic labels, and masks at the model output geometry |

The default configuration is [drmn.yaml](../configs/drmn.yaml). [Implementation notes](consistency-audit.md) document validation evidence, the Dice formula difference, and unresolved historical settings. [Structural variants](../configs/ablations/README.md) provide additional analysis configurations.
