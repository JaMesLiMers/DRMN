# Deformable attention recovery

The original `models/toy_model` operator directory is missing. This release uses the official Deformable DETR module and pure PyTorch reference kernel, preserving parameter names, initialization, and sampling computations while adding the debug return interface expected by archived callers.

Upstream source:

- [Multi-scale deformable attention module](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py)
- [Reference attention function](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py)

Local recovery backups are not distributed. The upstream Apache 2.0 license is included in [LICENSE-Deformable-DETR](LICENSE-Deformable-DETR). The repository retains its existing [MIT license](../LICENSE); third-party code remains subject to its original terms.

This is a reconstructed dependency with known provenance, not the recovered original custom operator. The pure PyTorch kernel supports autograd and CPU execution, but does not match the original CUDA extension's execution speed. Numerical equivalence to the missing operator has not been established and requires further evidence. Positional encoding was changed to follow the input device without changing its formula.
