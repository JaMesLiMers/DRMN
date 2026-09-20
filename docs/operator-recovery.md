# Deformable attention implementation

The operator uses the official Deformable DETR module and pure PyTorch reference kernel. The adapter preserves parameter names, initialization, and sampling computations, and adds the diagnostic outputs required by DRMN. Positional encoding follows the input device.

## Upstream sources and license

- [Multi-scale deformable attention module](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py)
- [Reference attention function](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py)
- [Apache 2.0 license](LICENSE-Deformable-DETR)

The kernel supports autograd and CPU execution without CUDA compilation. Sampling-value and gradient checks are included in the test suite. Numerical equivalence and performance relative to the unavailable original custom CUDA operator have not been established; see [reproducibility notes](consistency-audit.md).
