# Environment and dependencies

The locally validated environment uses Python 3.10.12, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, and NumPy 1.26.4. `requirements.txt` pins the full installed dependency set.

On a new machine, create a Python 3.10 virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

If `venv` is unavailable because the Python installation lacks `ensurepip`, use `virtualenv`:

```bash
python3 -m pip install --target .bootstrap virtualenv
PYTHONPATH=.bootstrap python3 -m virtualenv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The image encoder uses a torchvision R101/FPN adapter. Deformable attention uses a pure PyTorch reference implementation, so the default CPU setup does not require CUDA extension compilation. GPU execution, mixed precision, and performance benchmarks remain unvalidated. See [implementation notes](consistency-audit.md).

## GPU environment prerequisite

The pinned requirements install CPU-only PyTorch and torchvision builds. They cannot run the CUDA commands in the workflow guide. Before using `--device cuda`, configure a separate environment with mutually compatible CUDA-enabled PyTorch and torchvision builds and an appropriate NVIDIA driver. Do not reapply the CPU-pinned requirements over that environment.

Confirm CUDA availability before launching a GPU job:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

`torch.version.cuda` must be non-null and `torch.cuda.is_available()` must return `True`. These checks confirm CUDA availability only; GPU correctness and full training have not been validated for this release.
