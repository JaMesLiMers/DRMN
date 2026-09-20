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
