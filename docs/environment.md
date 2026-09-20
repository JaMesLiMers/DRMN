# Environment and dependencies

The locally validated environment uses Python 3.10.12, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, and NumPy 1.26.4. `requirements.txt` pins the full installed dependency set. The virtual environment is not distributed.

On a new machine, create a Python 3.10 virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The validation host lacked system `ensurepip`, so an isolated installation of `virtualenv` was used. If needed, this alternative creates the environment inside the project:

```bash
python3 -m pip install --target .bootstrap virtualenv
PYTHONPATH=.bootstrap python3 -m virtualenv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The supported computation path no longer requires compilation of the archived Detectron2 CUDA extensions. Image encoding uses a torchvision R101/FPN adapter with explicit parameter mapping. CPU execution has been checked; GPU execution, mixed precision, and numerical equivalence to the original CUDA operator have not. The pure PyTorch reference operator is slower, and synthetic checks do not establish full-dataset evaluation speed.
