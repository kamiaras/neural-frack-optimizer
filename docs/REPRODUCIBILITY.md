# Reproduce the synthetic example

The tested environment is Python 3.12.14 on Windows with CPU PyTorch. `requirements.txt` pins direct dependencies; `requirements-lock-windows-py312.txt` records the tested Windows dependency set. Other platforms and Python versions were not tested.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python examples/synthetic_demo.py --output-dir artifacts/synthetic-demo
```

Activate the environment using the command for your shell before installing. The example refuses to overwrite an existing output directory. Without `--output-dir`, it uses temporary files and removes them on exit. Installation needs package access; the example itself makes no network requests.

The check covers training, held-out loss/evaluation helpers, a checkpoint round trip, and single-output inverse scaling and Q reconstruction. It does not load original project artifacts or validate field performance.

For notebooks, install JupyterLab separately (`python -m pip install jupyterlab`). The optional Jupyter application was not part of the tested environment. Start with [`examples/synthetic_walkthrough.ipynb`](../examples/synthetic_walkthrough.ipynb). Its code cells run the same synthetic workflow.

The other notebooks contain reusable function definitions only. Their original call cells, data-dependent bounds and outputs were removed. They locate helper modules from the repository root; start the kernel inside this repository. Calling their training/inversion functions requires explicitly supplied inputs and can write result files. Their complete algorithms were not rerun in this cleanup.

The helpers assume 14 input columns followed by six outputs, matching scalers, and the correct architecture. They reconstruct one adjustable input from a ratio; applications must separately check any bounds on that reconstructed input. Only load trusted pickle scalers and checkpoints.
