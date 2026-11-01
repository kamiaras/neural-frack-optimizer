# Neural model inversion

Python tools for training a neural regression model and searching for inputs that approach a target prediction. The code came from hydraulic-fracturing model experiments. This public version contains code and invented examples only.

## Start here

Create and activate a Python virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python examples/synthetic_demo.py --output-dir artifacts/synthetic-demo
```

The example generates 64 synthetic rows, trains on 48 rows, checks 16 held-out rows, and searches 128 candidate inputs. It uses the existing model and single-output inversion helpers. It reads no external dataset or saved project model. Choose a new output directory for each saved run; omit `--output-dir` to use temporary files.

The [synthetic walkthrough notebook](examples/synthetic_walkthrough.ipynb) runs the same example. See [setup and tested scope](docs/REPRODUCIBILITY.md).

## Code map

| Location | Purpose |
|---|---|
| `helper/pipeline_utils.py` | Network, dataset, weighted loss and evaluation helpers |
| `Invert-PropOnly-saveCSV/` | Single-output inversion helpers and DE, GA, QPSO and mesh function notebooks |
| `Invert-Prop+cond-saveCSV/` | Two-output inversion helpers and solver function notebooks |
| `NN_for_Conductivity/`, `NN_for_PropLenght/` | Training and evaluation function notebooks |
| `examples/` | Complete synthetic command-line and notebook examples |

The older notebooks retain function definitions. Project-specific call cells, bounds, model names and saved results have been removed. They do not run a real-data experiment on opening.

## Limits and data

The inversion helpers use 14 inputs and six outputs. The first four inputs are adjustable; the second is reconstructed as `Q * first_input`. Models must use matching feature order, architecture and scalers. Only load trusted checkpoint and pickle files.

A small prediction residual does not prove a unique solution or a field outcome. The short synthetic search is not a solver benchmark or a proof of optimality. Two-output solver notebooks and full training workflows have not been validated by this small example.

No original datasets, fitted project scalers, trained project weights, run parameters or result tables are distributed. See [data policy](docs/DATA_PROVENANCE.md). No new license is granted by this cleanup.
