# common_inversion.py

import os
import uuid
import pandas as pd
import numpy as np
from typing import Callable, Sequence, Any, Dict

def process_inversion(
    config: Dict[str, Any],
    solver: Callable[..., Sequence[Sequence[Any]]],
    solver_params: Sequence[str],
    prefix: str
) -> None:
    """
    Shared routine to run a solver over every row in TEST_PATH, save parameters,
    collect actual+predicted solutions, and write per-row CSVs.

    Args:
        config:        dict containing keys:
                           TEST_PATH, X_SCALER_PATH, Y_SCALER_PATH, MODEL_PATH,
                           plus all solver-specific params matching solver_params.
        solver:        function to call, signature:
                           solver(x_scaler_path, y_scaler_path, model_path,
                                  x_fixed_raw, actual_x_adj_raw, target_raw,
                                  **solver-specific)
        solver_params: list of keys in config for solver-specific parameters
        prefix:        string prefix for output_dir and filenames (“GAQ” or “QPSOQ”)
    """
    # [1] Load test DataFrame & prepare output directory
    df          = pd.read_csv(config["TEST_PATH"])
    output_cols = df.columns[14:20].tolist()
    base        = os.path.splitext(os.path.basename(config["TEST_PATH"]))[0]
    tag         = uuid.uuid4().hex[:8]
    out_dir     = f"{prefix}_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    # [2] Save config to params file
    with open(os.path.join(out_dir, f"params_{tag}.txt"), "w") as f:
        for key, val in config.items():
            if isinstance(val, np.ndarray):
                f.write(f"{key}: {val.tolist()}\n")
            else:
                f.write(f"{key}: {val}\n")

    # [3] Iterate over each test sample
    for idx, row in df.iterrows():
        # split raw inputs into adjustable & fixed
        x_all         = row.iloc[:14].to_numpy(dtype=float)
        actual_x_adj  = x_all[:4]
        x_fixed       = x_all[4:]
        target_raw    = float(row.iloc[14])

        # build solver kwargs
        kwargs = {
            "x_scaler_path":       config["X_SCALER_PATH"],
            "y_scaler_path":       config["Y_SCALER_PATH"],
            "model_path":          config["MODEL_PATH"],
            "x_fixed_raw":         x_fixed,
            "actual_x_adj_raw":    actual_x_adj,
            "target_raw":          target_raw,
            **{k: config[k] for k in solver_params}
        }

        # call the solver
        sols = solver(**kwargs)

        # [4] Build result rows
        cols = list(df.columns[:4]) + output_cols + ["residual"]
        rows = []
        # actual (residual = 0)
        rows.append(
            list(actual_x_adj) +
            row.iloc[14:20].to_numpy().tolist() +
            [0.0]
        )
        # solver solutions
        for x_adj_raw, preds_raw, resid in sols:
            rows.append(
                list(x_adj_raw) +
                preds_raw.tolist() +
                [resid]
            )

        out_df   = pd.DataFrame(rows, columns=cols)
        filename = f"{base}_{idx}_{prefix}_{tag}.csv"
        out_df.to_csv(os.path.join(out_dir, filename), index=False)
