# common_two_output.py

import os
import uuid
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Tuple

def process_two_output_inversion(
    config: Dict[str, Any],
    solver: Callable[..., List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    solver_params: List[str],
    prefix: str,
    n_outputs: int = 2
) -> None:
    """
    Shared driver for two‐output inversion solvers.

    Args:
      config:         dict with keys TEST_PATH, X_SCALER_PATH, Y_SCALER_PATH,
                      MODEL1_PATH, MODEL2_PATH, plus solver‐specific params.
      solver:         function that does inversion, returning
                      List[(adj_raw(4,), preds_raw(n_outputs,), resid(n_outputs,))]
      solver_params:  list of config‐keys to pass through to solver
      prefix:         prefix for output_dir and filenames, e.g. "QPSO2Q"
      n_outputs:      number of outputs (2 here)
    """
    # 1) Load test DataFrame
    df = pd.read_csv(config["TEST_PATH"])
    # columns for the two targets
    output_cols = df.columns[14:14 + n_outputs].tolist()
    base        = os.path.splitext(os.path.basename(config["TEST_PATH"]))[0]

    # 2) Prepare output folder + save params
    tag        = uuid.uuid4().hex[:8]
    output_dir = f"{prefix}_{tag}"
    os.makedirs(output_dir, exist_ok=True)

    # exclude file paths from params file
    exclude = {"TEST_PATH", "X_SCALER_PATH", "Y_SCALER_PATH", "MODEL1_PATH", "MODEL2_PATH"}
    params = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in config.items() if k not in exclude
    }
    with open(os.path.join(output_dir, f"params_{tag}.txt"), "w") as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")

    # 3) Loop over each sample
    for idx, row in df.iterrows():
        x_all         = row.iloc[:14].to_numpy(dtype=float)
        actual_x_adj  = x_all[:4]
        x_fixed       = x_all[4:]
        targets       = row.iloc[14:14 + n_outputs].to_numpy(dtype=float)

        # build solver kwargs
        kwargs: Dict[str, Any] = {
            "x_scaler_path":      config["X_SCALER_PATH"],
            "y_scaler_path":      config["Y_SCALER_PATH"],
            "model1_path":        config["MODEL1_PATH"],
            "model2_path":        config["MODEL2_PATH"],
            "x_fixed_raw":        x_fixed,
            "actual_x_adj_raw":   actual_x_adj,
            "target1_raw":        targets[0],
            "target2_raw":        targets[1],
        }
        for p in solver_params:
            kwargs[p] = config[p]

        # run solver
        sols = solver(**kwargs)

        # 4) Build DataFrame of results
        cols = list(df.columns[:4]) \
             + output_cols \
             + [f"resid_{c}" for c in output_cols]
        rows: List[List[float]] = []

        # 4a) actual row (zero residuals)
        rows.append(
            list(actual_x_adj)
            + targets.tolist()
            + [0.0] * n_outputs
        )

        # 4b) all solver‐returned solutions
        for x_adj_raw, preds_raw, resid in sols:
            rows.append(
                list(x_adj_raw)
                + preds_raw.tolist()
                + resid.tolist()
            )

        out_df  = pd.DataFrame(rows, columns=cols)
        fname   = f"{base}_{idx}_{prefix}_{tag}.csv"
        out_df.to_csv(os.path.join(output_dir, fname), index=False)
