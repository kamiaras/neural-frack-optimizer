import numpy as np
import torch
import pickle
from typing import Tuple, Any, Dict

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from helper.pipeline_utils import Net


def fitness_s(
    candidate: np.ndarray,
    model: torch.nn.Module,
    x_fixed_s: np.ndarray,
    means_x: np.ndarray,
    scales_x: np.ndarray,
    target_scaled: float
) -> float:
    """
    Fitness for QPSO: squared error on the *scaled* first output.

    Args:
        candidate: np.array([x0_s, x2_s, x3_s, Q])
        model:     trained Net instance
        x_fixed_s: scaled fixed inputs (remaining features)
        means_x:   x_scaler.mean_ (length 14)
        scales_x:  x_scaler.scale_ (length 14)
        target_scaled: (target_raw - y_mean)/y_scale

    Returns:
        Squared error between model’s first scaled output and target_scaled.
    """
    x0_s, x2_s, x3_s, Q = candidate

    # reconstruct scaled x1 via Q * raw x0
    x0_raw = x0_s * scales_x[0] + means_x[0]
    x1_raw = Q    * x0_raw
    x1_s   = (x1_raw - means_x[1]) / scales_x[1]

    # full adjustable scaled vector [x0_s, x1_s, x2_s, x3_s]
    adj_s = np.array([x0_s, x1_s, x2_s, x3_s])
    inp   = torch.from_numpy(np.concatenate([adj_s, x_fixed_s])).float()

    model.eval()
    with torch.no_grad():
        pred0_s = model(inp)[0].item()

    return (pred0_s - target_scaled) ** 2



def unpack_qpso_solutions(
    candidates: list[np.ndarray],
    means_x: np.ndarray,
    scales_x: np.ndarray,
    x_fixed_s: np.ndarray,
    model: torch.nn.Module,
    y_scaler,
    target_raw: float
) -> list[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Inverse-scale QPSO candidates and compute raw predictions + residuals.

    Args:
        candidates:  List of arrays [x0_s, x2_s, x3_s, Q].
        means_x:     x_scaler.mean_ (length 14).
        scales_x:    x_scaler.scale_ (length 14).
        x_fixed_s:   Scaled fixed inputs (length 10).
        model:       Trained Net.
        y_scaler:    Fitted StandardScaler for outputs.
        target_raw:  Desired raw value of the first output.

    Returns:
        List of tuples:
           (x_adj_raw (4,), y_pred_raw (6,), residual_raw).
    """
    results = []
    mean_y, scale_y = y_scaler.mean_[0], y_scaler.scale_[0]

    for cand in candidates:
        x0_s, x2_s, x3_s, Q = cand

        # reconstruct raw adjustable inputs
        x0_raw = x0_s * scales_x[0] + means_x[0]
        x1_raw = Q    * x0_raw
        x2_raw = x2_s * scales_x[2] + means_x[2]
        x3_raw = x3_s * scales_x[3] + means_x[3]
        x_adj_raw = np.array([x0_raw, x1_raw, x2_raw, x3_raw])

        # predict full network output
        x1_s     = (x1_raw - means_x[1]) / scales_x[1]
        adj_s    = np.array([x0_s, x1_s, x2_s, x3_s])
        inp      = torch.from_numpy(np.concatenate([adj_s, x_fixed_s])).float()
        with torch.no_grad():
            preds_s = model(inp).numpy()

        # inverse-transform and compute residual on first output
        preds_raw = y_scaler.inverse_transform(preds_s.reshape(1, -1)).ravel()
        resid     = abs(preds_raw[0] - target_raw)

        results.append((x_adj_raw, preds_raw, resid))

    return results






def initialize_inversion(
    x_scaler_path: str,
    y_scaler_path: str,
    model_path: str,
    hidden_dims: list,
    activation: str,
    x_fixed_raw: np.ndarray,        # shape (10,)
    actual_x_adj_raw: np.ndarray,   # shape (4,)
    target_raw: float,
    x_adj_intervals_raw: np.ndarray,  # shape (4,2)
    Q_bounds: Tuple[float, float]     # (Q_min, Q_max)
) -> Dict[str, Any]:
    """
    Load scalers & model, scale inputs, compute the “actual” prediction,
    and build QPSO bounds + scaled target.

    Returns a dict with:
      - x_scaler, y_scaler
      - model
      - means_x, scales_x, mean_y, scale_y
      - x_fixed_s, x_adj_act_s
      - results: [(actual_x_adj_raw, pred_act_raw, resid_act)]
      - min_vec, max_vec  (for QPSO)
      - target_scaled
    """
    # [1] Load scalers
    with open(x_scaler_path, "rb") as f:
        x_scaler = pickle.load(f)
    with open(y_scaler_path, "rb") as f:
        y_scaler = pickle.load(f)

    # [2] Load model
    model = Net(in_dim=14, hidden_dims=hidden_dims, out_dim=6, activation_str=activation)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # [3] Extract scaling params
    means_x, scales_x = x_scaler.mean_, x_scaler.scale_
    mean_y, scale_y   = y_scaler.mean_[0], y_scaler.scale_[0]

    # [4] Scale fixed & actual adjustable inputs
    x_fixed_s   = (x_fixed_raw   - means_x[4:]) / scales_x[4:]
    x_adj_act_s = (actual_x_adj_raw - means_x[:4]) / scales_x[:4]

    # [5] Compute “actual” prediction & residual
    inp_act = torch.from_numpy(np.concatenate([x_adj_act_s, x_fixed_s])).float()
    with torch.no_grad():
        pred_act_s = model(inp_act).numpy()
    pred_act_raw = y_scaler.inverse_transform(pred_act_s.reshape(1, -1)).ravel()
    resid_act    = abs(pred_act_raw[0] - target_raw)

    results = [(actual_x_adj_raw.copy(), pred_act_raw, resid_act)]

    # [6] Build scaled search bounds for [x0, x2, x3, Q]
    min_raw, max_raw = x_adj_intervals_raw[:, 0], x_adj_intervals_raw[:, 1]
    mins = np.array([
        (min_raw[0] - means_x[0]) / scales_x[0],
        (min_raw[2] - means_x[2]) / scales_x[2],
        (min_raw[3] - means_x[3]) / scales_x[3],
    ])
    maxs = np.array([
        (max_raw[0] - means_x[0]) / scales_x[0],
        (max_raw[2] - means_x[2]) / scales_x[2],
        (max_raw[3] - means_x[3]) / scales_x[3],
    ])
    Q_min, Q_max = Q_bounds
    min_vec = np.concatenate([mins, [Q_min]])
    max_vec = np.concatenate([maxs, [Q_max]])

    # [7] Precompute scaled target for fitness
    target_scaled = (target_raw - mean_y) / scale_y

    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "model": model,
        "means_x": means_x,
        "scales_x": scales_x,
        "mean_y": mean_y,
        "scale_y": scale_y,
        "x_fixed_s": x_fixed_s,
        "x_adj_act_s": x_adj_act_s,
        "results": results,
        "min_vec": min_vec,
        "max_vec": max_vec,
        "target_scaled": target_scaled,
    }
