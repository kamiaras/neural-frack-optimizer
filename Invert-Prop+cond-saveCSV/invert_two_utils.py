# invert_two_utils.py

import pickle
import numpy as np
import torch

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from helper.pipeline_utils import  Net


def initialize_two_output_inversion(
    x_scaler_path: str,
    y_scaler_path: str,
    model1_path: str,
    model2_path: str,
    hidden_dims1: list[int],
    hidden_dims2: list[int],
    activation: str,
    x_fixed_raw: np.ndarray,
    actual_x_adj_raw: np.ndarray,
    target1_raw: float,
    target2_raw: float,
    x_adj_intervals_raw: np.ndarray,
    Q_bounds: tuple[float, float]
) -> dict:
    """
    1) Loads input/output scalers and two trained models.
    2) Computes scaled fixed/adaptive inputs.
    3) Evaluates the “actual” scaled→raw predictions & residuals.
    4) Builds search bounds [x0_s, x2_s, x3_s, Q_raw].
    5) Returns everything needed downstream.
    """
    # a) Scalers
    with open(x_scaler_path, "rb") as f:
        x_scaler = pickle.load(f)
    with open(y_scaler_path, "rb") as f:
        y_scaler = pickle.load(f)

    # b) Models (reuse pipeline_utils.Net)
    model1 = Net(in_dim=14, hidden_dims=hidden_dims1, out_dim=6, activation_str=activation)
    model1.load_state_dict(torch.load(model1_path, map_location="cpu"))
    model1.eval()
    model2 = Net(in_dim=14, hidden_dims=hidden_dims2, out_dim=6, activation_str=activation)
    model2.load_state_dict(torch.load(model2_path, map_location="cpu"))
    model2.eval()

    # c) Scaling parameters
    means_x, scales_x = x_scaler.mean_, x_scaler.scale_
    mean_y, scales_y = y_scaler.mean_, y_scaler.scale_

    # d) Scale inputs
    x_fixed_s   = (x_fixed_raw - means_x[4:]) / scales_x[4:]
    x_adj_act_s = (actual_x_adj_raw - means_x[:4]) / scales_x[:4]

    # e) Compute “actual” predictions
    inp_act = torch.from_numpy(np.concatenate([x_adj_act_s, x_fixed_s])).float()
    with torch.no_grad():
        p1_s = model1(inp_act)[0].item()
        p2_s = model2(inp_act)[1].item()

    # f) Map back to raw + residuals
    p1_raw     = p1_s * scales_y[0] + mean_y[0]
    p2_raw     = p2_s * scales_y[1] + mean_y[1]
    resid1_act = abs(p1_raw - target1_raw)
    resid2_act = abs(p2_raw - target2_raw)
    results    = [(actual_x_adj_raw.copy(),
                   np.array([p1_raw, p2_raw]),
                   np.array([resid1_act, resid2_act]))]

    # g) Build search bounds
    mins_raw = x_adj_intervals_raw[[0,2,3], 0]
    maxs_raw = x_adj_intervals_raw[[0,2,3], 1]
    mins_s   = (mins_raw - means_x[[0,2,3]]) / scales_x[[0,2,3]]
    maxs_s   = (maxs_raw - means_x[[0,2,3]]) / scales_x[[0,2,3]]
    Q_min, Q_max = Q_bounds
    min_vec = np.concatenate([mins_s, [Q_min]])
    max_vec = np.concatenate([maxs_s, [Q_max]])

    # h) Scaled targets for fitness
    target1_s = (target1_raw - mean_y[0]) / scales_y[0]
    target2_s = (target2_raw - mean_y[1]) / scales_y[1]

    return {
        "model1": model1,
        "model2": model2,
        "means_x": means_x, "scales_x": scales_x,
        "mean_y": mean_y,   "scales_y": scales_y,
        "x_fixed_s": x_fixed_s,
        "results": results,
        "min_vec": min_vec, "max_vec": max_vec,
        "target_s": np.array([target1_s, target2_s])
    }


def fitness_two_outputs(
    candidate: np.ndarray,
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    x_fixed_s: np.ndarray,
    means_x: np.ndarray,
    scales_x: np.ndarray,
    target_s: np.ndarray
) -> float:
    """
    Sum of squared errors on the two scaled outputs.
    candidate = [x0_s, x2_s, x3_s, Q_raw]
    """
    x0_s, x2_s, x3_s, Q = candidate
    # reconstruct x1_s
    x0_raw = x0_s * scales_x[0] + means_x[0]
    x1_raw = Q    * x0_raw
    x1_s   = (x1_raw - means_x[1]) / scales_x[1]

    # full scaled input for the nets
    xs  = np.array([x0_s, x1_s, x2_s, x3_s])
    inp = torch.from_numpy(np.concatenate([xs, x_fixed_s])).float()
    with torch.no_grad():
        err1 = model1(inp)[0].item() - target_s[0]
        err2 = model2(inp)[1].item() - target_s[1]

    return err1*err1 + err2*err2


def unpack_two_output_solutions(
    candidates: list[np.ndarray],
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    means_x: np.ndarray,
    scales_x: np.ndarray,
    x_fixed_s: np.ndarray,
    scales_y: np.ndarray,
    mean_y: np.ndarray,
    target1_raw: float,
    target2_raw: float
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Inverse‐scale the top candidates and compute raw preds & residuals.
    """
    results = []
    for cand in candidates:
        x0_s, x2_s, x3_s, Q = cand
        # build raw adjustable inputs
        x0_raw = x0_s * scales_x[0] + means_x[0]
        x1_raw = Q    * x0_raw
        x2_raw = x2_s * scales_x[2] + means_x[2]
        x3_raw = x3_s * scales_x[3] + means_x[3]
        adj_raw = np.array([x0_raw, x1_raw, x2_raw, x3_raw])

        # scaled input again for nets
        x1_s = (x1_raw - means_x[1]) / scales_x[1]
        xs   = np.array([x0_s, x1_s, x2_s, x3_s])
        inp  = torch.from_numpy(np.concatenate([xs, x_fixed_s])).float()

        with torch.no_grad():
            p1_s = model1(inp)[0].item()
            p2_s = model2(inp)[1].item()

        p1_r   = p1_s * scales_y[0] + mean_y[0]
        p2_r   = p2_s * scales_y[1] + mean_y[1]
        res1   = abs(p1_r - target1_raw)
        res2   = abs(p2_r - target2_raw)

        results.append((adj_raw, np.array([p1_r, p2_r]), np.array([res1, res2])))

    return results



import numpy as np
from typing import List

def select_top_k_unique(
    fitnesses: List[float],
    candidates: List[np.ndarray],
    k: int,
    precision: int = 6
) -> List[np.ndarray]:
    """
    Select the top-k unique candidates by ascending fitness.

    Args:
        fitnesses:  List of fitness values (lower is better).
        candidates: List of candidate vectors (same length as fitnesses).
        k:          Number of unique candidates to return.
        precision:  Decimal places to round for uniqueness.

    Returns:
        List of up to k unique candidate vectors.
    """
    # pair & sort by fitness
    scored = sorted(zip(fitnesses, candidates), key=lambda t: t[0])

    unique = []
    seen   = set()
    for _, cand in scored:
        # round to avoid floating-point dupes
        key = tuple(np.round(cand, precision))
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
        if len(unique) >= k:
            break

    return unique
