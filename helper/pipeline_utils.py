import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


class RegressionDataset(Dataset):
    """
    PyTorch Dataset for regression tasks.

    Args:
        X (np.ndarray): Inputs, shape (N, D).
        Y (np.ndarray): Targets, shape (N, O).
        augment (bool): Whether to add Gaussian noise to inputs.
        noise_std (float): Standard deviation of Gaussian noise.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, augment: bool = False, noise_std: float = 0.0):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.Y[idx]


class Net(nn.Module):
    """
    Simple feedforward network with configurable hidden layers and activation.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dims (list[int]): Sizes of hidden layers.
        out_dim (int): Output dimension.
        activation_str (str): Activation function name: "relu", "sigmoid", or "softplus".
    """
    def __init__(self, in_dim: int, hidden_dims: list, out_dim: int, activation_str: str):
        super().__init__()
        act_map = {
            "relu":    nn.ReLU(inplace=True),
            "sigmoid": nn.Sigmoid(),
            "softplus": nn.Softplus()
        }
        act_fn = act_map.get(activation_str.lower())
        if act_fn is None:
            raise ValueError(f"Unsupported activation: {activation_str}")

        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(act_fn)
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_loss(model: nn.Module,
                 loader: DataLoader,
                 weights: torch.Tensor,
                 device: torch.device) -> float:
    """
    Compute weighted mean squared error over a DataLoader.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = ((preds - yb) ** 2 * weights).mean().item()
            batch_size = xb.size(0)
            total_loss += loss * batch_size
            total_samples += batch_size
    return total_loss / total_samples





def compute_nth_r2(
    model: nn.Module,
    loader: DataLoader,
    y_scaler: StandardScaler,
    device: torch.device,
    index: int
) -> float:
    """
    Compute R² score for the specified output index after inverse-scaling.

    Args:
        model: trained PyTorch model
        loader: DataLoader yielding (X, Y_scaled)
        y_scaler: fitted StandardScaler for Y
        device: torch.device for inference
        index: which output dimension to evaluate (0-based)
    """
    model.eval()
    preds_list, true_list = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()[:, index]
            true  = yb[:, index].cpu().numpy()
            preds_list.append(preds)
            true_list.append(true)

    y_pred = np.concatenate(preds_list)
    y_true = np.concatenate(true_list)

    # build full arrays to inverse-transform
    n_samples = y_true.shape[0]
    n_outputs = y_scaler.scale_.shape[0]
    full_pred = np.zeros((n_samples, n_outputs))
    full_true = np.zeros((n_samples, n_outputs))
    full_pred[:, index] = y_pred
    full_true[:, index] = y_true

    inv_pred = y_scaler.inverse_transform(full_pred)[:, index]
    inv_true = y_scaler.inverse_transform(full_true)[:, index]

    return r2_score(inv_true, inv_pred)











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
        candidate: array([x0_s, x2_s, x3_s, Q])
        model:     trained Net from pipeline_utils
        x_fixed_s: scaled fixed inputs (the last 10 dims)
        means_x:   x_scaler.mean_ (full 14-dim)
        scales_x:  x_scaler.scale_
        target_scaled: (target_raw - y_mean)/y_scale

    Returns:
        Squared error between predicted first output (scaled) and target_scaled.
    """
    x0_s, x2_s, x3_s, Q = candidate

    # reconstruct scaled x1 via Q * raw x0
    x0_raw = x0_s * scales_x[0] + means_x[0]
    x1_raw = Q * x0_raw
    x1_s   = (x1_raw - means_x[1]) / scales_x[1]

    # assemble full adjustable vector [x0_s, x1_s, x2_s, x3_s]
    x_s_full = np.array([x0_s, x1_s, x2_s, x3_s])
    inp = torch.from_numpy(np.concatenate([x_s_full, x_fixed_s])).float()

    model.eval()
    with torch.no_grad():
        pred0_s = model(inp)[0].item()

    return (pred0_s - target_scaled) ** 2
