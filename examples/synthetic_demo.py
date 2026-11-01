"""Train a tiny synthetic model and exercise the existing single-output inversion helpers."""

import argparse
import json
from pathlib import Path
import pickle
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Invert-PropOnly-saveCSV"))

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from helper.pipeline_utils import Net, RegressionDataset, compute_loss, compute_nth_r2
from Helper_invert_1d import fitness_s, initialize_inversion, unpack_qpso_solutions


def run(output: Path) -> dict:
    rng = np.random.default_rng(7)
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # Match the existing helper's 14 inputs and six outputs, without reading project data.
    x = rng.uniform(0.5, 2.0, (64, 14))
    x[:, 1] = rng.uniform(0.5, 1.5, 64) * x[:, 0]
    y = x[:, :6] + 0.2 * x[:, 8:14]
    sx, sy = StandardScaler().fit(x[:48]), StandardScaler().fit(y[:48])
    train = DataLoader(RegressionDataset(sx.transform(x[:48]), sy.transform(y[:48])), batch_size=16, shuffle=True)
    test = DataLoader(RegressionDataset(sx.transform(x[48:]), sy.transform(y[48:])), batch_size=16)
    model = Net(14, [16], 6, "relu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(40):
        model.train()
        for xb, yb in train:
            optimizer.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            optimizer.step()
    test_loss = compute_loss(model, test, torch.ones(6), torch.device("cpu"))
    test_r2 = compute_nth_r2(model, test, sy, torch.device("cpu"), 0)
    assert np.isfinite([test_loss, test_r2]).all()
    # The legacy loader uses pickle. Only load scalers created by this process.
    for filename, scaler in [("synthetic-x-scaler.pkl", sx), ("synthetic-y-scaler.pkl", sy)]:
        with (output / filename).open("wb") as stream:
            pickle.dump(scaler, stream)
    checkpoint = output / "synthetic-model.pth"
    torch.save(model.state_dict(), checkpoint)
    state = initialize_inversion(
        str(output / "synthetic-x-scaler.pkl"), str(output / "synthetic-y-scaler.pkl"),
        str(checkpoint), [16], "relu", x[48, 4:], x[48, :4], float(y[48, 0]),
        np.array([[0.5, 2.0], [0.25, 3.0], [0.5, 2.0], [0.5, 2.0]]), (0.5, 1.5),
    )
    probe = torch.from_numpy(sx.transform(x[48:])).float()
    with torch.no_grad():
        assert torch.allclose(model(probe), state["model"](probe))
    candidates = rng.uniform(state["min_vec"], state["max_vec"], (128, 4))
    scores = [fitness_s(c, state["model"], state["x_fixed_s"], state["means_x"],
                        state["scales_x"], state["target_scaled"]) for c in candidates]
    best_index = int(np.argmin(scores))
    best = candidates[best_index]
    solutions = unpack_qpso_solutions([best], state["means_x"], state["scales_x"],
                                     state["x_fixed_s"], state["model"], sy, float(y[48, 0]))
    adjustable, predicted, residual = solutions[0]
    assert adjustable.shape == (4,) and predicted.shape == (6,)
    assert np.isfinite(adjustable).all() and np.isfinite(predicted).all()
    assert np.isclose(adjustable[1] / adjustable[0], best[3])
    assert np.isclose(residual, np.sqrt(scores[best_index]) * sy.scale_[0], rtol=1e-4, atol=1e-6)
    report = {"synthetic": True, "seed": 7, "training_rows": 48, "test_rows": 16, "epochs": 40,
              "candidate_count": len(candidates), "search": "bounded seeded random search",
              "synthetic_test_mse_scaled": test_loss, "synthetic_first_target_r2": test_r2,
              "synthetic_inverse_absolute_residual": float(residual),
              "checks": "training; held-out helpers; checkpoint round trip; inverse scaling and Q reconstruction",
              "scope": "Toy workflow only. No existing data, scalers or checkpoints are read. No field-performance or optimality claim."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for the synthetic checkpoint, scalers and report.")
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = run(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="frack-synthetic-") as folder:
            report = run(Path(folder))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
