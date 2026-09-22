"""
Evaluation: MSE, R², and parameter counts on the original target scale.
"""

import sys
import json
import pickle
import warnings
import numpy as np
import torch
import logging
from pathlib import Path
import mlflow

logging.basicConfig(level=logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config,
    denormalize_output,
    create_model_for_eval,
    get_flops_forward,
    get_function_definition,
    normalize_input,
    resolve_target,
)


def count_parameters_trainable(model):
    """Number of trainable parameters (requires_grad)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_total(model):
    """Total parameter count (trainable and frozen, for example KAN buffers)."""
    return sum(p.numel() for p in model.parameters())


def r2_score(y_true, y_pred):
    """R² on the original target scale."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--function_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    group = args.group
    function_name = args.function_name
    model_name = args.model_name
    model_file = args.model
    data_file = args.data
    output_file = args.output

    config = load_config()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    normalizers = data["normalizers"]
    _spec, dim, domain = resolve_target(config, function_name, group)
    if "domain" in data:
        domain = data["domain"]
    if "dim" in data and int(data["dim"]) != dim:
        raise ValueError(
            f"Dataset dim ({data['dim']}) does not match config dim ({dim}) for '{function_name}'."
        )

    is_2d = dim == 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model and load the weights
    model = create_model_for_eval(
        config,
        model_name,
        is_2d=is_2d,
        domain=domain if not is_2d else [domain[0], domain[1]],
    )
    if model is None:
        raise RuntimeError(f"Could not build model '{model_name}' (is pykan installed for KAN?)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            ckpt = torch.load(model_file, map_location=device, weights_only=True)
        except (pickle.UnpicklingError, Exception) as e:
            if isinstance(e, pickle.UnpicklingError) or "weights_only" in str(e) or "Unsupported class" in str(e):
                ckpt = torch.load(model_file, map_location=device, weights_only=False)
            else:
                raise
    # Require a state_dict (same format for MLP and KAN)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(
            f"Checkpoint {model_file} has no 'state_dict' key. "
            "Retrain so train.py saves a state_dict for both MLP and KAN."
        )
    state = ckpt["state_dict"]
    if not (isinstance(state, dict) and len(state) > 0):
        raise RuntimeError(f"Checkpoint {model_file}: 'state_dict' is empty or invalid.")
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    n_params_trainable = count_parameters_trainable(model)
    n_params_total = count_parameters_total(model)
    flops_forward = get_flops_forward(config, model, model_name, is_2d)
    out_norm = normalizers["output"]

    def predict(X):
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
            out = model(t)
            return out.cpu().numpy()

    # Predictions in normalized space
    pred_train = predict(X_train)
    pred_test = predict(X_test)

    # Map predictions back to the original scale
    y_train_orig = denormalize_output(y_train, out_norm)
    y_test_orig = denormalize_output(y_test, out_norm)
    pred_train_orig = denormalize_output(pred_train, out_norm)
    pred_test_orig = denormalize_output(pred_test, out_norm)

    # Metrics on the original scale
    mse_train = float(np.mean((np.asarray(y_train_orig) - np.asarray(pred_train_orig)) ** 2))
    mse_test = float(np.mean((np.asarray(y_test_orig) - np.asarray(pred_test_orig)) ** 2))
    r2_test = r2_score(y_test_orig, pred_test_orig)

    metrics = {
        "group": group,
        "function_name": function_name,
        "dim": dim,
        "model_name": model_name,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "r2_test": r2_test,
        "n_params": n_params_trainable,
        "n_params_trainable": n_params_trainable,
        "n_params_total": n_params_total,
    }
    if flops_forward is not None:
        metrics["flops_forward"] = int(flops_forward)

    run_id = json.loads(Path(model_file).with_name("training_info.json").read_text())["mlflow_run_id"]
    mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).resolve().parents[1] / 'mlflow.db'}")
    mlflow.set_experiment("mlp-kan-benchmark")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "mse_train": metrics["mse_train"],
            "mse_test": metrics["mse_test"],
            "r2_test": metrics["r2_test"],
            "n_params": metrics["n_params"],
        })

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Precompute plot data from this same loaded model
    out_dir = Path(output_file).parent
    x_min, x_max = float(domain[0]), float(domain[1])

    def _to_list(a):
        return np.asarray(a).tolist()

    if is_2d:
        n_plot = 80
        xx = np.linspace(x_min, x_max, n_plot)
        yy = np.linspace(x_min, x_max, n_plot)
        X_mesh = np.meshgrid(xx, yy)
        X_flat = np.stack([X_mesh[0].ravel(), X_mesh[1].ravel()], axis=1)
        X_flat_norm, _ = normalize_input(X_flat, domain)
        y_real_flat = np.asarray(get_function_definition(function_name)(X_flat)).flatten()
        y_pred_flat = denormalize_output(predict(X_flat_norm), out_norm).flatten()
        Z_real = y_real_flat.reshape(n_plot, n_plot)
        Z_pred = y_pred_flat.reshape(n_plot, n_plot)
        all_resid = np.concatenate([
            np.asarray(y_train_orig).flatten() - np.asarray(pred_train_orig).flatten(),
            np.asarray(y_test_orig).flatten() - np.asarray(pred_test_orig).flatten(),
        ])
        plot_data = {
            "is_2d": True,
            "xx": _to_list(xx),
            "yy": _to_list(yy),
            "Z_real": _to_list(Z_real),
            "Z_pred": _to_list(Z_pred),
            "all_resid": _to_list(all_resid),
        }
    else:
        x_grid_orig = np.linspace(x_min, x_max, 500).reshape(-1, 1)
        x_grid_norm, _ = normalize_input(x_grid_orig, domain)
        y_real_grid = np.asarray(get_function_definition(function_name)(x_grid_orig)).flatten()
        y_pred_grid = denormalize_output(predict(x_grid_norm), out_norm).flatten()
        y_train_flat = np.asarray(y_train_orig).flatten()
        y_test_flat = np.asarray(y_test_orig).flatten()
        pred_train_flat = np.asarray(pred_train_orig).flatten()
        pred_test_flat = np.asarray(pred_test_orig).flatten()
        x_train_orig = X_train[:, 0] * (x_max - x_min) / 2 + (x_min + x_max) / 2
        x_test_orig = X_test[:, 0] * (x_max - x_min) / 2 + (x_min + x_max) / 2
        resid_train = y_train_flat - pred_train_flat
        resid_test = y_test_flat - pred_test_flat
        all_resid = np.concatenate([resid_train, resid_test])
        plot_data = {
            "is_2d": False,
            "x_min": x_min,
            "x_max": x_max,
            "x_grid_orig": _to_list(x_grid_orig),
            "y_real_grid": _to_list(y_real_grid),
            "y_pred_grid": _to_list(y_pred_grid),
            "x_train_orig": _to_list(x_train_orig),
            "x_test_orig": _to_list(x_test_orig),
            "y_train_orig": _to_list(y_train_flat),
            "y_test_orig": _to_list(y_test_flat),
            "resid_train": _to_list(resid_train),
            "resid_test": _to_list(resid_test),
            "all_resid": _to_list(all_resid),
        }

    plot_data_path = out_dir / "plot_data.json"
    with open(plot_data_path, "w") as f:
        json.dump(plot_data, f, indent=0)


if __name__ == "__main__":
    main()
