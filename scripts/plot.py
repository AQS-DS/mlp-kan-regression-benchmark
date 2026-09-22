"""
Plots: loss, target vs prediction, and residuals on the original scale.
Supports 1D curves and 2D contours.
Writes one combined plots.png and individual files under plots/, using the same draw helpers.
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import mlflow

logging.basicConfig(level=logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))


def _metrics_text(metrics):
    """Metric text for the overlay box. Returns None when metrics are missing."""
    if not metrics:
        return None
    mse_test = metrics.get("mse_test", None)
    r2 = metrics.get("r2_test", None)
    n_params = metrics.get("n_params", None)
    parts = []
    if mse_test is not None:
        parts.append(f"MSE_test = {mse_test:.2e}")
    if r2 is not None:
        parts.append(f"R²_test = {r2:.4f}")
    if n_params is not None:
        parts.append(f"n_params = {n_params}")
    return "\n".join(parts) if parts else None


def _add_metrics_box(ax, metrics):
    """Draw the metric box in the upper-left corner of the axes."""
    txt = _metrics_text(metrics)
    if not txt:
        return
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="gray", linewidth=0.8),
    )


# ---------- 1D draw helpers (axes plus precomputed data) ----------


def draw_loss(ax, training_info):
    """Draw the loss curve."""
    has_loss = training_info and training_info.get("loss_history")
    if has_loss:
        loss_hist = training_info["loss_history"]
        steps = np.arange(1, len(loss_hist) + 1)
        ax.plot(steps, loss_hist, color="C0", linewidth=0.8)
        ax.set_xlabel("Epoch / step")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Training curve")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Training curve")


def draw_real_vs_pred_1d(ax, x_grid_orig, y_real_grid, y_pred_grid, x_train_orig, x_test_orig,
                         y_train_orig, y_test_orig, x_min, x_max, metrics=None):
    """Draw the target and the prediction. Adds the metric box when metrics are provided."""
    ax.plot(x_grid_orig.flatten(), y_real_grid, label="Target", color="C0", linewidth=1.5)
    ax.plot(x_grid_orig.flatten(), y_pred_grid, label="Prediction", color="C1", linewidth=1, linestyle="--")
    ax.scatter(x_train_orig, y_train_orig, s=8, alpha=0.5, label="Train", color="C0")
    ax.scatter(x_test_orig, y_test_orig, s=8, alpha=0.5, label="Test", color="C2")
    ax.set_xlabel("x (original scale)")
    ax.set_ylabel("y (original scale)")
    ax.set_title("Target vs prediction")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_metrics_box(ax, metrics)


def draw_residuals_1d(ax, x_train_orig, x_test_orig, resid_train, resid_test):
    """Draw residuals against x."""
    ax.scatter(x_train_orig, resid_train, s=12, alpha=0.6, label="Train", color="C0")
    ax.scatter(x_test_orig, resid_test, s=12, alpha=0.6, label="Test", color="C2")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xlabel("x (original scale)")
    ax.set_ylabel("Residual (original scale)")
    ax.set_title("Residuals vs x")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def draw_residuals_hist(ax, all_resid):
    """Draw the residual histogram."""
    ax.hist(all_resid, bins=min(30, max(10, len(all_resid) // 20)), color="C0", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual histogram")


# ---------- 2D draw helpers ----------


def draw_contour_real_2d(ax, X_mesh, Z_real):
    """Draw the 2D target contour."""
    cf = ax.contourf(X_mesh[0], X_mesh[1], Z_real, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Target (original scale)")


def draw_contour_pred_2d(ax, X_mesh, Z_pred):
    """Draw the 2D prediction contour."""
    cf = ax.contourf(X_mesh[0], X_mesh[1], Z_pred, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prediction (original scale)")


def _load_plot_data(plot_data_path):
    """Load plot_data.json written by evaluate and return numpy arrays."""
    with open(plot_data_path, "r") as f:
        raw = json.load(f)
    is_2d = raw["is_2d"]
    if is_2d:
        xx = np.array(raw["xx"])
        yy = np.array(raw["yy"])
        return {
            "is_2d": True,
            "X_mesh": np.meshgrid(xx, yy),
            "Z_real": np.array(raw["Z_real"]),
            "Z_pred": np.array(raw["Z_pred"]),
            "all_resid": np.array(raw["all_resid"]),
        }
    return {
        "is_2d": False,
        "x_min": raw["x_min"],
        "x_max": raw["x_max"],
        "x_grid_orig": np.array(raw["x_grid_orig"]),
        "y_real_grid": np.array(raw["y_real_grid"]),
        "y_pred_grid": np.array(raw["y_pred_grid"]),
        "x_train_orig": np.array(raw["x_train_orig"]),
        "x_test_orig": np.array(raw["x_test_orig"]),
        "y_train_orig": np.array(raw["y_train_orig"]),
        "y_test_orig": np.array(raw["y_test_orig"]),
        "resid_train": np.array(raw["resid_train"]),
        "resid_test": np.array(raw["resid_test"]),
        "all_resid": np.array(raw["all_resid"]),
    }


# ---------- Save the combined figure and the individual PNGs ----------


def save_plots_1d(group, function_name, model_name, data_1d, metrics, training_info,
                  combined_path, plots_dir, dpi=150):
    """Save the combined figure and four individual PNGs, reusing draw_*."""
    d = data_1d
    x_min, x_max = d["x_min"], d["x_max"]

    # Combined 2x2, same helpers as the individual files
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    draw_loss(axes[0, 0], training_info)
    draw_real_vs_pred_1d(
        axes[0, 1],
        d["x_grid_orig"], d["y_real_grid"], d["y_pred_grid"],
        d["x_train_orig"], d["x_test_orig"], d["y_train_orig"], d["y_test_orig"],
        x_min, x_max, metrics=metrics,
    )
    draw_residuals_1d(axes[1, 0], d["x_train_orig"], d["x_test_orig"], d["resid_train"], d["resid_test"])
    draw_residuals_hist(axes[1, 1], d["all_resid"])
    fig.suptitle(f"{group} — {function_name} — {model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    # Individual files, reusing the same helpers
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_loss(ax, training_info)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_real_vs_pred_1d(
        ax, d["x_grid_orig"], d["y_real_grid"], d["y_pred_grid"],
        d["x_train_orig"], d["x_test_orig"], d["y_train_orig"], d["y_test_orig"],
        x_min, x_max, metrics=metrics,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "real_vs_pred.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_residuals_1d(ax, d["x_train_orig"], d["x_test_orig"], d["resid_train"], d["resid_test"])
    plt.tight_layout()
    plt.savefig(plots_dir / "residuals.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_residuals_hist(ax, d["all_resid"])
    plt.tight_layout()
    plt.savefig(plots_dir / "residuals_hist.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def save_plots_2d(group, function_name, model_name, data_2d, metrics, training_info,
                  combined_path, plots_dir, dpi=150):
    """Save the combined figure and four individual PNGs for 2D."""
    d = data_2d
    X_mesh = d["X_mesh"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    draw_loss(axes[0, 0], training_info)
    draw_contour_real_2d(axes[0, 1], X_mesh, d["Z_real"])
    draw_contour_pred_2d(axes[1, 0], X_mesh, d["Z_pred"])
    draw_residuals_hist(axes[1, 1], d["all_resid"])
    # Metrics only on the prediction panel
    _add_metrics_box(axes[1, 0], metrics)
    fig.suptitle(f"{group} — {function_name} — {model_name} (2D)", fontsize=12)
    plt.tight_layout()
    plt.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_loss(ax, training_info)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    draw_contour_real_2d(ax, X_mesh, d["Z_real"])
    plt.tight_layout()
    plt.savefig(plots_dir / "real_vs_pred.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    draw_contour_pred_2d(ax, X_mesh, d["Z_pred"])
    _add_metrics_box(ax, metrics)
    plt.tight_layout()
    plt.savefig(plots_dir / "pred.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_residuals_hist(ax, d["all_resid"])
    plt.tight_layout()
    plt.savefig(plots_dir / "residuals_hist.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--function_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--plot-data", required=True, help="Path to plot_data.json written by evaluate")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--training-info", required=True, help="Path to training_info.json")
    parser.add_argument("--output-dir", required=True, help="Output directory: results/{group}/{function_name}/{model_name}")
    args = parser.parse_args()

    group = args.group
    function_name = args.function_name
    model_name = args.model_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_data = _load_plot_data(args.plot_data)
    with open(args.metrics, "r") as f:
        metrics = json.load(f)
    training_info = None
    if Path(args.training_info).exists():
        with open(args.training_info, "r") as f:
            training_info = json.load(f)

    combined_path = output_dir / "plots.png"
    plots_dir = output_dir / "plots"

    if plot_data["is_2d"]:
        save_plots_2d(group, function_name, model_name, plot_data, metrics, training_info,
                     combined_path, plots_dir)
    else:
        save_plots_1d(group, function_name, model_name, plot_data, metrics, training_info,
                     combined_path, plots_dir)

    run_id = (training_info or {}).get("mlflow_run_id")
    if not run_id:
        raise RuntimeError("training_info.json has no mlflow_run_id. Train the model first.")
    mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).resolve().parents[1] / 'mlflow.db'}")
    mlflow.set_experiment("mlp-kan-benchmark")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(combined_path))
        mlflow.log_artifacts(str(plots_dir), artifact_path="plots")


if __name__ == "__main__":
    main()
