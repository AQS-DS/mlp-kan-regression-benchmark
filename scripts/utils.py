"""
Shared helpers for data generation and the training pipeline.
"""

import os
import numpy as np
import tempfile
import torch
import logging
from scipy.special import airy, eval_hermite
from pathlib import Path

logging.basicConfig(level=logging.ERROR)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_input(X, domain):
    """
    Scale each input dimension to [-1, 1].

    Args:
        X: Input array (n_samples, n_dims)
        domain: [min, max] for 1D, or the same range applied to every 2D axis

    Returns:
        X_norm: Scaled array
        normalizer: Dict with min/max per dimension
    """
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_dims = X.shape[1]
    X_norm = np.zeros_like(X)
    normalizer = {}
    
    if n_dims == 1:
        # 1D: domain is [min, max]
        x_min, x_max = domain[0], domain[1]
        X_norm[:, 0] = 2 * (X[:, 0] - x_min) / (x_max - x_min) - 1
        normalizer['x'] = {'min': x_min, 'max': x_max}
    else:
        # 2D: the same [min, max] is applied to both axes
        x_min, x_max = domain[0], domain[1]
        for dim in range(n_dims):
            X_norm[:, dim] = 2 * (X[:, dim] - x_min) / (x_max - x_min) - 1
            normalizer[f'dim_{dim}'] = {'min': x_min, 'max': x_max}
    
    return X_norm, normalizer


def normalize_output(y, threshold=50):
    """
    Standardize the target when max(|y|) exceeds the threshold.

    Args:
        y: Target array
        threshold: Absolute-value cutoff for standardization

    Returns:
        y_norm: Scaled target
        normalizer: Dict with mean/std, or normalized=False when left unchanged
    """
    y = np.array(y).flatten()
    y_max_abs = np.max(np.abs(y))
    
    if y_max_abs > threshold:
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std < 1e-10:
            y_std = 1.0
        y_norm = (y - y_mean) / y_std
        normalizer = {'mean': y_mean, 'std': y_std, 'normalized': True}
    else:
        y_norm = y
        normalizer = {'normalized': False}
    
    return y_norm, normalizer


def denormalize_output(y_norm, normalizer):
    """
    Map a normalized target back to the original scale.

    Args:
        y_norm: Normalized array
        normalizer: Dict produced by normalize_output

    Returns:
        y: Array on the original scale
    """
    y_norm = np.array(y_norm).flatten()
    
    if normalizer.get('normalized', False):
        y = y_norm * normalizer['std'] + normalizer['mean']
    else:
        y = y_norm
    
    return y


def apply_output_normalizer(y, normalizer: dict):
    """
    Apply the output normalizer fitted on the training targets.
    The test set must use only those training parameters.
    """
    y = np.array(y).flatten()
    if normalizer.get("normalized", False):
        std = float(normalizer.get("std", 1.0))
        if abs(std) < 1e-12:
            std = 1.0
        mean = float(normalizer.get("mean", 0.0))
        return (y - mean) / std
    return y


def resolve_target(config, function_name, group=None):
    """
    Look up one target in config['targets'].

    Dimensionality comes from the target (1 or 2), not from the group name.
    When group is given, it must match the configured group.
    """
    targets = config.get("targets") or {}
    if function_name not in targets:
        known = ", ".join(sorted(targets)) or "(none)"
        raise ValueError(f"Function '{function_name}' is not in config targets. Known: {known}")
    spec = targets[function_name]
    if group is not None and spec.get("group") != group:
        raise ValueError(
            f"Group mismatch for '{function_name}': path has '{group}', config has '{spec.get('group')}'."
        )
    dim = int(spec["dim"])
    if dim not in (1, 2):
        raise ValueError(f"Target '{function_name}' dim must be 1 or 2, got {dim}.")
    domain = spec["domain"]
    if not isinstance(domain, (list, tuple)) or len(domain) != 2:
        raise ValueError(f"Target '{function_name}' domain must be [min, max].")
    return spec, dim, [float(domain[0]), float(domain[1])]


def get_function_definition(function_name):
    """
    Return the callable for a function name.

    Returns:
        func: Maps X (n_samples, n_dims) to y (n_samples,)
    """
    if function_name == 'airy_ai':
        # Ai(x) is a solution of the Airy differential equation y'' - x*y = 0.
        return lambda X: airy(X[:, 0])[0]
    if function_name == 'hermite_gaussian':
        # An unnormalized 2D harmonic-oscillator mode: H_3(x) H_2(y) exp(-(x²+y²)/2).
        # Distinct nodal lines and Gaussian tails make this a useful 2D regression target.
        return lambda X: (
            eval_hermite(3, X[:, 0])
            * eval_hermite(2, X[:, 1])
            * np.exp(-0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2))
        )

    raise ValueError(f"Function '{function_name}' has no definition")


def generate_dataset(function_name, dim, domain, n_samples=1000, train_split=0.8, seed=42):
    """
    Build a synthetic dataset for one target function.

    Args:
        function_name: Function name
        dim: Input dimension, 1 or 2
        domain: [min, max] for 1D, applied to both axes in 2D
        n_samples: Total number of samples
        train_split: Fraction used for training
        seed: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, normalizers
    """
    set_seed(seed)
    
    # Input dimension is a property of the target
    is_2d = int(dim) == 2
    n_dims = 2 if is_2d else 1
    
    # Uniform samples
    if n_dims == 1:
        X = np.random.uniform(domain[0], domain[1], size=(n_samples, 1))
    else:
        X = np.random.uniform(domain[0], domain[1], size=(n_samples, 2))
    
    # Evaluate the target
    func = get_function_definition(function_name)
    y = func(X)
    y = y.reshape(-1, 1)
    
    # Train/test split
    n_train = int(n_samples * train_split)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # Normalize inputs
    X_train_norm, input_norm = normalize_input(X_train, domain)
    X_test_norm, _ = normalize_input(X_test, domain)
    
    # Normalize targets
    # - fit the output normalizer on train only
    # - transform test with the training parameters (do not refit)
    y_train_norm, output_norm = normalize_output(y_train, threshold=50)
    y_test_norm = apply_output_normalizer(y_test, output_norm)
    
    # Store the normalizers
    normalizers = {
        'input': input_norm,
        'output': output_norm
    }
    
    return (
        X_train_norm.astype(np.float32),
        X_test_norm.astype(np.float32),
        y_train_norm.astype(np.float32).reshape(-1, 1),
        np.asarray(y_test_norm, dtype=np.float32).reshape(-1, 1),
        normalizers
    )


def load_config(config_path="config.yaml"):
    """Load the YAML config."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- FLOPs (one forward pass, one sample) ---

# Approximate activation cost (FLOPs per element)
_ACTIVATION_FLOPS = {"tanh": 8, "relu": 1, "sigmoid": 8}


def flops_mlp_forward(layers, activation="tanh"):
    """
    FLOPs for one MLP forward pass (batch=1).
    Linear(in, out): 2*in*out (multiply-add). Activation: _ACTIVATION_FLOPS per hidden unit.
    """
    if not layers or len(layers) < 2:
        return 0
    flops_linear = sum(2 * layers[i] * layers[i + 1] for i in range(len(layers) - 1))
    # Hidden units (everything except input and output)
    n_hidden = sum(layers[1:-1])
    act_flops = _ACTIVATION_FLOPS.get(activation.lower(), 8)
    return int(flops_linear + n_hidden * act_flops)


def flops_kan_forward(model, is_2d=False):
    """
    FLOPs for one KAN forward pass (batch=1), via the profiler.
    Requires fvcore. Returns None when it is not installed.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        logging.warning("fvcore is not installed; KAN FLOPs were not computed. pip install fvcore")
        return None
    device = next(model.parameters()).device
    batch_size = 1
    if is_2d:
        dummy = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
    else:
        dummy = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
    try:
        flop_counter = FlopCountAnalysis(model, (dummy,))
        total = flop_counter.total()
        return int(total) if total is not None else None
    except Exception as e:
        logging.warning("FlopCountAnalysis failed for KAN: %s", e)
        return None


def get_flops_forward(config, model, model_name, is_2d):
    """
    FLOPs for one forward pass of an already built model.
    model_name is 'mlp' or 'kan'. Returns int or None.
    """
    if model_name == "mlp":
        arch_key = "mlp_2d" if is_2d else "mlp_1d"
        arch = config["architectures"][arch_key]
        return flops_mlp_forward(arch["layers"], arch.get("activation", "tanh"))
    if model_name == "kan":
        return flops_kan_forward(model, is_2d=is_2d)
    return None


# --- Models for evaluation (same architecture as training) ---

def _create_mlp_module(config, is_2d=False):
    """Build an MLP from the config so a saved state_dict can be loaded."""
    import torch.nn as nn
    arch_key = "mlp_2d" if is_2d else "mlp_1d"
    arch = config["architectures"][arch_key]
    layers = arch["layers"]
    act = arch.get("activation", "tanh")
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
    act_fn = activations.get(act, nn.Tanh)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers_list = nn.ModuleList()
            for i in range(len(layers) - 1):
                self.layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            self.activation = act_fn()

        def forward(self, x):
            for i, layer in enumerate(self.layers_list):
                x = layer(x)
                if i < len(self.layers_list) - 1:
                    x = self.activation(x)
            return x

    return MLP()


def _create_kan_module(config, is_2d=False, domain=None):
    """Build a KAN from the config so a saved state_dict can be loaded."""
    arch_key = "kan_2d" if is_2d else "kan_1d"
    arch = config["architectures"][arch_key]
    width = arch["width"]
    grid = arch["grid"]
    k = arch["k"]
    # Inputs stored in the pickle are scaled to [-1, 1]
    grid_range = [-1.0, 1.0]
    try:
        from kan import KAN
        # Temporary ckpt_path so pykan does not create ./model or ./models in the project
        eval_ckpt = os.path.join(tempfile.gettempdir(), "pykan_eval_ckpt")
        os.makedirs(eval_ckpt, exist_ok=True)
        model = KAN(
            width=width,
            grid=grid,
            k=k,
            grid_range=grid_range,
            seed=config["training"].get("seed", 42),
            ckpt_path=eval_ckpt,
        )
        return model
    except Exception:
        return None


def create_model_for_eval(config, model_name, is_2d=False, domain=None):
    """
    Build an MLP or KAN from the config so a state_dict can be loaded.
    model_name is 'mlp' or 'kan'.
    """
    if model_name == "mlp":
        return _create_mlp_module(config, is_2d=is_2d)
    if model_name == "kan":
        return _create_kan_module(config, is_2d=is_2d, domain=domain)
    raise ValueError(f"model_name must be 'mlp' or 'kan', got: {model_name}")
