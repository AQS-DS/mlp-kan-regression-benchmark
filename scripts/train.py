"""
Generic training script for MLP and KAN.
"""

import sys
import pickle
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging
from pathlib import Path
import mlflow

logging.basicConfig(level=logging.ERROR)

# Make local imports work
sys.path.insert(0, str(Path(__file__).parent))

from utils import set_seed, load_config, get_flops_forward, resolve_target


def get_batch_size_config(config, n_train):
    """
    Read batch_size from the config. Returns (DataLoader batch size, value to record).
    - batch_size "full" -> full batch; recorded value: "full"
    - a number >= n_train -> full batch; recorded value: "full"
    - a number < n_train -> mini-batch; recorded value: that number
    """
    raw = config['training'].get('batch_size', 32)
    if isinstance(raw, str) and str(raw).strip().lower() == 'full':
        return n_train, 'full'
    try:
        b = int(raw)
    except (TypeError, ValueError):
        b = 32
    if b >= n_train:
        return n_train, 'full'
    return b, b


class MLP(nn.Module):
    """Generic MLP for 1D and 2D inputs."""
    
    def __init__(self, layers, activation='tanh'):
        super(MLP, self).__init__()
        
        self.layers_list = nn.ModuleList()
        
        # Linear layers
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))
        
        # Activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers_list):
            x = layer(x)
            # No activation on the last layer
            if i < len(self.layers_list) - 1:
                x = self.activation(x)
        return x


def count_parameters(model):
    """Count trainable parameters (requires_grad)."""
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        return 0


def count_parameters_total(model):
    """Count all parameters (trainable and frozen)."""
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def create_mlp(config, is_2d=False):
    """Build an MLP from the config."""
    arch_key = 'mlp_2d' if is_2d else 'mlp_1d'
    arch_config = config['architectures'][arch_key]
    
    model = MLP(
        layers=arch_config['layers'],
        activation=arch_config['activation']
    )
    return model


def create_kan(config, is_2d=False, domain=None, ckpt_path='./model'):
    """Build a KAN from the config."""
    try:
        from kan import KAN
    except ImportError:
        raise ImportError("pykan is not installed. Install it with: pip install pykan")
    
    arch_key = 'kan_2d' if is_2d else 'kan_1d'
    arch_config = config['architectures'][arch_key]
    
    # Inputs are already scaled to [-1, 1]
    grid_range = [-1.0, 1.0]
    
    # Pick the device
    device_config = config['training'].get('device', 'cpu')
    if device_config == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Create the checkpoint directory if needed
    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
    
    model = KAN(
        width=arch_config['width'],
        grid=arch_config['grid'],
        k=arch_config['k'],
        grid_range=grid_range,
        seed=config['training']['seed'],
        device=device,
        ckpt_path=ckpt_path
    )
    return model


def train_mlp(model, X_train, y_train, X_test, y_test, config):
    """Train an MLP with mini-batches (Adam or LBFGS)."""
    print("Starting MLP training...")
    
    # Pick the device
    device_config = config['training'].get('device', 'cpu')
    if device_config == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    n_train = len(X_train)
    batch_size_effective, batch_size_report = get_batch_size_config(config, n_train)
    print(f"Device: {device}, batch_size: {batch_size_report}")
    model = model.to(device)
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_effective, shuffle=True, drop_last=False)
    
    criterion = nn.MSELoss()
    
    optimizer_type = config['training'].get('optimizer', 'Adam').upper()
    learning_rate = config['training'].get('learning_rate', 0.001)
    
    if optimizer_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    epochs = config['training']['epochs']
    print(f"Training for {epochs} epochs with {optimizer_type} ({len(train_loader)} batches/epoch)...")
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            if optimizer_type == 'ADAM':
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            else:  # LBFGS
                def closure():
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
            epoch_loss += loss.item()
            n_batches += 1
        avg_epoch_loss = epoch_loss / n_batches if n_batches else epoch_loss
        loss_history.append(avg_epoch_loss)
        if device == 'cuda':
            torch.cuda.synchronize()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor).item()
            print(f"\rEpoch {epoch + 1}/{epochs} | train_loss: {avg_epoch_loss:.2e} | test_loss: {test_loss:.2e}   ", end="", flush=True)
    
    print()
    if device == 'cuda':
        torch.cuda.synchronize()
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds")
    return model, training_time, loss_history, batch_size_report


def train_kan(model, X_train, y_train, X_test, y_test, config):
    """Train a KAN with mini-batches (same loop as the MLP)."""
    print("Starting KAN training...")
    
    device_config = config['training'].get('device', 'cpu')
    if device_config == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    n_train = len(X_train)
    batch_size_effective, batch_size_report = get_batch_size_config(config, n_train)
    print(f"Device: {device}, batch_size: {batch_size_report}")
    
    model = model.to(device)
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_effective, shuffle=True, drop_last=False)
    
    criterion = nn.MSELoss()
    optimizer_type = config['training'].get('optimizer', 'Adam').upper()
    learning_rate = config['training'].get('learning_rate', 0.001)
    
    if optimizer_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    epochs = config['training']['epochs']
    print(f"Training for {epochs} epochs with {optimizer_type} ({len(train_loader)} batches/epoch)...")
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            if optimizer_type == 'ADAM':
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            else:
                def closure():
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
            epoch_loss += loss.item()
            n_batches += 1
        avg_epoch_loss = epoch_loss / n_batches if n_batches else epoch_loss
        loss_history.append(avg_epoch_loss)
        if device == 'cuda':
            torch.cuda.synchronize()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor).item()
            print(f"\rEpoch {epoch + 1}/{epochs} | train_loss: {avg_epoch_loss:.2e} | test_loss: {test_loss:.2e}   ", end="", flush=True)
    
    print()
    if device == 'cuda':
        torch.cuda.synchronize()
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds")
    return model, training_time, loss_history, batch_size_report


def sanitize_metadata_for_torch_save(obj):
    """
    Recursively convert numpy/torch values to plain Python so
    torch.load(..., weights_only=True) does not fail on numpy scalars.
    Use this on metadata (normalizers, domain) only, not on the state_dict.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: sanitize_metadata_for_torch_save(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_metadata_for_torch_save(x) for x in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.generic)):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return obj


def convert_to_serializable(obj):
    """Convert numpy/torch values to plain Python for JSON."""
    if obj is None:
        return None
    elif isinstance(obj, (np.ndarray, np.generic)):
        if obj.size == 1:
            return float(obj.item())
        else:
            return [float(x) for x in obj.tolist()]
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        else:
            arr = obj.detach().cpu().numpy()
            return [float(x) for x in arr.tolist()]
    elif isinstance(obj, list):
        result = []
        for x in obj:
            converted = convert_to_serializable(x)
            if isinstance(converted, list):
                result.extend(converted)
            else:
                result.append(converted)
        return result
    elif isinstance(obj, (int, float)):
        return float(obj)
    else:
        # Fall back to float when possible
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--function_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    group = args.group
    function_name = args.function_name
    model_name = args.model_name
    data_file = args.data
    output_file = args.output
    
    # Load configuration
    config = load_config()
    
    # Set the random seed
    set_seed(config["training"]["seed"])
    
    # Create the output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Load the dataset
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    normalizers = data['normalizers']
    domain = data['domain']

    _spec, dim, _domain = resolve_target(config, function_name, group)
    if "dim" in data and int(data["dim"]) != dim:
        raise ValueError(
            f"Dataset dim ({data['dim']}) does not match config dim ({dim}) for '{function_name}'."
        )
    is_2d = dim == 2

    print(f"Setting up {model_name.upper()} for {group}/{function_name}...")
    
    # Build the model
    if model_name == 'mlp':
        print("Building the MLP...")
        model = create_mlp(config, is_2d=is_2d)
        n_params = count_parameters(model)
        n_params_total = count_parameters_total(model)
        flops_fwd = get_flops_forward(config, model, 'mlp', is_2d)
        print(f"Trainable parameters: {n_params} | total: {n_params_total} | FLOPs/forward: {flops_fwd}")
        
        model, training_time, loss_history, batch_size_report = train_mlp(model, X_train, y_train, X_test, y_test, config)
        
        print("Saving the model...")
        save_dict = {
            'state_dict': model.state_dict(),
            'model_type': 'mlp',
            'group': group,
            'dim': dim,
            'function_name': function_name,
            'normalizers': sanitize_metadata_for_torch_save(normalizers),
            'domain': sanitize_metadata_for_torch_save(domain),
            'n_params': int(n_params),
            'training_time': float(training_time),
            'architecture': config['architectures']['mlp_2d' if is_2d else 'mlp_1d']
        }
        if flops_fwd is not None:
            save_dict['flops_forward'] = int(flops_fwd)
        torch.save(save_dict, output_file)
        
        # Training record for the JSON file
        # Make loss_history JSON-serializable
        loss_history_serializable = convert_to_serializable(loss_history) if loss_history else []
        if not isinstance(loss_history_serializable, list):
            loss_history_serializable = [loss_history_serializable]
        
        training_info = {
            'group': group,
            'dim': dim,
            'function_name': function_name,
            'model_name': model_name,
            'n_params': int(n_params),
            'training_time': float(training_time),
            'epochs': config['training']['epochs'],
            'batch_size': batch_size_report,
            'device': config['training'].get('device', 'cpu'),
            'loss_history': loss_history_serializable,
            'final_loss': float(loss_history_serializable[-1]) if loss_history_serializable else None,
            'initial_loss': float(loss_history_serializable[0]) if loss_history_serializable else None,
            'architecture': config['architectures']['mlp_2d' if is_2d else 'mlp_1d']
        }
        if flops_fwd is not None:
            training_info['flops_forward'] = int(flops_fwd)
    
    elif model_name == 'kan':
        print("Building the KAN...")
        
        # One checkpoint directory per experiment
        # pykan writes its own files here
        result_dir = Path(output_file).parent
        kan_ckpt_path = str(result_dir / 'kan_checkpoints')
        
        model = create_kan(config, is_2d=is_2d, domain=domain, ckpt_path=kan_ckpt_path)
        n_params = count_parameters(model)
        n_params_total = count_parameters_total(model)
        flops_fwd = get_flops_forward(config, model, 'kan', is_2d)
        print(f"Trainable parameters: {n_params} | total: {n_params_total} | FLOPs/forward: {flops_fwd}")
        
        model, training_time, loss_history, batch_size_report = train_kan(model, X_train, y_train, X_test, y_test, config)
        
        print("Saving the model...")
        model_dict = {
            'model_type': 'kan',
            'group': group,
            'dim': dim,
            'function_name': function_name,
            'normalizers': sanitize_metadata_for_torch_save(normalizers),
            'domain': sanitize_metadata_for_torch_save(domain),
            'n_params': int(n_params),
            'training_time': float(training_time),
            'architecture': config['architectures']['kan_2d' if is_2d else 'kan_1d'],
            'ckpt_path': kan_ckpt_path,
            'is_2d': is_2d,
            'state_dict': model.state_dict(),
        }
        if flops_fwd is not None:
            model_dict['flops_forward'] = int(flops_fwd)
        torch.save(model_dict, output_file)
        
        # Training record for the JSON file
        # Make loss_history JSON-serializable
        loss_history_serializable = convert_to_serializable(loss_history) if loss_history else []
        if not isinstance(loss_history_serializable, list):
            loss_history_serializable = [loss_history_serializable]
        
        training_info = {
            'group': group,
            'dim': dim,
            'function_name': function_name,
            'model_name': model_name,
            'n_params': int(n_params),
            'training_time': float(training_time),
            'epochs': config['training']['epochs'],
            'batch_size': batch_size_report,
            'device': config['training'].get('device', 'cpu'),
            'kan_lambda': float(config['training'].get('kan_lambda', 0.0)),
            'loss_history': loss_history_serializable,
            'final_loss': float(loss_history_serializable[-1]) if loss_history_serializable else None,
            'initial_loss': float(loss_history_serializable[0]) if loss_history_serializable else None,
            'architecture': config['architectures']['kan_2d' if is_2d else 'kan_1d'],
            'ckpt_path': str(kan_ckpt_path)
        }
        if flops_fwd is not None:
            training_info['flops_forward'] = int(flops_fwd)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

    mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).resolve().parents[1] / 'mlflow.db'}")
    mlflow.set_experiment("mlp-kan-benchmark")
    with mlflow.start_run(run_name=f"{group}/{function_name}/{model_name}") as run:
        mlflow.log_params({
            "group": group,
            "dim": dim,
            "function_name": function_name,
            "model_name": model_name,
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "optimizer": config["training"]["optimizer"],
            "seed": config["training"]["seed"],
            "architecture": json.dumps(training_info["architecture"]),
        })
        for step, loss in enumerate(training_info["loss_history"]):
            mlflow.log_metric("loss", loss, step=step)
        mlflow.log_metric("training_time", training_info["training_time"])
        training_info["mlflow_run_id"] = run.info.run_id

    # Write the training record
    training_json_path = Path(output_file).parent / 'training_info.json'
    with open(training_json_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"Training record saved to: {training_json_path}")
    
    print("Done.")

if __name__ == "__main__":
    main()
