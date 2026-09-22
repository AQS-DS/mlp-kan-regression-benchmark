"""
Generate a synthetic dataset (Snakemake wrapper).
"""

import sys
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from utils import generate_dataset, load_config, resolve_target, set_seed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--function_name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    group = args.group
    function_name = args.function_name
    output_file = args.output

    config = load_config()
    _spec, dim, domain = resolve_target(config, function_name, group)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    n_samples = config["training"]["n_samples"]
    train_split = config["training"]["train_split"]
    seed = config["training"]["seed"]

    set_seed(seed)

    X_train, X_test, y_train, y_test, normalizers = generate_dataset(
        function_name=function_name,
        dim=dim,
        domain=domain,
        n_samples=n_samples,
        train_split=train_split,
        seed=seed
    )

    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "normalizers": normalizers,
        "group": group,
        "function_name": function_name,
        "dim": dim,
        "domain": domain
    }

    with open(output_file, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
