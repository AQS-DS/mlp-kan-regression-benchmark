"""
Build summary.csv from the given metrics.json files.
Fails if a file or a required field is missing.
Columns: group, function_name, dim, model_name, mse_train, mse_test, r2_test, n_params, n_params_total.
Sort: group, function_name, model_name.
Plain CSV: UTF-8, comma-separated, with a header.
"""

import sys
import json
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

REQUIRED_KEYS = [
    "group", "function_name", "dim", "model_name",
    "mse_train", "mse_test", "r2_test",
    "n_params", "n_params_trainable", "n_params_total",
]
OUTPUT_COLUMNS = [
    "group", "function_name", "dim", "model_name",
    "mse_train", "mse_test", "r2_test",
    "n_params", "n_params_total",
    "flops_forward",
]


def load_metrics(path: str) -> dict:
    """Load a metrics.json file and require every mandatory field."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"{path}: missing required fields: {missing}")
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="*", help="Paths to metrics.json files (may be empty in partial mode)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_files = args.input
    output_file = args.output

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in input_files:
        data = load_metrics(path)
        # flops_forward is optional (older metrics files may omit it)
        row = {k: data.get(k) for k in OUTPUT_COLUMNS}
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        # Sort by group, then function_name, then model_name (mlp before kan)
        model_order = {"mlp": 0, "kan": 1}
        df["_model_ord"] = df["model_name"].map(lambda m: model_order.get(m, 2))
        df = df.sort_values(by=["group", "function_name", "_model_ord"]).drop(columns=["_model_ord"])

    df.to_csv(output_file, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
