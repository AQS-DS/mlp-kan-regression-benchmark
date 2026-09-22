# Environment setup

This is a **local** Python 3.10 / Snakemake / MLflow demo. Run commands from the repository root; all Python dependencies are specified in [`requirements.txt`](../requirements.txt).

## Create and activate a virtual environment

On Linux or macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell (with Python 3.10 available):

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The package versions in `requirements.txt` reflect the project's tested dependency constraints; a compatible Python environment is important. Do not upgrade individual packages without checking compatibility with PyTorch, `pykan` and Snakemake.

## Verify and run

```bash
python --version
snakemake --version
python -c "import torch, kan, mlflow; print('PyTorch, pykan, MLflow OK')"
snakemake --dry-run
snakemake --cores 1
```

A complete run should generate `results/report.html` and four model/target combinations. Snakemake may take time during the first training pass. To start again after changing the configured experiments, use `snakemake --cores 1 --forceall`.

## Inspect experiments

In a second terminal, activate the same environment, navigate to the repository root and run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the URL printed in the terminal (normally `http://127.0.0.1:5000`). The Python scripts use the same local SQLite tracking store, `mlflow.db`, and write run artifacts to local storage; neither is checked into Git.

## Optional workflow image

To generate the DAG screenshot for the README, install the **Graphviz system application** so the `dot` command is available, then run:

```bash
snakemake --dag | dot -Tpng -o docs/images/workflow.png
```

The workflow image is a documentation asset and can be versioned under `docs/images/`. Generated datasets and model results should remain excluded by `.gitignore`.
