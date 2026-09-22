import yaml
from pathlib import Path
import subprocess

# Load configuration
configfile: "config.yaml"
config = yaml.safe_load(open("config.yaml"))

# Collect function and model lists
models = config["models"]

# Build every (group, function) pair. dim lives on the target, not in the path.
functions = []
for func, spec in config["targets"].items():
    functions.append((spec["group"], func))

# Summary mode: full = every metrics file in the config; partial = only those that exist
summary_mode = config.get("summary", {}).get("mode", "full")


def summary_input(wildcards):
    import glob
    if summary_mode == "full":
        return [f"results/{g}/{f}/{m}/metrics.json" for (g, f) in functions for m in models]
    else:
        return sorted(glob.glob("results/*/*/*/metrics.json"))


def report_input(wildcards):
    import glob
    base = ["results/summary.csv"]
    if summary_mode == "full":
        # In full mode the report depends on every plot so Snakemake builds them
        return base + [f"results/{g}/{f}/{m}/plots.png" for (g, f) in functions for m in models]
    else:
        return base + sorted(glob.glob("results/*/*/*/plots/*.png"))


# Top rule: the pipeline ends at the report
rule all:
    input:
        "results/report.html"

# Data generation (nested: data/{group}/{function_name}.pkl)
# Note: scripts/utils.py is intentionally not an input. If you change a function definition,
# delete only the affected data/{group}/{func}.pkl files and rerun, so the rest is not rebuilt.
rule generate_data:
    output:
        "data/{group}/{function_name}.pkl"
    run:
        if (wildcards.group, wildcards.function_name) not in functions:
            raise ValueError(
                f"Invalid combination: group={wildcards.group}, function_name={wildcards.function_name}. "
                f"Valid: {functions}"
            )
        subprocess.run([
            "python", "scripts/generate_data.py",
            "--group", wildcards.group,
            "--function_name", wildcards.function_name,
            "--output", output[0]
        ], check=True)

# Training (results/{group}/{function_name}/{model_name}/)
# training_info.json is an output so plot (and any rule that needs it) triggers train
rule train:
    input:
        data = "data/{group}/{function_name}.pkl"
    output:
        model = "results/{group}/{function_name}/{model_name}/model.pt",
        training_info = "results/{group}/{function_name}/{model_name}/training_info.json"
    run:
        if (wildcards.group, wildcards.function_name) not in functions:
            raise ValueError(
                f"Invalid combination: group={wildcards.group}, function_name={wildcards.function_name}. "
                f"Valid: {functions}"
            )
        Path(output.model).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python", "scripts/train.py",
            "--group", wildcards.group,
            "--function_name", wildcards.function_name,
            "--model_name", wildcards.model_name,
            "--data", input.data,
            "--output", output.model
        ], check=True)

# Evaluation (metrics plus precomputed plot data; the model is loaded once)
rule evaluate:
    input:
        model = "results/{group}/{function_name}/{model_name}/model.pt",
        data = "data/{group}/{function_name}.pkl"
    output:
        metrics = "results/{group}/{function_name}/{model_name}/metrics.json",
        plot_data = "results/{group}/{function_name}/{model_name}/plot_data.json"
    run:
        if (wildcards.group, wildcards.function_name) not in functions:
            raise ValueError(
                f"Invalid combination: group={wildcards.group}, function_name={wildcards.function_name}. "
                f"Valid: {functions}"
            )
        Path(output.metrics).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python", "scripts/evaluate.py",
            "--group", wildcards.group,
            "--function_name", wildcards.function_name,
            "--model_name", wildcards.model_name,
            "--model", input.model,
            "--data", input.data,
            "--output", output.metrics
        ], check=True)

# Plotting: combined plots.png and individual plots/*.png (reads cached plot_data only).
# The report flag does not change this rule.
rule plot:
    input:
        plot_data = "results/{group}/{function_name}/{model_name}/plot_data.json",
        metrics = "results/{group}/{function_name}/{model_name}/metrics.json",
        training_info = "results/{group}/{function_name}/{model_name}/training_info.json"
    output:
        combined = "results/{group}/{function_name}/{model_name}/plots.png",
        plot_dir = directory("results/{group}/{function_name}/{model_name}/plots")
    run:
        if (wildcards.group, wildcards.function_name) not in functions:
            raise ValueError(
                f"Invalid combination: group={wildcards.group}, function_name={wildcards.function_name}. "
                f"Valid: {functions}"
            )
        out_dir = Path(output.combined).parent
        subprocess.run([
            "python", "scripts/plot.py",
            "--group", wildcards.group,
            "--function_name", wildcards.function_name,
            "--model_name", wildcards.model_name,
            "--plot-data", input.plot_data,
            "--metrics", input.metrics,
            "--training-info", input.training_info,
            "--output-dir", str(out_dir)
        ], check=True)

# Aggregate results (inputs follow summary.mode: full or partial)
rule summary:
    input: summary_input
    output:
        "results/summary.csv"
    run:
        subprocess.run([
            "python", "scripts/summarize.py",
            "--input"] + list(input) + [
            "--output", output[0]
        ], check=True)

# HTML report. comparison_scatter.png is always written.
# comparison_by_group.png is written only when report.group_by is group,
# so it is not a required output of this rule.
rule report:
    input: report_input
    output:
        html = "results/report.html",
        comparison_scatter = "results/comparison_scatter.png",
    run:
        subprocess.run([
            "python", "scripts/report.py",
            "--summary", input[0],
            "--output", output.html
        ], check=True)
