"""
Write results/report.html from summary.csv.

report.group_by in config.yaml selects the layout:
  target — one section per function, a flat table, and the MLP vs KAN scatter
  group  — the same, plus means, a bar chart, and tables by group

Per-run figures are produced by plot.py either way. This script only
assembles the HTML and the comparison charts.
"""

import html
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config


INK = "#1c1915"
ACCENT = "#1f4e5f"
KAN = "#8c4a32"
PAPER = "#f3efe6"

PLOT_1D = [
    ("real_vs_pred.png", "Target and prediction"),
    ("loss.png", "Training curve"),
    ("residuals.png", "Residuals"),
    ("residuals_hist.png", "Residual histogram"),
]
PLOT_2D = [
    ("real_vs_pred.png", "Target"),
    ("pred.png", "Prediction"),
    ("loss.png", "Training curve"),
    ("residuals_hist.png", "Residual histogram"),
]

DISPLAY = {
    "function_name": "Function",
    "group": "Group",
    "dim": "Dim",
    "model_name": "Model",
    "mse_train": "MSE train",
    "mse_test": "MSE test",
    "r2_test": "R² test",
    "n_params": "Trainable params",
    "n_params_total": "Total params",
    "flops_forward": "FLOPs",
}

STYLE = """
:root {
  --ink: #1c1915;
  --muted: #5c564c;
  --paper: #f3efe6;
  --card: #faf8f4;
  --line: #d9d2c5;
  --accent: #1f4e5f;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font-family: Palatino, "Palatino Linotype", "Iowan Old Style", Georgia, serif;
  line-height: 1.5;
}
.wrap { max-width: 980px; margin: 0 auto; padding: 2.5rem 1.4rem 4rem; }
header { border-top: 4px solid var(--accent); padding-top: 1.1rem; }
.kicker {
  margin: 0;
  color: var(--accent);
  font-size: 0.78rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
h1 { font-weight: 500; font-size: 2rem; letter-spacing: -0.02em; margin: 0.2rem 0 0.6rem; }
h2 {
  font-weight: 500;
  font-size: 1.35rem;
  margin: 2.4rem 0 0.6rem;
  padding-bottom: 0.25rem;
  border-bottom: 1px solid var(--line);
}
h3 { font-weight: 500; font-size: 1.05rem; margin: 1.4rem 0 0.4rem; }
.lede, .meta { color: var(--muted); }
.lede { max-width: 40rem; margin-top: 0; }
.formula {
  font-family: "Cambria Math", Georgia, serif;
  background: var(--card);
  border-left: 3px solid var(--accent);
  padding: 0.35rem 0.75rem;
  margin: 0.6rem 0;
}
nav.toc {
  background: var(--card);
  border: 1px solid var(--line);
  padding: 0.85rem 1.15rem 1rem;
  margin: 1.6rem 0 0.4rem;
}
nav.toc h2 { margin: 0; border: 0; padding: 0; font-size: 1rem; }
nav.toc ol { margin: 0.45rem 0 0; padding-left: 1.2rem; }
nav.toc li { margin: 0.15rem 0; }
nav.toc a, .back a { color: var(--accent); text-underline-offset: 0.15em; }
.back { margin: 0.8rem 0 0; font-size: 0.85rem; }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--card);
  font-size: 0.92rem;
  font-variant-numeric: tabular-nums;
  margin: 0.4rem 0 0.8rem;
}
th, td { padding: 0.4rem 0.65rem; text-align: left; border-bottom: 1px solid var(--line); }
thead th { background: var(--accent); color: #f7f4ee; font-weight: 500; }
.fig-row { display: flex; flex-wrap: wrap; gap: 1rem; margin: 0.35rem 0 0.8rem; }
.fig-cell { display: flex; flex-direction: column; align-items: flex-start; }
.model-label {
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.25rem;
}
.fig-cell img, .comparison-plot {
  background: white;
  border: 1px solid var(--line);
}
.fig-cell img { max-width: 440px; height: auto; }
.comparison-plot { max-width: 520px; height: auto; }
.caption { margin: 0.8rem 0 0.2rem; color: var(--muted); font-size: 0.92rem; }
"""


def fmt_mse(x):
    return f"{float(x):.2e}"


def fmt_r2(x):
    return f"{float(x):.6f}"


def fmt_int(x):
    return str(int(round(float(x))))


def slug(text):
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text))
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "section"


def report_mode(config):
    mode = (config.get("report") or {}).get("group_by", "target")
    if mode not in ("target", "group"):
        raise ValueError("report.group_by must be 'target' or 'group'")
    return mode


def _format_frame(df):
    out = df.copy()
    for col in out.columns:
        if "mse" in col:
            out[col] = out[col].apply(lambda x: fmt_mse(x) if pd.notna(x) else "—")
        elif "r2" in col:
            out[col] = out[col].apply(lambda x: fmt_r2(x) if pd.notna(x) else "—")
        elif "n_params" in col or "flops" in col or col == "dim":
            out[col] = out[col].apply(lambda x: fmt_int(x) if pd.notna(x) else "—")
    return out


def _html_table(df, css_class):
    renamed = df.rename(columns={c: DISPLAY.get(c, c) for c in df.columns})
    return renamed.to_html(index=False, border=0, classes=css_class)


def build_summary_global(df):
    if df.empty:
        return "<p>No data.</p>"
    cols = [c for c in ["function_name", "group", "dim", "model_name", "mse_train", "mse_test", "r2_test", "n_params", "n_params_total", "flops_forward"] if c in df.columns]
    return _html_table(_format_frame(df[cols]), "runs")


def _metric_columns(df):
    cols = ["mse_train", "mse_test", "r2_test", "n_params"]
    if "flops_forward" in df.columns:
        cols.append("flops_forward")
    return cols


def build_averages_by_group(df):
    if df.empty:
        return "<p>No data.</p>"
    values = _metric_columns(df)
    agg = df.groupby(["group", "model_name"])[values].mean().reset_index()
    wide = agg.pivot(index="group", columns="model_name", values=values)
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    cols = ["group"]
    for metric in values:
        for model in ("mlp", "kan"):
            name = f"{metric}_{model}"
            if name in wide.columns:
                cols.append(name)
    wide = _format_frame(wide[cols])
    labels = {"group": "Group"}
    for metric in values:
        pretty = DISPLAY.get(metric, metric)
        for model in ("mlp", "kan"):
            labels[f"{metric}_{model}"] = f"{pretty} · {model.upper()}"
    wide = wide.rename(columns=labels)
    return wide.to_html(index=False, border=0, classes="means")


def build_group_tables(df):
    tables = []
    cols = [c for c in ["function_name", "dim", "model_name", "mse_train", "mse_test", "r2_test", "n_params", "n_params_total", "flops_forward"] if c in df.columns]
    for group in sorted(df["group"].unique()):
        sub = _format_frame(df.loc[df["group"] == group, cols])
        tables.append((group, _html_table(sub, "group")))
    return tables


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_comparison_by_group(df, out_path):
    fig, ax = plt.subplots(figsize=(max(6, df["group"].nunique() * 1.3), 4), facecolor="white")
    if df.empty or "model_name" not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        agg = df.groupby(["group", "model_name"])["mse_test"].mean().reset_index()
        groups = sorted(agg["group"].unique())
        mlp = [agg[(agg["group"] == g) & (agg["model_name"] == "mlp")]["mse_test"].iloc[0] if len(agg[(agg["group"] == g) & (agg["model_name"] == "mlp")]) else np.nan for g in groups]
        kan = [agg[(agg["group"] == g) & (agg["model_name"] == "kan")]["mse_test"].iloc[0] if len(agg[(agg["group"] == g) & (agg["model_name"] == "kan")]) else np.nan for g in groups]
        x = np.arange(len(groups))
        w = 0.35
        ax.bar(x - w / 2, mlp, w, label="MLP", color=ACCENT)
        ax.bar(x + w / 2, kan, w, label="KAN", color=KAN)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_ylabel("MSE test (mean)")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        ax.set_title("Mean test MSE by group")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_comparison_scatter(df, out_path, titles):
    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="white")
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _style_axes(ax)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return
    pivot = df.pivot_table(index=["group", "function_name"], columns="model_name", values="mse_test").reset_index()
    if "mlp" not in pivot.columns or "kan" not in pivot.columns:
        ax.text(0.5, 0.5, "Both MLP and KAN are required", ha="center", va="center", transform=ax.transAxes)
    else:
        pivot = pivot.dropna(subset=["mlp", "kan"])
        if pivot.empty:
            ax.text(0.5, 0.5, "No MLP–KAN pairs", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.scatter(pivot["mlp"], pivot["kan"], s=48, color=ACCENT, zorder=3)
            for _, row in pivot.iterrows():
                label = titles.get(row["function_name"], row["function_name"])
                ax.annotate(label, (row["mlp"], row["kan"]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
            lims = [min(pivot["mlp"].min(), pivot["kan"].min()), max(pivot["mlp"].max(), pivot["kan"].max())]
            ax.plot(lims, lims, linestyle="--", color="#8a8175", label="Equal test MSE")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("MSE test · MLP")
            ax.set_ylabel("MSE test · KAN")
            ax.legend(frameon=False, fontsize=8)
            ax.set_title("Test MSE, one point per target")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _cell(model_name, rel_path, alt):
    return (
        f'<div class="fig-cell"><span class="model-label">{html.escape(model_name.upper())}</span>'
        f'<img src="{html.escape(rel_path)}" alt="{html.escape(alt)}" /></div>'
    )


def _figure_block(results_dir, group, function_name, filename, caption):
    cells = []
    for model_name in ("mlp", "kan"):
        path = results_dir / group / function_name / model_name / "plots" / filename
        if path.exists():
            rel = f"{group}/{function_name}/{model_name}/plots/{filename}"
            cells.append(_cell(model_name, rel, f"{function_name} {model_name} {caption}"))
    if not cells:
        return ""
    return f'<p class="caption">{html.escape(caption)}</p><div class="fig-row">{"".join(cells)}</div>'


def _pair_value(sub, model_name, column, formatter):
    rows = sub[sub["model_name"] == model_name]
    if rows.empty or column not in rows.columns or pd.isna(rows.iloc[0][column]):
        return "—"
    return formatter(rows.iloc[0][column])


def build_target_metrics(sub):
    rows = [
        ("MSE train", "mse_train", fmt_mse),
        ("MSE test", "mse_test", fmt_mse),
        ("R² test", "r2_test", fmt_r2),
        ("Trainable params", "n_params", fmt_int),
        ("Total params", "n_params_total", fmt_int),
    ]
    if "flops_forward" in sub.columns:
        rows.append(("FLOPs", "flops_forward", fmt_int))
    body = []
    for label, column, formatter in rows:
        mlp = _pair_value(sub, "mlp", column, formatter)
        kan = _pair_value(sub, "kan", column, formatter)
        body.append(f"<tr><td>{html.escape(label)}</td><td>{mlp}</td><td>{kan}</td></tr>")
    return (
        '<table class="target"><thead><tr><th>Metric</th><th>MLP</th><th>KAN</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def ordered_functions(df, config):
    present = set(df["function_name"].astype(str))
    configured = list((config.get("targets") or {}).keys())
    ordered = [name for name in configured if name in present]
    ordered += sorted(name for name in present if name not in ordered)
    return ordered


def target_fields(config, function_name):
    spec = (config.get("targets") or {}).get(function_name, {})
    return {
        "title": spec.get("title") or function_name,
        "formula": spec.get("formula") or "",
        "description": spec.get("description") or "",
        "domain": spec.get("domain"),
    }


def build_target_section(results_dir, df, config, function_name):
    sub = df[df["function_name"] == function_name]
    group = str(sub["group"].iloc[0])
    dim = int(sub["dim"].iloc[0])
    fields = target_fields(config, function_name)
    domain = fields["domain"]
    if domain is None:
        domain_text = ""
    else:
        domain_text = f"[{domain[0]:g}, {domain[1]:g}]"
    parts = [build_target_metrics(sub)]
    if fields["formula"]:
        parts.insert(0, f'<p class="formula">{html.escape(fields["formula"])}</p>')
    meta_bits = [f"{dim}D input"]
    if domain_text:
        meta_bits.append(f"domain {domain_text}")
    meta_bits.append(f"group {group}")
    parts.insert(0, f'<p class="meta">{html.escape(" · ".join(meta_bits))}</p>')
    if fields["description"]:
        parts.insert(0, f'<p>{html.escape(fields["description"])}</p>')
    plots = PLOT_2D if dim == 2 else PLOT_1D
    figures = [_figure_block(results_dir, group, function_name, filename, caption) for filename, caption in plots]
    figures = [block for block in figures if block]
    if figures:
        parts.extend(figures)
    else:
        parts.append("<p class=\"meta\">No figures for this target yet.</p>")
    return fields["title"], "\n".join(parts)


def build_contents(entries):
    items = "\n".join(f'<li><a href="#{anchor}">{html.escape(title)}</a></li>' for anchor, title in entries)
    return f'<nav class="toc" id="contents"><h2>Contents</h2><ol>{items}</ol></nav>'


def build_section(anchor, title, body):
    return (
        f'<section id="{anchor}">'
        f"<h2>{html.escape(title)}</h2>"
        f"{body}"
        f'<p class="back"><a href="#contents">Contents</a></p>'
        f"</section>"
    )


def page(body):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MLP vs KAN benchmark report</title>
<style>
{STYLE}
</style>
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>
"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to results/summary.csv")
    parser.add_argument("--output", required=True, help="Path to results/report.html")
    parser.add_argument("--results-dir", default=None, help="Results directory (default: the output file's directory)")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_path = Path(args.output)
    results_dir = Path(args.results_dir) if args.results_dir else output_path.parent
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config()
    mode = report_mode(config)

    df = pd.read_csv(summary_path, encoding="utf-8")
    required = ["group", "function_name", "dim", "model_name", "mse_train", "mse_test", "r2_test", "n_params", "n_params_total"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"summary.csv must contain column '{col}'")

    titles = {name: target_fields(config, name)["title"] for name in df["function_name"].astype(str).unique()}

    scatter_path = results_dir / "comparison_scatter.png"
    plot_comparison_scatter(df, scatter_path, titles)

    group_chart = results_dir / "comparison_by_group.png"
    legacy_chart = results_dir / "comparison_by_category.png"
    if legacy_chart.exists():
        legacy_chart.unlink()
    if mode == "group":
        plot_comparison_by_group(df, group_chart)
    elif group_chart.exists():
        group_chart.unlink()

    if mode == "group":
        lede = (
            "Each target has its own section. Averages and the bar chart group runs that share a label. "
            "The numbers describe this configuration: one seed, and MLP and KAN parameter counts that are not matched."
        )
    else:
        lede = (
            "Each target has its own section. "
            "The numbers describe this configuration: one seed, and MLP and KAN parameter counts that are not matched."
        )

    entries = [("runs", "All runs")]
    sections = [build_section("runs", "All runs", build_summary_global(df))]

    for function_name in ordered_functions(df, config):
        title, body = build_target_section(results_dir, df, config, function_name)
        anchor = f"target-{slug(function_name)}"
        entries.append((anchor, title))
        sections.append(build_section(anchor, title, body))

    entries.append(("comparison", "MLP versus KAN"))
    sections.append(build_section(
        "comparison",
        "MLP versus KAN",
        '<p class="meta">Each point is one target. The dashed line is equal test MSE.</p>'
        '<p><img src="comparison_scatter.png" alt="MLP versus KAN test MSE" class="comparison-plot" /></p>',
    ))

    if mode == "group":
        entries.append(("means", "Means by group"))
        sections.append(build_section(
            "means",
            "Means by group",
            build_averages_by_group(df)
            + '<p><img src="comparison_by_group.png" alt="Mean test MSE by group" class="comparison-plot" /></p>',
        ))
        group_bodies = []
        for group, table_html in build_group_tables(df):
            group_bodies.append(f"<h3>{html.escape(str(group))}</h3>\n{table_html}")
        entries.append(("by-group", "Results by group"))
        sections.append(build_section("by-group", "Results by group", "\n".join(group_bodies)))

    body = (
        "<header>"
        '<p class="kicker">Synthetic regression</p>'
        "<h1>MLP vs KAN</h1>"
        f'<p class="lede">{html.escape(lede)}</p>'
        "</header>"
        + build_contents(entries)
        + "\n".join(sections)
    )
    output_path.write_text(page(body), encoding="utf-8")


if __name__ == "__main__":
    main()
