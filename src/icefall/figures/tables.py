"""Table generation utilities: results -> Markdown + LaTeX.

Reads aggregated experiment results and produces publication-ready tables
in both Markdown (for quick inspection) and LaTeX (for paper.tex).

Supported tables:
  - pareto:         Best F1 operating point per algorithm (original format)
  - best_f1:        Compact best F1 table with F1 and delegation in each cell
  - fixed_budget:   F1 at fixed delegation budgets (d<=20%, d<=30%)
  - min_delegation: Minimum delegation to reach target F1 (>=0.90, >=0.95)

Usage:
    uv run -m icefall.figures.tables --input results/exp1_pareto/batch-4096_dop-1/exp1_pareto.csv --experiment all
    uv run -m icefall.figures.tables --input results/exp1_pareto/batch-4096_dop-1/exp1_pareto.csv --experiment best_f1
    uv run -m icefall.figures.tables --input results/exp1_pareto/batch-4096_dop-1/exp1_pareto.csv --experiment fixed_budget --budgets 0.20,0.30
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

ALGO_ORDER = ["proxy_only", "supg_rt", "supg_joint", "supg_it", "gamcal", "oracle_only"]
PAPER_TABLE_ALGOS = ["proxy_only", "supg_rt", "supg_joint", "supg_it", "gamcal"]
FIXED_BUDGET_ALGOS = ["supg_joint", "supg_it", "gamcal"]

ALGO_PAPER_NAMES = {
    "proxy_only": "Proxy-only",
    "supg_rt": "SUPG",
    "supg_joint": "SUPG-SP",
    "supg_it": "SUPG-IT",
    "gamcal": "GAMCAL",
    "oracle_only": "Oracle",
}

ALGO_LATEX_NAMES = {
    "proxy_only": "Proxy-only",
    "supg_rt": "\\sn",
    "supg_joint": "\\supgsp",
    "supg_it": "\\supgit",
    "gamcal": "\\gamcal",
    "oracle_only": "Oracle",
}

DATASET_LABELS = {
    "arxiv": "ArXiv",
    "boolq": "BoolQ",
    "imdb": "IMDB",
    "mmlu": "MMLU",
    "nyt": "NYT",
    "sst2": "SST-2",
}

DATASET_ORDER = ["arxiv", "boolq", "imdb", "mmlu", "nyt", "sst2"]


def _fmt(mean: float, std: float, decimals: int = 3) -> str:
    """Format mean +/- std."""
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _fmt_latex(mean: float, std: float, decimals: int = 3) -> str:
    """Format mean +/- std for LaTeX."""
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _agg_by_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (dataset, algorithm, sweep_value): mean/std across seeds."""
    metrics = ["f1", "precision", "recall", "delegation_rate"]
    agg_cols = {m: ["mean", "std"] for m in metrics}
    grouped = df.groupby(["dataset", "algorithm", "sweep_value"]).agg(agg_cols).reset_index()
    grouped.columns = ["_".join(c).rstrip("_") for c in grouped.columns]
    return grouped


def _order_datasets(datasets):
    """Sort datasets in canonical order."""
    rank = {d: i for i, d in enumerate(DATASET_ORDER)}
    return sorted(datasets, key=lambda d: rank.get(d, 999))


def generate_pareto_summary(
    results_path: str | Path,
    output_dir: str | Path,
    metrics: list[str] = None,
) -> pd.DataFrame:
    """Generate summary table: mean +/- std of metrics per algorithm x dataset.

    Groups by (dataset, algorithm), computes mean/std across seeds and sweep values,
    then picks the best operating point (highest F1) for each algorithm.

    Args:
        results_path: Path to exp1_pareto.csv
        output_dir: Directory to write output files
        metrics: Metrics to include (default: f1, precision, recall, delegation_rate)

    Returns:
        Aggregated DataFrame
    """
    if metrics is None:
        metrics = ["f1", "precision", "recall", "delegation_rate"]

    df = pd.read_csv(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_cols = {m: ["mean", "std"] for m in metrics}
    grouped = df.groupby(["dataset", "algorithm", "sweep_value"]).agg(agg_cols).reset_index()
    grouped.columns = ["_".join(c).rstrip("_") for c in grouped.columns]

    best_idx = grouped.groupby(["dataset", "algorithm"])["f1_mean"].idxmax()
    best = grouped.loc[best_idx].copy()

    algo_order = ["proxy_only", "supg_rt", "supg_joint", "supg_it", "gamcal", "oracle_only"]
    best["algo_rank"] = best["algorithm"].map({a: i for i, a in enumerate(algo_order)})
    best = best.sort_values(["dataset", "algo_rank"]).drop(columns=["algo_rank"])

    md_lines = _pareto_markdown(best, metrics)
    md_path = output_dir / "exp1_pareto_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Markdown table saved to {md_path}")

    tex_lines = _pareto_latex(best, metrics)
    tex_path = output_dir / "exp1_pareto_summary.tex"
    tex_path.write_text("\n".join(tex_lines))
    print(f"LaTeX table saved to {tex_path}")

    print("\n" + "\n".join(md_lines))

    return best


def _pareto_markdown(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    """Generate Markdown table from Pareto summary."""
    lines = ["## Experiment 1: Pareto Summary (Best F1 per Algorithm)", ""]

    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        lines.append(f"### {dataset}")
        lines.append("")

        header = (
            "| Algorithm | "
            + " | ".join(m.replace("_", " ").title() for m in metrics)
            + " | Sweep Value |"
        )
        sep = "|" + "|".join(["---"] * (len(metrics) + 2)) + "|"
        lines.extend([header, sep])

        for _, row in subset.iterrows():
            cells = [f"{row['algorithm']:<12}"]
            for m in metrics:
                cells.append(_fmt(row[f"{m}_mean"], row[f"{m}_std"]))
            sv = row.get("sweep_value", "")
            cells.append(f"{sv}" if pd.notna(sv) else "-")
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    return lines


def _pareto_latex(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    """Generate LaTeX table from Pareto summary."""
    n_cols = len(metrics) + 2
    col_spec = "l" + "c" * (len(metrics) + 1)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cost-quality tradeoff: best F1 operating point per algorithm.}",
        "\\label{tab:pareto-summary}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    header_cells = ["Algorithm"] + [m.replace("_", " ").title() for m in metrics] + ["Param"]
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    prev_dataset = None
    for _, row in df.iterrows():
        dataset = row["dataset"]
        if dataset != prev_dataset:
            if prev_dataset is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{dataset}}}}} \\\\")
            lines.append("\\midrule")
            prev_dataset = dataset

        cells = [row["algorithm"].replace("_", "\\_")]
        for m in metrics:
            cells.append(_fmt_latex(row[f"{m}_mean"], row[f"{m}_std"]))
        sv = row.get("sweep_value", "")
        cells.append(f"{sv:.3f}" if pd.notna(sv) and sv is not None else "--")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return lines


def generate_best_f1_table(
    results_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Compact table: best F1 per algorithm with delegation rate.

    Each cell shows F1 and delegation rate. Boldfaces the best cascade
    algorithm (excluding Oracle-only) per dataset.

    Methodology: for each (dataset, algorithm), aggregates across seeds at
    each sweep value, then picks the sweep value with highest mean F1.
    Baselines (proxy_only, oracle_only) are handled separately since they
    have no sweep parameter.
    """
    df = pd.read_csv(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _agg_by_sweep(df)
    best_idx = grouped.groupby(["dataset", "algorithm"])["f1_mean"].idxmax()
    best = grouped.loc[best_idx].copy()

    for baseline in ["proxy_only", "oracle_only"]:
        bsub = df[df["algorithm"] == baseline]
        if bsub.empty:
            continue
        for ds in bsub["dataset"].unique():
            dsub = bsub[bsub["dataset"] == ds]
            row = {
                "dataset": ds,
                "algorithm": baseline,
                "sweep_value": np.nan,
                "f1_mean": dsub["f1"].mean(),
                "f1_std": dsub["f1"].std(),
                "delegation_rate_mean": dsub["delegation_rate"].mean(),
                "delegation_rate_std": dsub["delegation_rate"].std(),
                "precision_mean": dsub["precision"].mean(),
                "precision_std": dsub["precision"].std(),
                "recall_mean": dsub["recall"].mean(),
                "recall_std": dsub["recall"].std(),
            }
            best = pd.concat([best, pd.DataFrame([row])], ignore_index=True)

    datasets = _order_datasets(best["dataset"].unique())
    algos = [a for a in PAPER_TABLE_ALGOS if a in best["algorithm"].unique()]

    md = _best_f1_markdown(best, datasets, algos)
    (output_dir / "best_f1_per_algorithm.md").write_text("\n".join(md))

    tex = _best_f1_latex(best, datasets, algos)
    (output_dir / "best_f1_per_algorithm.tex").write_text("\n".join(tex))

    print(f"best_f1_per_algorithm.{{md,tex}} saved to {output_dir}")
    return best


def _best_f1_markdown(df: pd.DataFrame, datasets: list, algos: list) -> list[str]:
    lines = [
        "## Best F1 per Algorithm",
        "",
        "Each cell: F1 (delegation %). Bold = best cascade algorithm.",
        "",
    ]
    header = "| Dataset | " + " | ".join(ALGO_PAPER_NAMES.get(a, a) for a in algos) + " |"
    sep = "|---" * (len(algos) + 1) + "|"
    lines.extend([header, sep])

    cascade_algos = [a for a in algos if a not in ("proxy_only", "oracle_only")]

    for ds in datasets:
        cells = [DATASET_LABELS.get(ds, ds)]
        best_cascade_f1 = -1
        for a in cascade_algos:
            row = df[(df["dataset"] == ds) & (df["algorithm"] == a)]
            if not row.empty:
                best_cascade_f1 = max(best_cascade_f1, row["f1_mean"].values[0])

        for a in algos:
            row = df[(df["dataset"] == ds) & (df["algorithm"] == a)]
            if row.empty:
                cells.append("---")
                continue
            f1 = row["f1_mean"].values[0]
            d = row["delegation_rate_mean"].values[0]
            cell = f".{f1:.3f}"[1:] + f" ({d:.0%})"
            if a in cascade_algos and abs(f1 - best_cascade_f1) < 1e-4:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def _best_f1_latex(df: pd.DataFrame, datasets: list, algos: list) -> list[str]:
    n_algo = len(algos)
    col_spec = "l" + "c" * n_algo

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Best $F_1$ operating point per algorithm. "
        "Each cell shows $F_1$ score and delegation rate $d$. "
        "Bold indicates the best cascade algorithm per dataset.}",
        "\\label{tab:best-f1}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    header = "Dataset & " + " & ".join(ALGO_LATEX_NAMES.get(a, a) for a in algos) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    cascade_algos = [a for a in algos if a not in ("proxy_only", "oracle_only")]

    for ds in datasets:
        best_cascade_f1 = -1
        for a in cascade_algos:
            row = df[(df["dataset"] == ds) & (df["algorithm"] == a)]
            if not row.empty:
                best_cascade_f1 = max(best_cascade_f1, row["f1_mean"].values[0])

        cells = [DATASET_LABELS.get(ds, ds)]
        for a in algos:
            row = df[(df["dataset"] == ds) & (df["algorithm"] == a)]
            if row.empty:
                cells.append("---")
                continue
            f1 = row["f1_mean"].values[0]
            d = row["delegation_rate_mean"].values[0]
            d_pct = f"{d * 100:.0f}\\%"
            if a in cascade_algos and abs(f1 - best_cascade_f1) < 1e-4:
                cell = f"$\\mathbf{{{f1:.3f}}}$~({d_pct})"
            else:
                cell = f"${f1:.3f}$~({d_pct})"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return lines


def generate_fixed_budget_table(
    results_path: str | Path,
    output_dir: str | Path,
    budgets: list[float] = None,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Table: best F1 achievable within a delegation budget.

    Methodology: for each (dataset, algorithm, budget), finds all operating
    points where mean delegation rate <= budget, then picks the one with highest
    mean F1. Uses actual sweep data points, no interpolation.

    Args:
        budgets: delegation budget levels (default: [0.20, 0.30])
        tolerance: legacy margin (default 0; strict d <= budget)
    """
    if budgets is None:
        budgets = [0.20, 0.30]

    df = pd.read_csv(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _agg_by_sweep(df)
    datasets = _order_datasets(grouped["dataset"].unique())
    algos = [a for a in FIXED_BUDGET_ALGOS if a in grouped["algorithm"].unique()]

    results = {}
    for ds in datasets:
        for budget in budgets:
            for algo in algos:
                sub = grouped[
                    (grouped["dataset"] == ds)
                    & (grouped["algorithm"] == algo)
                    & (grouped["delegation_rate_mean"] <= budget + tolerance)
                ]
                if len(sub) > 0:
                    best = sub.loc[sub["f1_mean"].idxmax()]
                    results[(ds, budget, algo)] = best["f1_mean"]
                else:
                    results[(ds, budget, algo)] = None

    md = _fixed_budget_markdown(results, datasets, algos, budgets)
    (output_dir / "f1_at_fixed_budget.md").write_text("\n".join(md))

    tex = _fixed_budget_latex(results, datasets, algos, budgets)
    (output_dir / "f1_at_fixed_budget.tex").write_text("\n".join(tex))

    print(f"f1_at_fixed_budget.{{md,tex}} saved to {output_dir}")
    return grouped


def _find_best_in_budget_column(results: dict, ds: str, budget: float, algos: list) -> float:
    """Best F1 among algorithms for one dataset and budget column."""
    vals = [results.get((ds, budget, a)) for a in algos]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else -1


def _fixed_budget_markdown(results, datasets, algos, budgets) -> list[str]:
    lines = [
        "## F1 at Fixed Delegation Budgets",
        "",
        "Best F1 achievable within delegation budget (strict d <= budget).",
        "Bold = best algorithm per budget column.",
        "",
    ]
    header_parts = ["Dataset"]
    for budget in budgets:
        for a in algos:
            header_parts.append(f"{ALGO_PAPER_NAMES.get(a, a)} (d<={budget:.0%})")
    header = "| " + " | ".join(header_parts) + " |"
    sep = "|---" * len(header_parts) + "|"
    lines.extend([header, sep])

    for ds in datasets:
        cells = [DATASET_LABELS.get(ds, ds)]
        for budget in budgets:
            best_f1 = _find_best_in_budget_column(results, ds, budget, algos)
            for a in algos:
                val = results.get((ds, budget, a))
                if val is None:
                    cells.append("---")
                else:
                    cell = f"{val:.3f}"
                    if abs(val - best_f1) < 1e-4:
                        cell = f"**{cell}**"
                    cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def _fixed_budget_latex(results, datasets, algos, budgets) -> list[str]:
    n_data_cols = len(budgets) * len(algos)
    col_spec = "l" + "c" * n_data_cols

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{$F_1$ at fixed delegation budgets. Each cell shows the best "
        "$F_1$ achievable with delegation rate $d \\leq$ budget. "
        "Bold marks the best algorithm per column and dataset.}",
        "\\label{tab:fixed-budget}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    top_header = " & "
    for i, budget in enumerate(budgets):
        top_header += f"\\multicolumn{{{len(algos)}}}{{c}}{{$d \\leq {budget * 100:.0f}\\%$}}"
        if i < len(budgets) - 1:
            top_header += " & "
    top_header += " \\\\"
    lines.append(top_header)

    cmidrule_parts = []
    for i, _budget in enumerate(budgets):
        start = 2 + i * len(algos)
        end = start + len(algos) - 1
        cmidrule_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrule_parts))

    sub_header = (
        "Dataset & "
        + " & ".join(ALGO_LATEX_NAMES.get(a, a) for _ in budgets for a in algos)
        + " \\\\"
    )
    lines.append(sub_header)
    lines.append("\\midrule")

    for ds in datasets:
        cells = [DATASET_LABELS.get(ds, ds)]
        for budget in budgets:
            best_f1 = _find_best_in_budget_column(results, ds, budget, algos)
            for a in algos:
                val = results.get((ds, budget, a))
                if val is None:
                    cells.append("---")
                elif abs(val - best_f1) < 1e-4:
                    cells.append(f"$\\mathbf{{{val:.3f}}}$")
                else:
                    cells.append(f"${val:.3f}$")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return lines


def generate_min_delegation_table(
    results_path: str | Path,
    output_dir: str | Path,
    targets: list[float] = None,
) -> pd.DataFrame:
    """Table: minimum delegation rate to reach a target F1.

    Methodology: for each (dataset, algorithm), finds all operating points
    where mean F1 >= target, then picks the one with lowest mean delegation
    rate. Uses actual sweep data points, no interpolation.

    Args:
        targets: F1 target levels (default: [0.90, 0.95])
    """
    if targets is None:
        targets = [0.90, 0.95]

    df = pd.read_csv(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _agg_by_sweep(df)
    datasets = _order_datasets(grouped["dataset"].unique())
    algos = ["supg_joint", "supg_it", "gamcal"]
    algos = [a for a in algos if a in grouped["algorithm"].unique()]

    results = {}
    for ds in datasets:
        for target in targets:
            for algo in algos:
                sub = grouped[
                    (grouped["dataset"] == ds)
                    & (grouped["algorithm"] == algo)
                    & (grouped["f1_mean"] >= target)
                ]
                if len(sub) > 0:
                    best = sub.loc[sub["delegation_rate_mean"].idxmin()]
                    results[(ds, target, algo)] = best["delegation_rate_mean"]
                else:
                    results[(ds, target, algo)] = None

    md = _min_deleg_markdown(results, datasets, algos, targets)
    (output_dir / "min_delegation_for_target.md").write_text("\n".join(md))

    tex = _min_deleg_latex(results, datasets, algos, targets)
    (output_dir / "min_delegation_for_target.tex").write_text("\n".join(tex))

    print(f"min_delegation_for_target.{{md,tex}} saved to {output_dir}")
    return grouped


def _find_min_in_row(results: dict, ds: str, target: float, algos: list) -> float:
    """Find the lowest delegation among algorithms for this row."""
    vals = [results.get((ds, target, a)) for a in algos]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else float("inf")


def _min_deleg_markdown(results, datasets, algos, targets) -> list[str]:
    lines = [
        "## Minimum Delegation for Target F1",
        "",
        "Minimum delegation rate to achieve target F1 (no interpolation).",
        "Bold = most cost-efficient algorithm per row. N/R = not reachable.",
        "",
    ]

    header_parts = ["Dataset"]
    for t in targets:
        for a in algos:
            header_parts.append(f"{ALGO_PAPER_NAMES.get(a, a)} (F1>={t})")
    header = "| " + " | ".join(header_parts) + " |"
    sep = "|---" * (len(header_parts)) + "|"
    lines.extend([header, sep])

    for ds in datasets:
        cells = [DATASET_LABELS.get(ds, ds)]
        for t in targets:
            best_d = _find_min_in_row(results, ds, t, algos)
            for a in algos:
                val = results.get((ds, t, a))
                if val is None:
                    cells.append("N/R")
                else:
                    cell = f"{val:.1%}"
                    if abs(val - best_d) < 1e-4:
                        cell = f"**{cell}**"
                    cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def _min_deleg_latex(results, datasets, algos, targets) -> list[str]:
    n_data_cols = len(targets) * len(algos)
    col_spec = "l" + "c" * n_data_cols

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Minimum delegation rate to achieve target $F_1$. "
        "Bold indicates the most cost-efficient algorithm. "
        "N/R = target not reachable by this algorithm.}",
        "\\label{tab:min-delegation}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    top_header = " & "
    for i, t in enumerate(targets):
        top_header += f"\\multicolumn{{{len(algos)}}}{{c}}{{$F_1 \\geq {t}$}}"
        if i < len(targets) - 1:
            top_header += " & "
    top_header += " \\\\"
    lines.append(top_header)

    cmidrule_parts = []
    for i, t in enumerate(targets):
        start = 2 + i * len(algos)
        end = start + len(algos) - 1
        cmidrule_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrule_parts))

    sub_header = (
        "Dataset & "
        + " & ".join(ALGO_LATEX_NAMES.get(a, a) for _ in targets for a in algos)
        + " \\\\"
    )
    lines.append(sub_header)
    lines.append("\\midrule")

    for ds in datasets:
        cells = [DATASET_LABELS.get(ds, ds)]
        for t in targets:
            best_d = _find_min_in_row(results, ds, t, algos)
            for a in algos:
                val = results.get((ds, t, a))
                if val is None:
                    cells.append("N/R")
                else:
                    v_pct = f"{val * 100:.1f}\\%"
                    if abs(val - best_d) < 1e-4:
                        cells.append(f"$\\mathbf{{{v_pct}}}$")
                    else:
                        cells.append(f"${v_pct}$")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return lines


def generate_all_publication_tables(
    results_path: str | Path,
    output_dir: str | Path,
    budgets: list[float] = None,
    targets: list[float] = None,
):
    """Generate all three publication tables."""
    generate_best_f1_table(results_path, output_dir)
    generate_fixed_budget_table(results_path, output_dir, budgets=budgets)
    generate_min_delegation_table(results_path, output_dir, targets=targets)


if __name__ == "__main__":
    import argparse

    EXPERIMENTS = ["pareto", "best_f1", "fixed_budget", "min_delegation", "all"]

    parser = argparse.ArgumentParser(description="Generate experiment tables")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output-dir", default=None, help="Output directory for generated tables")
    parser.add_argument(
        "--experiment", choices=EXPERIMENTS, default="pareto", help="Which table(s) to generate"
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="0.20,0.30",
        help="Delegation budgets for fixed_budget (comma-separated)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="0.90,0.95",
        help="F1 targets for min_delegation (comma-separated)",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(args.input).parent.parent / "tables"
    )
    budgets = [float(b) for b in args.budgets.split(",")]
    targets = [float(t) for t in args.targets.split(",")]

    if args.experiment == "pareto":
        generate_pareto_summary(args.input, output_dir)
    elif args.experiment == "best_f1":
        generate_best_f1_table(args.input, output_dir)
    elif args.experiment == "fixed_budget":
        generate_fixed_budget_table(args.input, output_dir, budgets=budgets)
    elif args.experiment == "min_delegation":
        generate_min_delegation_table(args.input, output_dir, targets=targets)
    elif args.experiment == "all":
        generate_all_publication_tables(args.input, output_dir, budgets=budgets, targets=targets)
