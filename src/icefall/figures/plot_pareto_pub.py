"""Publication-grade Pareto frontier figures for Experiment 1.

Produces:
  - Hero figure: F1 vs delegation rate, 1x3 or 2x3 layout, GAMCAL vs SUPG-IT
  - Appendix full grid: NxM layout with all algorithms and metrics
  - Appendix precision/recall decomposition: 1x3

Usage:
    uv run -m icefall.figures.plot_pareto_pub
    uv run -m icefall.figures.plot_pareto_pub --datasets imdb,mmlu,sst2,arxiv,boolq --layout 2x3
    uv run -m icefall.figures.plot_pareto_pub --y-min 0.3
    uv run -m icefall.figures.plot_pareto_pub --layout 2x3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from icefall.datasets import CURATED_DATASET_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_CSV = _REPO_ROOT / "results" / "exp1_pareto" / "batch-4096_dop-1" / "exp1_pareto.csv"
OUTPUT_DIR = _REPO_ROOT / "results" / "publication_figures"

PUB_RCPARAMS = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
}

PLOTLY_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

ALGO_STYLES = {
    "gamcal": {
        "color": PLOTLY_COLORS[0],
        "marker": "D",
        "label": "GAMCAL",
        "linewidth": 0.9,
        "markersize": 2,
        "zorder": 10,
    },
    "supg_it": {
        "color": PLOTLY_COLORS[1],
        "marker": "o",
        "label": "SUPG-IT",
        "linewidth": 0.9,
        "markersize": 2,
        "zorder": 9,
    },
    "supg_joint": {
        "color": PLOTLY_COLORS[2],
        "marker": "^",
        "label": "SUPG-SP",
        "linewidth": 0.7,
        "markersize": 1.8,
        "zorder": 8,
    },
    "supg_rt": {
        "color": PLOTLY_COLORS[3],
        "marker": "s",
        "label": "SUPG",
        "linewidth": 0.7,
        "markersize": 1.8,
        "linestyle": "--",
        "zorder": 7,
    },
}


def _build_algo_styles(scheme: str = "plotly") -> dict:
    """Build algorithm visual styles for the plotly palette."""
    c = {
        "gamcal": PLOTLY_COLORS[0],
        "supg_it": PLOTLY_COLORS[1],
        "supg_joint": PLOTLY_COLORS[2],
        "supg_rt": PLOTLY_COLORS[3],
    }
    return {
        "gamcal": {
            "color": c["gamcal"],
            "marker": "D",
            "label": "GAMCAL",
            "linewidth": 0.9,
            "markersize": 2,
            "zorder": 10,
        },
        "supg_it": {
            "color": c["supg_it"],
            "marker": "o",
            "label": "SUPG-IT",
            "linewidth": 0.9,
            "markersize": 2,
            "zorder": 9,
        },
        "supg_joint": {
            "color": c["supg_joint"],
            "marker": "^",
            "label": "SUPG-SP",
            "linewidth": 0.7,
            "markersize": 1.8,
            "zorder": 8,
        },
        "supg_rt": {
            "color": c["supg_rt"],
            "marker": "s",
            "label": "SUPG",
            "linewidth": 0.7,
            "markersize": 1.8,
            "linestyle": "--",
            "zorder": 7,
        },
    }


def _build_prec_recall_styles(scheme: str = "plotly") -> dict:
    """Build precision/recall decomposition styles with 4 distinct colors."""
    if scheme != "plotly":
        raise ValueError(f"Unknown color scheme: {scheme!r}")
    return {
        ("gamcal", "precision"): {
            "color": "#636EFA",
            "marker": "D",
            "linestyle": "-",
            "label": "GAMCAL Prec",
        },
        ("gamcal", "recall"): {
            "color": "#93A0FC",
            "marker": "d",
            "linestyle": "--",
            "label": "GAMCAL Rec",
        },
        ("supg_it", "precision"): {
            "color": "#EF553B",
            "marker": "o",
            "linestyle": "-",
            "label": "SUPG-IT Prec",
        },
        ("supg_it", "recall"): {
            "color": "#F59383",
            "marker": "s",
            "linestyle": "--",
            "label": "SUPG-IT Rec",
        },
    }


DATASET_LABELS = {
    "arxiv": "ArXiv",
    "boolq": "BoolQ",
    "imdb": "IMDB",
    "mmlu": "MMLU",
    "nyt": "NYT",
    "sst2": "SST-2",
}

DEFAULT_DATASETS = list(CURATED_DATASET_NAMES)


def _apply_pub_style():
    """Apply publication rcParams; fall back to non-LaTeX fonts if TeX is unavailable."""
    params = dict(PUB_RCPARAMS)
    try:
        plt.rcParams.update(params)
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_xlabel(r"$x$")
        fig.canvas.draw()
        plt.close(fig)
    except Exception:
        params["text.usetex"] = False
        params.pop("text.latex.preamble", None)
        plt.rcParams.update(params)
    else:
        plt.rcParams.update(params)


def _auto_y_min(df: pd.DataFrame, datasets: list[str], metric: str, algorithms: list[str]) -> float:
    """Compute a y_min that clips empty space below the data."""
    vals = []
    for ds in datasets:
        for algo in algorithms:
            sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo)]
            if sub.empty:
                continue
            agg = sub.groupby("sweep_value")[metric].mean()
            vals.append(agg.min())
    if not vals:
        return 0.0
    raw_min = min(vals)
    return max(0.0, math.floor(raw_min / 0.05) * 0.05 - 0.02)


def _plot_algo_curve(
    ax, df: pd.DataFrame, dataset: str, algorithm: str, metric: str, algo_styles: dict | None = None
):
    """Plot one algorithm's Pareto curve with horizontal+vertical error bars."""
    if algo_styles is None:
        algo_styles = ALGO_STYLES
    style = algo_styles[algorithm]
    sub = df[(df["dataset"] == dataset) & (df["algorithm"] == algorithm)]
    if sub.empty:
        return

    agg = (
        sub.groupby("sweep_value")
        .agg(
            deleg_mean=("delegation_rate", "mean"),
            deleg_std=("delegation_rate", "std"),
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
        )
        .reset_index()
        .sort_values("deleg_mean")
    )

    ax.errorbar(
        agg["deleg_mean"],
        agg["metric_mean"],
        xerr=agg["deleg_std"],
        yerr=agg["metric_std"],
        color=style["color"],
        marker=style["marker"],
        markersize=style["markersize"],
        linewidth=style["linewidth"],
        linestyle=style.get("linestyle", "-"),
        label=style["label"],
        zorder=style["zorder"],
        capsize=1.5,
        capthick=0.8,
        elinewidth=0.7,
        alpha=0.85,
    )


def _add_baselines(ax, df: pd.DataFrame, dataset: str, metric: str):
    """Add proxy-only and oracle-only as horizontal reference lines."""
    for baseline, ls, label in [
        ("proxy_only", (0, (5, 5)), "Proxy-only"),
        ("oracle_only", (0, (1, 1)), "Oracle"),
    ]:
        sub = df[(df["dataset"] == dataset) & (df["algorithm"] == baseline)]
        if sub.empty:
            continue
        val = sub[metric].mean()
        ax.axhline(
            val, color="#7f8c8d", linestyle=ls, linewidth=0.8, alpha=0.7, zorder=1, label=label
        )


def plot_hero(
    df: pd.DataFrame,
    datasets: list[str],
    out_dir: Path,
    algorithms: list[str] = None,
    y_min: float | str = "auto",
    y_max: float = 1.02,
    layout: str = "auto",
    color_scheme: str = "plotly",
):
    """Main paper figure: F1 vs delegation rate."""
    if algorithms is None:
        algorithms = ["gamcal", "supg_it"]

    algo_styles = _build_algo_styles(color_scheme)

    n = len(datasets)
    if layout == "auto":
        if n <= 3:
            nrows, ncols = 1, n
        else:
            nrows, ncols = 2, math.ceil(n / 2)
    elif layout == "2x3":
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 1, n

    fig_w = 7.0
    panel_h = 1.85
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, panel_h * nrows + 0.3),
        squeeze=False,
    )

    for idx, ds in enumerate(datasets):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        _add_baselines(ax, df, ds, "f1")
        for algo in algorithms:
            _plot_algo_curve(ax, df, ds, algo, "f1", algo_styles=algo_styles)

        if y_min == "auto":
            panel_ymin = _auto_y_min(df, [ds], "f1", algorithms)
        else:
            panel_ymin = float(y_min)

        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(panel_ymin, y_max)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

        if col == 0:
            ax.set_ylabel(r"$F_1$ Score")
        else:
            ax.set_ylabel("")
        if row == nrows - 1:
            ax.set_xlabel("Delegation Rate")
        else:
            ax.set_xlabel("")

    for idx in range(len(datasets), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    handles, labels = [], []
    for row in range(nrows):
        for col in range(ncols):
            panel_handles, panel_labels = axes[row, col].get_legend_handles_labels()
            if panel_handles:
                handles, labels = panel_handles, panel_labels
                break
        if handles:
            break

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.01),
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{nrows}x{ncols}" if nrows > 1 or ncols > 3 else ""
    color_sfx = f"_{color_scheme}" if color_scheme != "plotly" else ""
    for ext in ["pdf", "png"]:
        path = out_dir / f"hero_pareto_f1{suffix}{color_sfx}.{ext}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def plot_full_grid(
    df: pd.DataFrame,
    datasets: list[str],
    out_dir: Path,
    algorithms: list[str] = None,
    y_min: float | str = "auto",
    y_max: float = 1.02,
    color_scheme: str = "plotly",
):
    """Appendix figure: all datasets x all metrics, all algorithms."""
    if algorithms is None:
        algorithms = ["gamcal", "supg_it", "supg_joint", "supg_rt"]

    algo_styles = _build_algo_styles(color_scheme)

    metrics = [
        ("f1", "F1"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    n_ds = len(datasets)
    n_met = len(metrics)

    fig, axes = plt.subplots(
        n_ds,
        n_met,
        figsize=(2.4 * n_met + 0.3, 2.0 * n_ds + 0.4),
        sharex=True,
        squeeze=False,
    )

    for row, ds in enumerate(datasets):
        for col, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[row, col]

            if y_min == "auto":
                ym = _auto_y_min(df, [ds], metric_col, algorithms)
            else:
                ym = float(y_min)

            _add_baselines(ax, df, ds, metric_col)
            for algo in algorithms:
                _plot_algo_curve(ax, df, ds, algo, metric_col, algo_styles=algo_styles)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(ym, y_max)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

            if row == 0:
                ax.set_title(metric_label)
            if col == 0:
                ax.set_ylabel(DATASET_LABELS.get(ds, ds), fontweight="bold")
            else:
                ax.set_ylabel("")
            if row == n_ds - 1:
                ax.set_xlabel("Delegation Rate")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    color_sfx = f"_{color_scheme}" if color_scheme != "plotly" else ""
    for ext in ["pdf", "png"]:
        path = out_dir / f"appendix_pareto_grid{color_sfx}.{ext}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def plot_prec_recall_decomposition(
    df: pd.DataFrame,
    datasets: list[str],
    out_dir: Path,
    algorithms: list[str] = None,
    y_min: float | str = "auto",
    y_max: float = 1.02,
    layout: str = "auto",
    color_scheme: str = "plotly",
):
    """Appendix figure: precision and recall curves per dataset (2x3 grid)."""
    if algorithms is None:
        algorithms = ["gamcal", "supg_it"]

    n = len(datasets)
    if layout == "auto":
        if n <= 3:
            nrows, ncols = 1, n
        else:
            nrows, ncols = 2, math.ceil(n / 2)
    elif layout == "2x3":
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 1, n

    pr_styles = _build_prec_recall_styles(color_scheme)

    fig_w = 7.0
    panel_h = 1.85
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, panel_h * nrows + 0.3),
        squeeze=False,
    )

    for idx, ds in enumerate(datasets):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if y_min == "auto":
            ym = min(
                _auto_y_min(df, [ds], "precision", algorithms),
                _auto_y_min(df, [ds], "recall", algorithms),
            )
        else:
            ym = float(y_min)

        for algo in algorithms:
            for metric in ["precision", "recall"]:
                st = pr_styles.get((algo, metric), {})
                sub = df[(df["dataset"] == ds) & (df["algorithm"] == algo)]
                if sub.empty:
                    continue
                agg = (
                    sub.groupby("sweep_value")
                    .agg(
                        deleg_mean=("delegation_rate", "mean"),
                        deleg_std=("delegation_rate", "std"),
                        metric_mean=(metric, "mean"),
                        metric_std=(metric, "std"),
                    )
                    .reset_index()
                    .sort_values("deleg_mean")
                )

                ax.errorbar(
                    agg["deleg_mean"],
                    agg["metric_mean"],
                    xerr=agg["deleg_std"],
                    yerr=agg["metric_std"],
                    color=st.get("color", "gray"),
                    linestyle=st.get("linestyle", "-"),
                    marker=st.get("marker", "."),
                    linewidth=0.9,
                    label=st.get("label", ""),
                    markersize=3,
                    capsize=1.5,
                    capthick=0.7,
                    elinewidth=0.6,
                    alpha=0.85,
                )

        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(ym, y_max)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        if col == 0:
            ax.set_ylabel("Precision / Recall")
        else:
            ax.set_ylabel("")
        if row == nrows - 1:
            ax.set_xlabel("Delegation Rate")
        else:
            ax.set_xlabel("")

    for idx in range(len(datasets), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        frameon=True,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{nrows}x{ncols}" if nrows > 1 or ncols > 3 else ""
    color_sfx = f"_{color_scheme}" if color_scheme != "plotly" else ""
    for ext in ["pdf", "png"]:
        path = out_dir / f"appendix_prec_recall{suffix}{color_sfx}.{ext}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Publication-grade Pareto figures for Experiment 1"
    )
    parser.add_argument("--input", type=str, default=str(RESULTS_CSV))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "1xN", "2x3"])
    parser.add_argument(
        "--y-min", type=str, default="auto", help="Y-axis minimum: 'auto' or float (e.g. 0.3)"
    )
    parser.add_argument("--y-max", type=float, default=1.02)
    parser.add_argument(
        "--algorithms",
        type=str,
        default="gamcal,supg_it",
        help="Algorithms for hero figure (comma-separated)",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--colors",
        type=str,
        default="plotly",
        choices=["plotly"],
        help="Color scheme (plotly only in public artifact)",
    )
    parser.add_argument("--hero-only", action="store_true", help="Only generate hero figure")
    args = parser.parse_args()

    _apply_pub_style()

    df = pd.read_csv(args.input)
    datasets = [d.strip() for d in args.datasets.split(",")]
    algorithms = [a.strip() for a in args.algorithms.split(",")]
    out_dir = Path(args.output_dir)
    y_min = args.y_min if args.y_min == "auto" else float(args.y_min)
    color_schemes = ["plotly"]

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Datasets: {datasets}")
    print(f"Algorithms (hero): {algorithms}")
    print(f"Y-min: {y_min}, Y-max: {args.y_max}")
    print(f"Color scheme(s): {color_schemes}")
    print(f"Output: {out_dir}")

    for scheme in color_schemes:
        if len(color_schemes) > 1:
            print(f"Color scheme: {scheme}")

        print("Generating hero figure...")
        plot_hero(
            df,
            datasets,
            out_dir,
            algorithms,
            y_min=y_min,
            y_max=args.y_max,
            layout=args.layout,
            color_scheme=scheme,
        )

        if not args.hero_only:
            print("Generating appendix full grid...")
            plot_full_grid(
                df, datasets, out_dir, y_min=y_min, y_max=args.y_max, color_scheme=scheme
            )

            print("Generating appendix precision/recall decomposition...")
            plot_prec_recall_decomposition(
                df,
                datasets,
                out_dir,
                y_min=y_min,
                y_max=args.y_max,
                layout=args.layout,
                color_scheme=scheme,
            )

    print("Done!")


if __name__ == "__main__":
    main()
