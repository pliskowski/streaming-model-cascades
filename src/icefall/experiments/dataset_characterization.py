"""Dataset Characterization: compute key statistics for all cached datasets.

Produces Table 1 for the paper: size, class balance, proxy accuracy,
proxy calibration quality (ECE), and proxy-oracle agreement.

Usage:
    uv run -m icefall.experiments.dataset_characterization
    uv run -m icefall.experiments.dataset_characterization --datasets all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from icefall.datasets import (
    ALL_DATASETS,
    ORACLE_RESULT_COL,
    PROXY_SCORE_COL,
    get_datasets,
    load_dataset,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
EXPERIMENT_NAME = "dataset_characterization"


def _make_run_dir(base: Path, name: str) -> Path:
    """Create a stable run directory: base/name/."""
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions into bins by predicted probability,
    then measures the gap between average confidence and accuracy in each bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / total) * abs(avg_accuracy - avg_confidence)

    return ece


def characterize_dataset(name: str) -> dict:
    """Compute characterization statistics for a single dataset."""
    df = load_dataset(name)
    n = len(df)

    proxy_scores = df[PROXY_SCORE_COL].values
    oracle_labels = df[ORACLE_RESULT_COL].values.astype(bool)

    positive_rate = oracle_labels.mean()

    proxy_preds = proxy_scores >= 0.5
    proxy_accuracy = (proxy_preds == oracle_labels).mean()

    agreement = proxy_accuracy

    ece = expected_calibration_error(oracle_labels.astype(float), proxy_scores)

    score_mean = proxy_scores.mean()
    score_std = proxy_scores.std()

    proxy_f1 = f1_score(oracle_labels, proxy_preds, zero_division=0)
    proxy_precision = precision_score(oracle_labels, proxy_preds, zero_division=0)
    proxy_recall = recall_score(oracle_labels, proxy_preds, zero_division=0)

    return {
        "dataset": name,
        "n_rows": n,
        "positive_rate": positive_rate,
        "proxy_accuracy": proxy_accuracy,
        "proxy_f1": proxy_f1,
        "proxy_precision": proxy_precision,
        "proxy_recall": proxy_recall,
        "proxy_ece": ece,
        "proxy_score_mean": score_mean,
        "proxy_score_std": score_std,
        "agreement": agreement,
        "description": ALL_DATASETS[name].description,
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset characterization")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Dataset selector: 'all', 'curated', or comma-separated names",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<experiment>/<timestamp>)",
    )
    args = parser.parse_args()

    datasets = get_datasets(args.datasets)

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _make_run_dir(RESULTS_DIR, EXPERIMENT_NAME)

    output_path = run_dir / f"{EXPERIMENT_NAME}.csv"
    print(f"Output directory: {run_dir}")

    print(f"Characterizing {len(datasets)} datasets...\n")

    results = []
    for name in sorted(datasets.keys()):
        try:
            stats = characterize_dataset(name)
            results.append(stats)
            print(
                f"  {name:<20} n={stats['n_rows']:>6}  pos={stats['positive_rate']:.2f}  "
                f"acc={stats['proxy_accuracy']:.3f}  F1={stats['proxy_f1']:.3f}  "
                f"ECE={stats['proxy_ece']:.3f}"
            )
        except Exception as e:
            print(f"  {name:<20} FAILED: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 100)
    print("DATASET CHARACTERIZATION SUMMARY")
    print("=" * 100)
    fmt = "{:<20} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}"
    print(fmt.format("Dataset", "Rows", "Pos%", "Acc", "F1", "Prec", "Recall", "ECE"))
    print("-" * 100)
    for _, row in df.iterrows():
        print(
            fmt.format(
                row["dataset"],
                row["n_rows"],
                f"{row['positive_rate']:.2f}",
                f"{row['proxy_accuracy']:.3f}",
                f"{row['proxy_f1']:.3f}",
                f"{row['proxy_precision']:.3f}",
                f"{row['proxy_recall']:.3f}",
                f"{row['proxy_ece']:.3f}",
            )
        )

    _write_dataset_table_latex(df, run_dir)


def _write_dataset_table_latex(df: pd.DataFrame, run_dir: Path) -> None:
    """Write LaTeX rows for paper Table 1 (tab:datasets)."""
    task_labels = {
        "mmlu": r"\texttt{AI\_CLASSIFY}",
        "boolq": r"\texttt{AI\_FILTER}",
        "imdb": r"\texttt{AI\_FILTER}",
        "arxiv": r"\texttt{AI\_FILTER}",
        "sst2": r"\texttt{AI\_FILTER}",
        "nyt": r"\texttt{AI\_JOIN}",
    }
    name_labels = {
        "mmlu": "MMLU",
        "boolq": "BoolQ",
        "imdb": "IMDB",
        "arxiv": "ArXiv",
        "sst2": "SST-2",
        "nyt": "NYT",
    }
    lines = [
        "% Dataset characteristics (tab:datasets)",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Task & Rows & Pos\\% & Proxy F1 & ECE \\\\",
        "\\midrule",
    ]
    for _, row in df.sort_values("dataset").iterrows():
        ds = row["dataset"]
        pct = 100 * row["positive_rate"]
        if abs(pct - round(pct)) < 0.05:
            pos_pct = f"{round(pct):.0f}\\%"
        else:
            pos_pct = f"{pct:.1f}\\%"
        lines.append(
            f"{name_labels.get(ds, ds)} & {task_labels.get(ds, '')} & "
            f"{int(row['n_rows']):,} & {pos_pct} & "
            f"{row['proxy_f1']:.3f} & {row['proxy_ece']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path = run_dir / "dataset_characterization.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
