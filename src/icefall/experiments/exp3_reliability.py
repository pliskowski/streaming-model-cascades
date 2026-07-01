"""Experiment 3: Target Reliability — Does SUPG-IT meet user-specified targets?

Runs an asymmetric grid of (t_P, t_R) pairs on SUPG-IT and checks whether
achieved precision/recall meet the requested targets.

Usage:
    uv run -m icefall.experiments.exp3_reliability
    uv run -m icefall.experiments.exp3_reliability --datasets imdb,mmlu --n-seeds 5
"""

from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from icefall.datasets import load_dataset
from icefall.harness import CascadeConfig, _result_to_dict, run_single

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

TARGET_VALUES_COARSE = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
TARGET_VALUES_FINE = [round(0.55 + i * 0.025, 3) for i in range(17)]  # 0.55 to 0.95 step 0.025
DEFAULT_DATASETS = ["arxiv", "imdb", "mmlu", "boolq", "sst2"]

_log_path: Path | None = None


def _log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_path:
        with open(_log_path, "a") as f:
            f.write(line + "\n")


def build_target_grid(fine: bool = False) -> list[tuple[float, float]]:
    vals = TARGET_VALUES_FINE if fine else TARGET_VALUES_COARSE
    return [(tp, tr) for tp in vals for tr in vals]


def main():
    parser = argparse.ArgumentParser(description="Exp 3: Target Reliability")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dop", type=int, default=1)
    parser.add_argument(
        "--fine-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fine 0.025-step grid (17x17=289 pairs) instead of coarse 6x6=36",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    grid = build_target_grid(fine=args.fine_grid)
    n_configs = len(grid) * args.n_seeds * len(dataset_names)

    results_root = RESULTS_DIR / "exp3_reliability"
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = results_root / "batch-4096_dop-1"
    run_dir.mkdir(parents=True, exist_ok=True)

    global _log_path
    _log_path = run_dir / "exp3.log"

    csv_path = run_dir / "exp3_reliability.csv"

    base_config = CascadeConfig(
        algorithm="supg_it",
        batch_size=args.batch_size,
        dop=args.dop,
    )

    _log("Experiment 3: Target Reliability")
    _log(f"  Output:     {run_dir}")
    _log(f"  Datasets:   {', '.join(dataset_names)}")
    grid_label = "fine 0.025-step" if args.fine_grid else "coarse"
    n_vals = len(TARGET_VALUES_FINE) if args.fine_grid else len(TARGET_VALUES_COARSE)
    _log(f"  Grid:       {n_vals}x{n_vals} = {len(grid)} (t_P, t_R) pairs ({grid_label})")
    _log(f"  Seeds:      {args.n_seeds}")
    _log(f"  batch_size: {args.batch_size}, dop: {args.dop}")
    _log(
        f"  supg_it_batch_size: {base_config.supg_it_batch_size}, "
        f"failure_prob: {base_config.failure_probability}"
    )
    _log(f"  Total runs: {n_configs}")

    all_results: list[dict] = []
    exp_t0 = time.time()

    for ds_idx, ds_name in enumerate(dataset_names, 1):
        ds_t0 = time.time()
        _log(f"\nDataset [{ds_idx}/{len(dataset_names)}]: {ds_name}")
        data = load_dataset(ds_name)
        _log(f"  Loaded {len(data)} rows")

        ds_runs = len(grid) * args.n_seeds
        ds_done = 0

        for pair_idx, (tp, tr) in enumerate(grid, 1):
            for seed in range(args.n_seeds):
                cfg = copy.deepcopy(base_config)
                cfg.precision_target = tp
                cfg.recall_target = tr
                cfg.seed = seed

                result = run_single(ds_name, config=cfg, data=data)
                row = _result_to_dict(result)
                row["target_precision"] = tp
                row["target_recall"] = tr
                row["achieved_precision"] = result.precision
                row["achieved_recall"] = result.recall
                row["precision_met"] = result.precision >= tp
                row["recall_met"] = result.recall >= tr
                row["both_met"] = (result.precision >= tp) and (result.recall >= tr)
                all_results.append(row)
                ds_done += 1

            ds_elapsed = time.time() - ds_t0
            eta_ds = ds_elapsed / ds_done * (ds_runs - ds_done) if ds_done > 0 else 0
            _log(
                f"  (t_P={tp:.2f}, t_R={tr:.2f})  [{pair_idx}/{len(grid)}]  "
                f"P={result.precision:.3f} R={result.recall:.3f} "
                f"deleg={result.delegation_rate:.3f}  "
                f"ETA_ds={eta_ds:.0f}s"
            )

        ds_elapsed = time.time() - ds_t0
        _log(f"  Dataset {ds_name} done in {ds_elapsed:.0f}s")

        combined = pd.DataFrame(all_results)
        combined.to_csv(csv_path, index=False)
        _log(f"  Saved incremental results ({len(combined)} rows)")

    total_elapsed = time.time() - exp_t0
    combined = pd.DataFrame(all_results)
    combined.to_csv(csv_path, index=False)
    _log(f"DONE -- {len(combined)} rows, {total_elapsed:.0f}s total")
    _log(f"Results: {csv_path}")

    _print_summary(combined)

    _generate_reliability_table(combined, run_dir)


def _print_summary(df: pd.DataFrame):
    """Print a quick satisfaction-rate summary."""
    _log("  Satisfaction Rate Summary (fraction of seeds meeting target)")

    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]
        sat = (
            ds_df.groupby(["target_precision", "target_recall"])
            .agg(
                prec_sat=("precision_met", "mean"),
                rec_sat=("recall_met", "mean"),
                both_sat=("both_met", "mean"),
                mean_deleg=("delegation_rate", "mean"),
            )
            .reset_index()
        )

        _log(f"\n  Dataset: {ds}")
        _log(f"  {'t_P':>5} {'t_R':>5} | {'P_sat':>6} {'R_sat':>6} {'Both':>6} | {'Deleg':>6}")

        for _, r in sat.iterrows():
            _log(
                f"  {r['target_precision']:5.2f} {r['target_recall']:5.2f} | "
                f"{r['prec_sat']:6.0%} {r['rec_sat']:6.0%} {r['both_sat']:6.0%} | "
                f"{r['mean_deleg']:6.3f}"
            )


def _generate_reliability_table(df: pd.DataFrame, run_dir: Path) -> None:
    """Write LaTeX/Markdown summary matching paper Table 5 (tab:reliability)."""
    rows = []
    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]
        sym_df = ds_df[ds_df["target_precision"] == ds_df["target_recall"]]
        all_sat = 100.0 * ds_df["both_met"].mean()
        sym_sat = 100.0 * sym_df["both_met"].mean() if len(sym_df) else float("nan")
        failures = int((~ds_df["both_met"]).sum())
        deleg = 100.0 * ds_df["delegation_rate"].mean()
        rows.append(
            {
                "dataset": ds,
                "all_sat_pct": all_sat,
                "sym_sat_pct": sym_sat,
                "failures": failures,
                "delegation_pct": deleg,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "reliability_summary.csv", index=False)

    labels = {
        "arxiv": "ArXiv",
        "imdb": "IMDB",
        "mmlu": "MMLU",
        "boolq": "BoolQ",
        "sst2": "SST-2",
    }
    tex_lines = [
        "% Reliability table (tab:reliability)",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & All & $t_P{=}t_R$ & Failures & Delegation (\\%) \\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        name = labels.get(r["dataset"], r["dataset"])
        tex_lines.append(
            f"{name} & {r['all_sat_pct']:.1f} & {r['sym_sat_pct']:.1f} & "
            f"{int(r['failures'])} & {r['delegation_pct']:.1f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path = run_dir / "reliability_summary.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    _log(f"Reliability table saved to {tex_path}")


if __name__ == "__main__":
    main()
