"""Experiment 1: Cost-Quality Pareto Frontier.

Sweeps each algorithm's native control parameter to trace the Pareto frontier:
  - GAMCAL: alpha (error-vs-cost tradeoff) in [0.10, 0.80], budget=1.0
  - SUPG-IT: shared_target (recall=precision) in [0.55, 0.95]
  - SUPG-SP: same sweep, single-pass
  - SUPG-RT: recall_target sweep, single threshold
  - Proxy-only and Oracle-only baselines

Alternative mode (--gamcal-sweep budget): sweep budget_percentage instead of
alpha.  Useful for studying GAMCAL under budget constraints.

Usage:
    uv run -m icefall.experiments.exp1_pareto
    uv run -m icefall.experiments.exp1_pareto --datasets arxiv,boolq --n-seeds 5
    uv run -m icefall.experiments.exp1_pareto --gamcal-sweep budget
"""

from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from icefall.datasets import get_datasets, load_dataset
from icefall.harness import (
    CascadeConfig,
    _result_to_dict,
    run_single,
)


def _log(msg: str) -> None:
    """Print a timestamped, immediately-flushed log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
EXPERIMENT_NAME = "exp1_pareto"


def _make_run_dir(base: Path, name: str, batch_size: int = 4096, dop: int = 1) -> Path:
    """Create a stable run directory: base/name/batch-<B>_dop-<D>/."""
    config_tag = f"batch-{batch_size}_dop-{dop}"
    run_dir = base / name / config_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# GAMCAL alpha sweep at full budget (paper Figure 2 / Table 2–4 sweep).
GAMCAL_ALPHA_VALUES = [
    0.100,
    0.133,
    0.167,
    0.200,
    0.233,
    0.267,
    0.300,
    0.333,
    0.367,
    0.400,
    0.433,
    0.467,
    0.500,
    0.533,
    0.567,
    0.600,
    0.633,
    0.667,
    0.700,
    0.733,
    0.767,
    0.800,
]

# GAMCAL budget sweep at fixed alpha=0.5.
GAMCAL_BUDGET_VALUES = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.80,
    1.00,
]

SHARED_TARGET_VALUES = [
    0.550,
    0.575,
    0.600,
    0.625,
    0.650,
    0.675,
    0.700,
    0.725,
    0.750,
    0.775,
    0.800,
    0.825,
    0.850,
    0.875,
    0.900,
    0.925,
    0.950,
]

RECALL_TARGET_VALUES = SHARED_TARGET_VALUES


def build_algorithms_config(gamcal_sweep: str = "alpha") -> dict:
    """Build the algorithms configuration for the sweep.

    Args:
        gamcal_sweep: "alpha" (default) sweeps alpha at budget=1.0;
                      "budget" sweeps sampling_percentage at alpha=0.5.
    """
    if gamcal_sweep == "alpha":
        gamcal_cfg = {
            "sweep_param": "alpha",
            "sweep_values": GAMCAL_ALPHA_VALUES,
            "fixed": {"sampling_percentage": 1.0},
        }
    elif gamcal_sweep == "budget":
        gamcal_cfg = {
            "sweep_param": "sampling_percentage",
            "sweep_values": GAMCAL_BUDGET_VALUES,
            "fixed": {"alpha": 0.5},
        }
    else:
        raise ValueError(f"Unknown gamcal_sweep mode: {gamcal_sweep!r}")

    return {
        "gamcal": gamcal_cfg,
        "supg_it": {
            "sweep_param": "recall_target",
            "sweep_values": SHARED_TARGET_VALUES,
        },
        "supg_joint": {
            "sweep_param": "recall_target",
            "sweep_values": SHARED_TARGET_VALUES,
        },
        "supg_rt": {
            "sweep_param": "recall_target",
            "sweep_values": RECALL_TARGET_VALUES,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Pareto Frontier")
    parser.add_argument(
        "--datasets",
        type=str,
        default="curated",
        help="Dataset selector: 'curated', 'all', or comma-separated names",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10, help="Number of random seeds per configuration"
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size (rowset size)")
    parser.add_argument("--dop", type=int, default=1, help="Degree of parallelism")
    parser.add_argument(
        "--gamcal-sweep",
        type=str,
        default="alpha",
        choices=["alpha", "budget"],
        help="GAMCAL sweep mode: 'alpha' (default, budget=1.0) or 'budget' (alpha=0.5)",
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
        run_dir = _make_run_dir(
            RESULTS_DIR, EXPERIMENT_NAME, batch_size=args.batch_size, dop=args.dop
        )

    csv_path = run_dir / f"{EXPERIMENT_NAME}.csv"
    tables_dir = run_dir / "tables"

    algorithms_config = build_algorithms_config(gamcal_sweep=args.gamcal_sweep)
    n_algos = len(algorithms_config) + 2  # +2 for baselines
    total_configs_per_dataset = (
        sum(len(a["sweep_values"]) * args.n_seeds for a in algorithms_config.values())
        + 2 * args.n_seeds
    )  # baselines
    dataset_names = sorted(datasets.keys())

    base_config = CascadeConfig(
        batch_size=args.batch_size,
        dop=args.dop,
    )

    _log("Experiment 1: Pareto Frontier")
    _log(f"  Output:     {run_dir}")
    _log(f"  Datasets:   {', '.join(dataset_names)} ({len(dataset_names)} total)")
    _log(f"  batch_size: {args.batch_size}, dop: {args.dop}")
    _log(f"  Algorithms: {', '.join(algorithms_config.keys())} + baselines ({n_algos} total)")
    _log(
        f"  GAMCAL:     sweep={args.gamcal_sweep}, gamcal_batch_size={base_config.gamcal_batch_size}"
    )
    _log(
        f"  SUPG-IT:    supg_it_batch_size={base_config.supg_it_batch_size}, failure_prob={base_config.failure_probability}"
    )
    _log(f"  Seeds:      {args.n_seeds}")
    _log(f"  Configs/dataset: {total_configs_per_dataset}")
    _log(f"  Total runs: {total_configs_per_dataset * len(dataset_names)}")

    all_results = []
    exp_t0 = time.time()

    for i, dataset_name in enumerate(dataset_names, 1):
        _log(f"{'=' * 70}")
        _log(f"Dataset [{i}/{len(dataset_names)}]: {dataset_name}")
        _log(f"{'=' * 70}")

        t0 = time.time()
        df = run_algorithms_sweep_with_shared_target(
            dataset_name=dataset_name,
            algorithms_config=algorithms_config,
            n_seeds=args.n_seeds,
            base_config=base_config,
        )
        elapsed = time.time() - t0
        total_elapsed = time.time() - exp_t0
        _log(
            f"Done {dataset_name} in {elapsed:.0f}s  "
            f"(total elapsed: {total_elapsed:.0f}s, {len(df)} result rows)"
        )

        all_results.append(df)

        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(csv_path, index=False)
        _log(f"Saved incremental results to {csv_path} ({len(combined)} rows)")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(csv_path, index=False)
    total_elapsed = time.time() - exp_t0
    _log(f"{'=' * 70}")
    _log(f"DONE -- {len(combined)} rows, {total_elapsed:.0f}s total")
    _log(f"Results: {csv_path}")
    _log(f"{'=' * 70}")

    _generate_tables(csv_path, tables_dir)


def _generate_tables(csv_path: Path, tables_dir: Path) -> None:
    """Generate summary tables from results."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    try:
        from icefall.figures.tables import generate_pareto_summary

        generate_pareto_summary(csv_path, tables_dir)
        _log("Tables done")
    except Exception as e:
        _log(f"WARNING: table generation failed: {e}")


def run_algorithms_sweep_with_shared_target(
    dataset_name: str,
    algorithms_config: dict,
    n_seeds: int,
    base_config: CascadeConfig,
) -> pd.DataFrame:
    """Run sweeps, ensuring precision_target = recall_target for SUPG variants.

    The standard run_algorithms_sweep doesn't sync precision_target with recall_target.
    This wrapper handles that by running each algorithm individually.
    """
    data = load_dataset(dataset_name)
    _log(f"  Loaded {len(data)} rows")
    all_results = []

    algo_names = list(algorithms_config.keys())
    for algo_idx, algo_name in enumerate(algo_names, 1):
        algo_cfg = algorithms_config[algo_name]
        sweep_param = algo_cfg["sweep_param"]
        sweep_values = algo_cfg["sweep_values"]
        fixed_overrides = algo_cfg.get("fixed", {})

        total = len(sweep_values) * n_seeds
        done = 0
        algo_t0 = time.time()

        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_overrides.items())
        _log(
            f"  Algorithm [{algo_idx}/{len(algo_names)}]: {algo_name}  "
            f"({len(sweep_values)} values x {n_seeds} seeds = {total} runs)"
            + (f"  fixed: {fixed_str}" if fixed_str else "")
        )

        for val in sweep_values:
            for seed in range(n_seeds):
                cfg = copy.deepcopy(base_config)
                cfg.algorithm = algo_name
                cfg.seed = seed

                for k, v in fixed_overrides.items():
                    setattr(cfg, k, v)

                setattr(cfg, sweep_param, val)

                if algo_name in ("supg_it", "supg_joint") and sweep_param == "recall_target":
                    cfg.precision_target = val

                result = run_single(dataset_name, config=cfg, data=data)
                result_dict = _result_to_dict(result)
                result_dict["sweep_param"] = sweep_param
                result_dict["sweep_value"] = val
                all_results.append(result_dict)
                done += 1

            algo_elapsed = time.time() - algo_t0
            eta = algo_elapsed / done * (total - done) if done > 0 else 0
            _log(
                f"    {sweep_param}={val:.3f}  [{done}/{total}]  "
                f"F1={result.f1:.3f}  deleg={result.delegation_rate:.3f}  "
                f"ETA {eta:.0f}s"
            )

        algo_elapsed = time.time() - algo_t0
        _log(f"  Finished {algo_name} in {algo_elapsed:.0f}s")

    _log("  Running baselines (proxy_only, oracle_only)...")
    for baseline in ["proxy_only", "oracle_only"]:
        for seed in range(n_seeds):
            result = run_single(dataset_name, data=data, algorithm=baseline, seed=seed)
            result_dict = _result_to_dict(result)
            result_dict["sweep_param"] = "none"
            result_dict["sweep_value"] = None
            all_results.append(result_dict)
    _log("  Baselines done")

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    main()
