"""Self-contained evaluation harness for cascade paper experiments.

Plain Python API + argparse CLI for running cascade algorithms on pre-scored datasets:
  SUPG-RT, SUPG-SP, SUPG-IT, GAMCAL, proxy-only, and oracle-only.
Deterministic seeding; structured CSV output with one row per (algorithm, parameter, seed, dataset).

Usage as library:
    from icefall.harness import run_single, run_sweep

    result = run_single("mmlu", algorithm="supg_it", recall_target=0.8, seed=42)
    results_df = run_sweep("mmlu", algorithm="supg_it",
                           sweep_param="recall_target", sweep_values=[0.6, 0.7, 0.8, 0.9],
                           n_seeds=10)
"""

from __future__ import annotations

import copy
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# Suppress verbose loguru logging from cascade internals (e.g. GAMCAL threshold updates)
from loguru import logger as _loguru_logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

_loguru_logger.remove()
_LOGURU_LOG_LEVEL = os.environ.get("CASCADE_LOG_LEVEL", "WARNING")
_loguru_logger.add(sys.stderr, level=_LOGURU_LOG_LEVEL)

from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
from icefall.cascades.supg_sp_cascade import SupgSpCascade
from icefall.cascades.model_cascade import ModelCascade
from icefall.cascades.supg_recall_only_cascade import SUPGRecallOnlyCascade
from icefall.cascades.supg_it_cascade import SupgItCascade
from icefall.datasets import (
    ORACLE_RESULT_COL,
    PROXY_SCORE_COL,
    load_dataset,
)
from icefall.stream_runtime.engine import PipelineEngine
from icefall.stream_runtime.result_sink import ResultSink
from icefall.stream_runtime.table_scan import TableScan
from icefall.stream_runtime.expr import RowExpr
from icefall.stream_runtime.batch_operator import BatchOperator

ALGORITHMS = [
    "supg_rt",  # SUPG Recall-Only (original SUPG Algorithm 5)
    "supg_joint",  # SUPG-SP (single-pass joint targets)
    "supg_it",  # SUPG-IT (SupgItCascade, iterative streaming)
    "gamcal",  # GAMCAL (GamCalDynamicCascade)
    "proxy_only",  # Proxy-only baseline
    "oracle_only",  # Oracle-only baseline
]


@dataclass
class CascadeConfig:
    """Configuration for a single cascade evaluation run."""

    algorithm: str = "supg_it"

    sampling_percentage: float = 0.1  # rho: budget fraction
    importance_sampling_weight: float = 0.5  # eta
    recall_target: float = 0.8
    precision_target: float = 0.8
    failure_probability: float = 0.2  # delta

    batch_size: int = 4096  # rowset size (min_rows = max_rows = batch_size)
    dop: int = 1  # degree of parallelism

    supg_it_batch_size: int = 128  # internal sampling batch for SUPG-IT's iterative loop
    delegate_uncertain_to_oracle: bool = True
    override_corrected_recall_threshold: bool = True
    replace_small_ub_with_lb: bool = False
    corrected_delta: float = 0.05

    alpha: float = 0.5
    smooth_lambda: float = 1.0
    n_min: int = 10
    calibration_method: str = "naive_ci"
    opt_method: str = "f1"
    beta: float = 1.0
    gamcal_batch_size: int = 128  # internal sampling batch for GAMCAL's iterative loop
    fixed_quantile: Optional[float] = (
        None  # if set, disables stochastic routing (uses deterministic q)
    )
    quantile_range: tuple[float, float] = (
        0.0,
        1.0,
    )  # range for random q when fixed_quantile is None
    quantile_distribution: str = "uniform"  # "uniform", "t", "normal", or "cauchy"
    quantile_per_batch: bool = False  # if True, one q per batch (coherent exploration)
    quantile_df: float = 3.0  # degrees of freedom for t-distribution
    expanded_sampling_fraction: float = 0.0  # fraction of budget for sampling from all rows
    alpha_decay_gamma: float = 0.0  # power-law decay rate for adaptive alpha (0 = fixed)
    alpha_floor: float = 0.0  # minimum alpha when using adaptive decay
    adaptive_lambda: bool = False  # if True, use GCV gridsearch for GAM lambda

    sampling_strategy: str = "uniform"
    info_lambda: float = 0.0
    info_method: str = "ci_width"
    info_candidate_cap: int = 200
    class_value_weight: float = 1.0

    seed: Optional[int] = None

    cascade_output_column: str = "cascade_output"


@dataclass
class EvalResult:
    """Result of a single evaluation run."""

    dataset: str
    algorithm: str
    seed: Optional[int]
    accuracy: float
    f1: float
    precision: float
    recall: float
    delegation_rate: float
    elapsed_seconds: float
    n_rows: int
    config: dict[str, Any] = field(default_factory=dict)


class CascadeFilterExpr(RowExpr):
    def __init__(self, model_cascade: ModelCascade) -> None:
        self.model_cascade = model_cascade

    def compute(self, rows: pd.DataFrame) -> pd.DataFrame:
        return self.model_cascade.execute(rows)


def _build_cascade(
    config: CascadeConfig,
    oracle_executor: PrecomputedModelExecutor,
    proxy_executor: PrecomputedModelExecutor,
) -> Optional[ModelCascade]:
    """Build a cascade instance from config. Returns None for baselines."""

    if config.algorithm == "supg_rt":
        return SUPGRecallOnlyCascade(
            oracle_executor=oracle_executor,
            proxy_executor=proxy_executor,
            sampling_percentage=config.sampling_percentage,
            importance_sampling_weight=config.importance_sampling_weight,
            recall_target=config.recall_target,
            failure_probability=config.failure_probability,
            cascade_output_column=config.cascade_output_column,
            override_corrected_recall_threshold=config.override_corrected_recall_threshold,
            corrected_delta=config.corrected_delta,
        )

    elif config.algorithm == "supg_joint":
        return SupgSpCascade(
            oracle_executor=oracle_executor,
            proxy_executor=proxy_executor,
            sampling_percentage=config.sampling_percentage,
            importance_sampling_weight=config.importance_sampling_weight,
            recall_target=config.recall_target,
            precision_target=config.precision_target,
            failure_probability=config.failure_probability,
            cascade_output_column=config.cascade_output_column,
            override_corrected_recall_threshold=config.override_corrected_recall_threshold,
            replace_small_ub_with_lb=config.replace_small_ub_with_lb,
            corrected_delta=config.corrected_delta,
        )

    elif config.algorithm == "supg_it":
        return SupgItCascade(
            oracle_executor=oracle_executor,
            proxy_executor=proxy_executor,
            budget_percentage=config.sampling_percentage,
            batch_size=config.supg_it_batch_size,
            importance_sampling_weight=config.importance_sampling_weight,
            recall_target=config.recall_target,
            precision_target=config.precision_target,
            failure_probability=config.failure_probability,
            cascade_output_column=config.cascade_output_column,
            delegate_uncertain_to_oracle=config.delegate_uncertain_to_oracle,
        )

    elif config.algorithm == "gamcal":
        # Lazy import to avoid loading heavy dependencies unless needed
        from icefall.cascades.gam_cal_dynamic_cascade import GamCalDynamicCascade

        return GamCalDynamicCascade(
            oracle_executor=oracle_executor,
            proxy_executor=proxy_executor,
            smooth_lambda=config.smooth_lambda,
            model_method=config.calibration_method,
            opt_method=config.opt_method,
            alpha=config.alpha,
            beta=config.beta,
            batch_size=config.gamcal_batch_size,
            budget_percentage=config.sampling_percentage,
            cascade_output_column=config.cascade_output_column,
            fixed_quantile=config.fixed_quantile,
            quantile_range=config.quantile_range,
            quantile_distribution=config.quantile_distribution,
            quantile_per_batch=config.quantile_per_batch,
            quantile_df=config.quantile_df,
            expanded_sampling_fraction=config.expanded_sampling_fraction,
            alpha_decay_gamma=config.alpha_decay_gamma,
            alpha_floor=config.alpha_floor,
            adaptive_lambda=config.adaptive_lambda,
            sampling_strategy=config.sampling_strategy,
            info_lambda=config.info_lambda,
            info_method=config.info_method,
            info_candidate_cap=config.info_candidate_cap,
            class_value_weight=config.class_value_weight,
        )

    elif config.algorithm in ("proxy_only", "oracle_only"):
        return None

    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}. Available: {ALGORITHMS}")


def run_single(
    dataset_name: str,
    config: Optional[CascadeConfig] = None,
    data: Optional[pd.DataFrame] = None,
    **kwargs,
) -> EvalResult:
    """Run a single evaluation.

    Args:
        dataset_name: Name of the dataset (from datasets.py registry)
        config: CascadeConfig instance (or pass kwargs to create one)
        data: Pre-loaded DataFrame (if None, loads from registry)
        **kwargs: Override config fields

    Returns:
        EvalResult with all metrics
    """
    if config is None:
        config = CascadeConfig(**kwargs)
    else:
        config = copy.deepcopy(config)
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

    if data is None:
        data = load_dataset(dataset_name)

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    data = data.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    n_rows = len(data)

    if config.algorithm == "proxy_only":
        t0 = time.perf_counter()
        predictions = (data[PROXY_SCORE_COL] >= 0.5).astype(bool)
        elapsed = time.perf_counter() - t0
        return _compute_result(
            dataset_name,
            config,
            data,
            predictions,
            delegation_rate=0.0,
            elapsed=elapsed,
            n_rows=n_rows,
        )

    if config.algorithm == "oracle_only":
        t0 = time.perf_counter()
        predictions = data[ORACLE_RESULT_COL].astype(bool)
        elapsed = time.perf_counter() - t0
        return _compute_result(
            dataset_name,
            config,
            data,
            predictions,
            delegation_rate=1.0,
            elapsed=elapsed,
            n_rows=n_rows,
        )

    # Synthetic index column for PrecomputedModelExecutor row lookup
    data = data.reset_index(drop=True)
    data["_idx"] = data.index
    input_columns = ["_idx"]

    oracle_executor = PrecomputedModelExecutor(data, input_columns, [ORACLE_RESULT_COL])
    proxy_executor = PrecomputedModelExecutor(data, input_columns, [PROXY_SCORE_COL])

    cascade = _build_cascade(config, oracle_executor, proxy_executor)

    BatchOperator.min_rows = config.batch_size
    BatchOperator.max_rows = config.batch_size
    BatchOperator.dop = config.dop

    cascade_expr = CascadeFilterExpr(cascade)
    cascade_op = BatchOperator(cascade_expr)

    executor = PipelineEngine()
    table_scan_op = TableScan(data[input_columns])
    result_sink_op = ResultSink()

    executor.add_operator("TableScan#1", table_scan_op)
    executor.add_operator("ModelCascade#1", cascade_op)
    executor.add_operator("ResultWorker#1", result_sink_op)
    executor.add_link("TableScan#1", "ModelCascade#1")
    executor.add_link("ModelCascade#1", "ResultWorker#1")

    t0 = time.perf_counter()
    executor.execute()
    elapsed = time.perf_counter() - t0

    results = result_sink_op.get_result()
    results = pd.merge(data, results, on=input_columns, how="left")

    predictions = results[config.cascade_output_column].astype(bool)
    delegation_rate = oracle_executor.get_rows_retrieved() / n_rows

    return _compute_result(
        dataset_name,
        config,
        data,
        predictions,
        delegation_rate,
        elapsed,
        n_rows,
    )


def _compute_result(
    dataset_name: str,
    config: CascadeConfig,
    data: pd.DataFrame,
    predictions: pd.Series,
    delegation_rate: float,
    elapsed: float,
    n_rows: int,
) -> EvalResult:
    """Compute metrics and build EvalResult."""
    truth = data[ORACLE_RESULT_COL].astype(bool)

    config_dict = {
        "sampling_percentage": config.sampling_percentage,
        "importance_sampling_weight": config.importance_sampling_weight,
        "recall_target": config.recall_target,
        "precision_target": config.precision_target,
        "failure_probability": config.failure_probability,
        "batch_size": config.batch_size,
        "dop": config.dop,
        "alpha": config.alpha,
        "sampling_strategy": config.sampling_strategy,
        "info_lambda": config.info_lambda,
        "info_method": config.info_method,
        "info_candidate_cap": config.info_candidate_cap,
        "class_value_weight": config.class_value_weight,
        "gamcal_batch_size": config.gamcal_batch_size,
        "supg_it_batch_size": config.supg_it_batch_size,
    }

    return EvalResult(
        dataset=dataset_name,
        algorithm=config.algorithm,
        seed=config.seed,
        accuracy=float(accuracy_score(truth, predictions)),
        f1=float(f1_score(truth, predictions, zero_division=0)),
        precision=float(precision_score(truth, predictions, zero_division=0)),
        recall=float(recall_score(truth, predictions, zero_division=0)),
        delegation_rate=delegation_rate,
        elapsed_seconds=elapsed,
        n_rows=n_rows,
        config=config_dict,
    )


def run_sweep(
    dataset_name: str,
    algorithm: str,
    sweep_param: str,
    sweep_values: list[Any],
    n_seeds: int = 10,
    base_config: Optional[CascadeConfig] = None,
    **fixed_kwargs,
) -> pd.DataFrame:
    """Run a parameter sweep with multiple seeds.

    Args:
        dataset_name: Dataset name
        algorithm: Algorithm name
        sweep_param: Config parameter to sweep
        sweep_values: Values to sweep over
        n_seeds: Number of random seeds per configuration
        base_config: Base configuration (optional)
        **fixed_kwargs: Additional fixed config overrides

    Returns:
        DataFrame with one row per (sweep_value, seed) combination
    """
    if base_config is None:
        base_config = CascadeConfig(algorithm=algorithm)
    else:
        base_config = copy.deepcopy(base_config)
        base_config.algorithm = algorithm

    for k, v in fixed_kwargs.items():
        if hasattr(base_config, k):
            setattr(base_config, k, v)

    data = load_dataset(dataset_name)

    results = []
    total = len(sweep_values) * n_seeds
    done = 0

    for val in sweep_values:
        for seed in range(n_seeds):
            cfg = copy.deepcopy(base_config)
            setattr(cfg, sweep_param, val)
            cfg.seed = seed

            result = run_single(dataset_name, config=cfg, data=data)
            result_dict = _result_to_dict(result)
            result_dict[sweep_param] = val
            results.append(result_dict)

            done += 1
            if done % 10 == 0 or done == total:
                print(
                    f"  [{done}/{total}] {algorithm} {sweep_param}={val} seed={seed} "
                    f"F1={result.f1:.3f} deleg={result.delegation_rate:.3f}"
                )

    return pd.DataFrame(results)


def run_algorithms_sweep(
    dataset_name: str,
    algorithms_config: dict[str, dict[str, Any]],
    n_seeds: int = 10,
    base_config: Optional[CascadeConfig] = None,
) -> pd.DataFrame:
    """Run sweeps for multiple algorithms on a single dataset.

    Args:
        dataset_name: Dataset name
        algorithms_config: Dict mapping algorithm name to
            {"sweep_param": str, "sweep_values": list, **extra_kwargs}
        n_seeds: Number of seeds per configuration
        base_config: Base configuration

    Returns:
        Combined DataFrame with results from all algorithms
    """
    data = load_dataset(dataset_name)
    all_results = []

    for algo_name, algo_cfg in algorithms_config.items():
        sweep_param = algo_cfg["sweep_param"]
        sweep_values = algo_cfg["sweep_values"]
        extra = {k: v for k, v in algo_cfg.items() if k not in ("sweep_param", "sweep_values")}

        if base_config is None:
            cfg = CascadeConfig(algorithm=algo_name, **extra)
        else:
            cfg = copy.deepcopy(base_config)
            cfg.algorithm = algo_name
            for k, v in extra.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        total = len(sweep_values) * n_seeds
        done = 0

        for val in sweep_values:
            for seed in range(n_seeds):
                run_cfg = copy.deepcopy(cfg)
                setattr(run_cfg, sweep_param, val)
                run_cfg.seed = seed

                result = run_single(dataset_name, config=run_cfg, data=data)
                result_dict = _result_to_dict(result)
                result_dict["sweep_param"] = sweep_param
                result_dict["sweep_value"] = val
                all_results.append(result_dict)

                done += 1
                if done % 10 == 0 or done == total:
                    print(
                        f"  [{done}/{total}] {algo_name} {sweep_param}={val} seed={seed} "
                        f"F1={result.f1:.3f} deleg={result.delegation_rate:.3f}"
                    )

    for baseline in ["proxy_only", "oracle_only"]:
        for seed in range(n_seeds):
            result = run_single(dataset_name, data=data, algorithm=baseline, seed=seed)
            result_dict = _result_to_dict(result)
            result_dict["sweep_param"] = "none"
            result_dict["sweep_value"] = None
            all_results.append(result_dict)

    return pd.DataFrame(all_results)


def _result_to_dict(result: EvalResult) -> dict[str, Any]:
    """Flatten an EvalResult into a dict for DataFrame construction."""
    d = {
        "dataset": result.dataset,
        "algorithm": result.algorithm,
        "seed": result.seed,
        "accuracy": result.accuracy,
        "f1": result.f1,
        "precision": result.precision,
        "recall": result.recall,
        "delegation_rate": result.delegation_rate,
        "elapsed_seconds": result.elapsed_seconds,
        "n_rows": result.n_rows,
    }
    for k, v in result.config.items():
        d[f"cfg_{k}"] = v
    return d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cascade evaluation")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--recall-target", type=float, default=0.8)
    parser.add_argument("--precision-target", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.5, help="GAMCAL alpha")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Rowset size for BatchOperator (min_rows = max_rows)",
    )
    parser.add_argument(
        "--gamcal-batch-size",
        type=int,
        default=128,
        help="GAMCAL internal sampling batch (oracle samples per loop iteration)",
    )
    parser.add_argument(
        "--supg-it-batch-size",
        type=int,
        default=128,
        help="SUPG-IT internal sampling batch (oracle samples per loop iteration)",
    )
    parser.add_argument("--dop", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, help="Output CSV path")
    args = parser.parse_args()

    config = CascadeConfig(
        algorithm=args.algorithm,
        recall_target=args.recall_target,
        precision_target=args.precision_target,
        alpha=args.alpha,
        batch_size=args.batch_size,
        gamcal_batch_size=args.gamcal_batch_size,
        supg_it_batch_size=args.supg_it_batch_size,
        dop=args.dop,
        seed=args.seed,
    )

    result = run_single(args.dataset, config=config)
    print(f"\nDataset: {result.dataset}")
    print(f"Algorithm: {result.algorithm}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"F1: {result.f1:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall: {result.recall:.4f}")
    print(f"Delegation rate: {result.delegation_rate:.4f}")
    print(f"Time: {result.elapsed_seconds:.3f}s")

    if args.output:
        df = pd.DataFrame([_result_to_dict(result)])
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
