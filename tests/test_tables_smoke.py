"""Smoke tests for publication table generators."""

from pathlib import Path

import pandas as pd

from icefall.figures.tables import generate_all_publication_tables


def _minimal_results_csv(path: Path) -> None:
    rows = []
    for dataset in ("mmlu", "boolq"):
        for algorithm in ("proxy_only", "supg_rt", "supg_joint", "supg_it", "gamcal"):
            for sweep_value in (0.6, 0.8):
                for seed in range(2):
                    rows.append(
                        {
                            "dataset": dataset,
                            "algorithm": algorithm,
                            "sweep_value": sweep_value,
                            "seed": seed,
                            "f1": 0.7 + 0.1 * sweep_value,
                            "precision": 0.75,
                            "recall": 0.72,
                            "delegation_rate": 0.15 + 0.2 * sweep_value,
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_all_publication_tables(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    out_dir = tmp_path / "tables"
    _minimal_results_csv(csv_path)

    generate_all_publication_tables(csv_path, out_dir)

    expected = [
        "best_f1_per_algorithm.tex",
        "f1_at_fixed_budget.tex",
        "min_delegation_for_target.tex",
    ]
    for name in expected:
        assert (out_dir / name).exists()
        assert (out_dir / name).read_text().strip()
