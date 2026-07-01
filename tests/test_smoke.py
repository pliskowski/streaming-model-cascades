"""Smoke tests for the cascade paper evaluation harness.

Runs each algorithm on a tiny synthetic dataset to verify end-to-end
execution without crashes and that outputs have expected structure.
"""

import numpy as np
import pandas as pd
import pytest

from icefall.datasets import ORACLE_RESULT_COL, PROXY_SCORE_COL
from icefall.harness import ALGORITHMS, CascadeConfig, run_single


def _make_synthetic_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            PROXY_SCORE_COL: rng.uniform(0, 1, n),
            ORACLE_RESULT_COL: rng.choice([True, False], n),
        }
    )


SYNTHETIC_DATA = _make_synthetic_data()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithm_runs(algorithm):
    config = CascadeConfig(
        algorithm=algorithm,
        batch_size=200,
        dop=1,
        seed=0,
    )
    result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)

    assert result.dataset == "synthetic"
    assert result.algorithm == algorithm
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.delegation_rate <= 1.0
    assert result.n_rows == len(SYNTHETIC_DATA)


def test_gamcal_fixed_quantile():
    config = CascadeConfig(
        algorithm="gamcal",
        batch_size=200,
        dop=1,
        seed=0,
        fixed_quantile=0.5,
    )
    result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
    assert 0.0 <= result.f1 <= 1.0


def test_gamcal_t_distribution():
    config = CascadeConfig(
        algorithm="gamcal",
        batch_size=200,
        dop=1,
        seed=0,
        quantile_distribution="t",
        quantile_df=3.0,
    )
    result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
    assert 0.0 <= result.f1 <= 1.0


def test_gamcal_expanded_sampling():
    config = CascadeConfig(
        algorithm="gamcal",
        batch_size=200,
        dop=1,
        seed=0,
        expanded_sampling_fraction=0.1,
    )
    result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
    assert 0.0 <= result.f1 <= 1.0
