"""SUPG Recall-Only Cascade (SUPG-RT baseline).

Faithful implementation of SUPG Algorithm 5 (Kang et al., 2020) for
recall-targeted (RT) queries.  Uses a single threshold τ derived from
importance-weighted oracle samples so that

    Recall(τ) ≥ recall_target  with probability ≥ 1 − δ.

Key differences from SUPG-SP (joint-target cascade):
  - Single threshold: accept if proxy score ≥ τ, reject otherwise.
  - No uncertain region, no extra oracle delegation for low-confidence
    records.  Records below τ are simply rejected (classified False).
  - No precision constraint.
"""

import sys
from math import floor

import pandas as pd

from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
from icefall.cascades.model_cascade import ModelCascade
from icefall.cascades.sampling.iterative_sampling_method import IterativeSamplingMethod
from icefall.cascades.sampling.iterative_sampling_state import IterativeSamplingState


class SUPGRecallOnlyCascade(ModelCascade):
    """Single-threshold, recall-only, single-pass cascade (SUPG-RT)."""

    def __init__(
        self,
        oracle_executor: PrecomputedModelExecutor,
        proxy_executor: PrecomputedModelExecutor,
        sampling_percentage: float = 0.1,
        importance_sampling_weight: float = 0.5,
        recall_target: float = 0.8,
        failure_probability: float = 0.05,
        cascade_output_column: str = "cascade_output",
        override_corrected_recall_threshold: bool = True,
        corrected_delta: float = 0.05,
    ) -> None:
        self.oracle_executor = oracle_executor
        self.proxy_executor = proxy_executor
        self.sampling_percentage = sampling_percentage
        self.importance_sampling_weight = importance_sampling_weight
        self.cascade_output_column = cascade_output_column

        # Reuse IterativeSamplingState for threshold computation.
        # Set precision_target=0.0 so that the computed tau_pos is
        # effectively irrelevant – we only use the recall threshold.
        self.sampling_state = IterativeSamplingState(
            recall_target=recall_target,
            precision_target=0.0,  # not used; we extract only the recall threshold
            failure_probability=failure_probability,
            override_corrected_recall_threshold=override_corrected_recall_threshold,
            replace_small_ub_with_lb=True,
            corrected_delta=corrected_delta,
        )
        self.sampling_method = IterativeSamplingMethod(sys.maxsize, self.importance_sampling_weight)

    def execute(self, rows: pd.DataFrame) -> pd.DataFrame:
        input_selections = list(range(len(rows)))

        # Get proxy scores
        proxy_result = self.proxy_executor.execute(rows, input_selections)
        proxy_scores = proxy_result[self.proxy_executor.output_columns[0]]

        # Sample a percentage of the rowset for threshold estimation
        budget = floor(self.sampling_percentage * len(rows))
        results = pd.Series([None] * len(rows), dtype=object)

        if budget > 0:
            # Importance-weighted oracle sampling + threshold computation
            sampling_selections, oracle_results = self.sample_more_rows_for_oracle(
                rows, proxy_scores, budget, input_selections
            )
            input_selections = self.subtract_sample_from_selection(
                input_selections, sampling_selections
            )
            # Sampled records get their oracle labels directly
            self.populate_sample_results_from_oracle(results, oracle_results, sampling_selections)

        # Extract the recall threshold (tau_neg_prime).
        # IterativeSamplingState stores it as low_threshold after
        # compute_thresholds_with_new_samples is called.
        tau = self.sampling_state.low_threshold

        # Single-threshold classification for remaining records:
        # Accept (True) if proxy score ≥ τ, reject (False) otherwise.
        for idx in input_selections:
            results.iloc[idx] = proxy_scores.iloc[idx] >= tau

        return pd.concat([rows, results.rename(self.cascade_output_column).astype(bool)], axis=1)
