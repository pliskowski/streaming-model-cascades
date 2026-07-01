"""SUPG-SP cascade: single-pass joint precision-recall targeting (LOTUS baseline).

Faithful streaming reimplementation of the LOTUS semantic-filter cascade
(Patel et al., 2025): one oracle sample batch per partition, then threshold
estimation with uncertain-region delegation.
"""

import sys
from math import floor

import pandas as pd

from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
from icefall.cascades.model_cascade import ModelCascade
from icefall.cascades.sampling.iterative_sampling_method import IterativeSamplingMethod
from icefall.cascades.sampling.iterative_sampling_state import IterativeSamplingState


class SupgSpCascade(ModelCascade):
    """Single-pass SUPG cascade with joint precision and recall targets."""

    def __init__(
        self,
        oracle_executor: PrecomputedModelExecutor,
        proxy_executor: PrecomputedModelExecutor,
        sampling_percentage: float = 0.1,
        importance_sampling_weight: float = 0.5,
        recall_target: float = 0.8,
        precision_target: float = 0.8,
        failure_probability: float = 0.05,
        cascade_output_column: str = "cascade_output",
        override_corrected_recall_threshold: bool = False,
        replace_small_ub_with_lb: bool = True,
        corrected_delta: float = 0.05,
    ) -> None:
        self.oracle_executor = oracle_executor
        self.proxy_executor = proxy_executor
        self.sampling_percentage = sampling_percentage
        self.importance_sampling_weight = importance_sampling_weight
        self.sampling_state = IterativeSamplingState(
            recall_target,
            precision_target,
            failure_probability,
            override_corrected_recall_threshold=override_corrected_recall_threshold,
            replace_small_ub_with_lb=replace_small_ub_with_lb,
            corrected_delta=corrected_delta,
        )
        # Rowset-sized sampling: budget is derived from len(rows) in execute().
        self.sampling_method = IterativeSamplingMethod(sys.maxsize, self.importance_sampling_weight)

        self.cascade_output_column = cascade_output_column

    def execute(self, rows: pd.DataFrame) -> pd.DataFrame:

        input_selections = list(range(len(rows)))

        proxy_result = self.proxy_executor.execute(rows, input_selections)
        proxy_scores = proxy_result[self.proxy_executor.output_columns[0]]

        budget = floor(self.sampling_percentage * len(rows))

        results = pd.Series([None] * len(rows), dtype=object)

        if budget > 0:
            sampling_selections, oracle_results = self.sample_more_rows_for_oracle(
                rows, proxy_scores, budget, input_selections
            )
            budget -= len(sampling_selections)
            input_selections = self.subtract_sample_from_selection(
                input_selections, sampling_selections
            )
            self.populate_sample_results_from_oracle(results, oracle_results, sampling_selections)

        low_confidence_selections = self.compute_low_confidence_selections(
            input_selections,
            proxy_scores,
            self.sampling_state.get_high_confidence_thresholds(),
        )

        self.populate_high_confidence_results_from_proxy(
            results,
            proxy_scores,
            self.sampling_state.get_high_confidence_thresholds(),
            input_selections,
            low_confidence_selections,
        )

        if len(low_confidence_selections) > 0:
            oracle_results = self.oracle_executor.execute(rows, low_confidence_selections)
            oracle_results = oracle_results[self.oracle_executor.output_columns[0]]
            self.populate_sample_results_from_oracle(
                results, oracle_results, low_confidence_selections
            )

        return pd.concat([rows, results.rename(self.cascade_output_column).astype(bool)], axis=1)
