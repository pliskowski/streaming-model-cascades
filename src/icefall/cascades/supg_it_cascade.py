"""SUPG-IT cascade: iterative streaming refinement with joint precision-recall targets."""

import pandas as pd

from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
from icefall.cascades.model_cascade import ModelCascade
from icefall.cascades.sampling.iterative_sampling_method import IterativeSamplingMethod
from icefall.cascades.sampling.iterative_sampling_state import IterativeSamplingState


class SupgItCascade(ModelCascade):
    """Iterative SUPG cascade with per-batch threshold refinement (SUPG-IT)."""

    def __init__(
        self,
        oracle_executor: PrecomputedModelExecutor,
        proxy_executor: PrecomputedModelExecutor,
        budget_percentage: float = 0.1,
        batch_size: int = 1,
        importance_sampling_weight: float = 0.5,
        recall_target: float = 0.8,
        precision_target: float = 0.8,
        failure_probability: float = 0.05,
        cascade_output_column: str = "cascade_output",
        delegate_uncertain_to_oracle: bool = True,
    ) -> None:
        """
        Args:
            oracle_executor: Precomputed oracle labels/scores.
            proxy_executor: Precomputed proxy scores.
            budget_percentage: Fraction of rows budgeted for oracle sampling per batch.
            batch_size: Oracle samples drawn per refinement iteration.
            importance_sampling_weight: Mixing weight for uniform vs. importance sampling.
            recall_target: Target recall (t_R).
            precision_target: Target precision (t_P).
            failure_probability: Allowed failure probability (delta).
            cascade_output_column: Output column name for cascade decisions.
            delegate_uncertain_to_oracle: If True, delegate the uncertain region to the oracle;
                if False, classify uncertain rows with a proxy threshold (ablation).
        """
        self.oracle_executor = oracle_executor
        self.proxy_executor = proxy_executor

        self.budget_percentage = budget_percentage

        self.sampling_method = IterativeSamplingMethod(batch_size, importance_sampling_weight)
        if delegate_uncertain_to_oracle:
            self.sampling_state = IterativeSamplingState(
                recall_target,
                precision_target,
                failure_probability,
                fix_recall_target_correction=True,
                override_corrected_recall_threshold=True,
                replace_small_ub_with_lb=False,
                corrected_delta=0.05,
            )
        else:
            self.sampling_state = IterativeSamplingState(
                recall_target, precision_target, failure_probability
            )

        self.cascade_output_column = cascade_output_column

        self.delegate_uncertain_to_oracle = delegate_uncertain_to_oracle

    def execute(self, rows: pd.DataFrame) -> pd.DataFrame:
        input_selections = list(range(len(rows)))
        proxy_result = self.proxy_executor.execute(rows, input_selections)
        proxy_scores = proxy_result[self.proxy_executor.output_columns[0]]

        budget = max(1, int(len(rows) * self.budget_percentage))

        results = pd.Series([None] * len(rows), dtype=object)

        if not self.sampling_state.has_consumed_samples():
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

        while budget > 0 and len(low_confidence_selections) > 0:
            if self.delegate_uncertain_to_oracle:
                sampling_selections, oracle_results = self.sample_more_rows_for_oracle(
                    rows, proxy_scores, budget, input_selections
                )
            else:
                sampling_selections, oracle_results = self.sample_more_rows_for_oracle(
                    rows, proxy_scores, budget, low_confidence_selections
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
            if self.delegate_uncertain_to_oracle:
                self.populate_low_confidence_results_from_oracle(
                    results, rows, low_confidence_selections
                )
            else:
                self.populate_low_confidence_results_from_proxy(
                    results,
                    proxy_scores,
                    self.sampling_state.compute_low_confidence_threshold(),
                    low_confidence_selections,
                )

        return pd.concat([rows, results.rename(self.cascade_output_column).astype(bool)], axis=1)

    def compute_low_confidence_selections(
        self,
        current_selections: list[int],
        proxy_scores: pd.Series,
        thresholds: tuple[float, float],
    ) -> list[int]:
        """Return row indices whose proxy scores fall in the uncertain region."""
        low_threshold, high_threshold = thresholds
        if self.delegate_uncertain_to_oracle:
            return [
                i
                for i in current_selections
                if proxy_scores[i] > low_threshold and proxy_scores[i] < high_threshold
            ]
        return [
            i
            for i in current_selections
            if proxy_scores[i] >= low_threshold and proxy_scores[i] <= high_threshold
        ]

    def populate_low_confidence_results_from_oracle(
        self,
        results: pd.Series,
        rows: pd.DataFrame,
        low_confidence_selections: list[int],
    ) -> None:
        """Fill uncertain-region rows with oracle labels."""
        oracle_results = self.oracle_executor.execute(rows, low_confidence_selections)
        oracle_results = oracle_results[self.oracle_executor.output_columns[0]]
        results.iloc[low_confidence_selections] = oracle_results.iloc[
            low_confidence_selections
        ].astype(bool)
