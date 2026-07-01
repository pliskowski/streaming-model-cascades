from abc import ABC, abstractmethod

import pandas as pd


class ModelCascade(ABC):
    @abstractmethod
    def execute(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the model cascade on the given rows.

        Args:
            rows (pd.DataFrame): The input rows to process.

        Returns:
            pd.DataFrame: The processed rows after executing the model cascade.
        """
        pass

    def compute_low_confidence_selections(
        self,
        current_selections: list[int],
        proxy_scores: pd.Series,
        thresholds: tuple[float, float],
    ) -> list[int]:
        """
        Compute the low confidence selections based on the current selections and proxy scores.

        Args:
            current_selections (list[int]): The current selections to process.
            proxy_scores (pd.Series): The proxy scores to use for processing.
            thresholds (tuple[float, float]): The low and high thresholds.

        Returns:
            list[int]: A list of indices representing the low confidence selections.
        """
        low_threshold, high_threshold = thresholds
        return [
            i
            for i in current_selections
            if low_threshold <= proxy_scores[i] and proxy_scores[i] < high_threshold
        ]

    def sample_more_rows_for_oracle(
        self,
        rows: pd.DataFrame,
        proxy_scores: pd.Series,
        budget: int,
        low_confidence_selections: list[int],
    ) -> tuple[list[int], pd.Series]:
        """
        Sample more rows for the oracle based on the low confidence selections.
        Args:
            rows (pd.DataFrame): The rows to sample from.
            proxy_scores (pd.Series): The proxy scores to use for sampling.
            budget (int): The budget for sampling.
            low_confidence_selections (list[int]): The low confidence selections to sample from.

        Returns:
            tuple[list[int], pd.Series]: A tuple containing the sampled indices and the corresponding proxy scores.
        """

        sampling_result = self.sampling_method.compute_sample(
            low_confidence_selections, proxy_scores, budget
        )
        sampling_selections = sampling_result.sample_selections

        oracle_results = self.oracle_executor.execute(rows, sampling_selections)
        oracle_results = oracle_results[self.oracle_executor.output_columns[0]]

        self.sampling_state.compute_thresholds_with_new_samples(
            oracle_results, proxy_scores, sampling_result
        )

        return sampling_selections, oracle_results

    def populate_sample_results_from_oracle(
        self,
        results: pd.Series,
        oracle_results: pd.Series,
        sampling_selections: list[int],
    ) -> None:
        """
        Populate the sample results from the oracle.

        Args:
            results (pd.Series): The results to populate.
            oracle_results (pd.Series): The oracle results to use for population.
            sampling_selections (list[int]): The sampling selections to use for population.
        """
        results.iloc[sampling_selections] = oracle_results.iloc[sampling_selections].astype(bool)

    def subtract_sample_from_selection(
        self,
        current_selections: list[int],
        sampling_selections: list[int],
    ):
        """
        Subtract the sampled selections from the current selections.

        Args:
            current_selections (list[int]): The current selections to process.
            sampling_selections (list[int]): The sampled selections to subtract.

        Returns:
            list[int]: A list of indices representing the remaining selections.
        """
        return list(set(current_selections) - set(sampling_selections))

    def populate_high_confidence_results_from_proxy(
        self,
        results: pd.Series,
        proxy_results: pd.Series,
        thresholds: tuple[float, float],
        current_selections: list[int],
        low_confidence_selections: list[int],
    ) -> None:
        high_confidence_selections = self.subtract_sample_from_selection(
            current_selections, low_confidence_selections
        )

        results.iloc[high_confidence_selections] = (
            proxy_results.iloc[high_confidence_selections]
            >= thresholds[
                1
            ]  # Since rows are high confidence if they are lower than the high threshold, then it is lower than the low threshold
        )

    def populate_low_confidence_results_from_proxy(
        self,
        results: pd.Series,
        proxy_scores: pd.Series,
        threshold: float,
        low_confidence_selections: list[int],
    ) -> None:
        results.iloc[low_confidence_selections] = (
            proxy_scores.iloc[low_confidence_selections] > threshold
        )
