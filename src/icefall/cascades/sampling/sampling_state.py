import abc
from typing import final

import pandas as pd

from icefall.cascades.sampling.sampling_method import SamplingResult

INVALID_THRESHOLD = -1.0


class SamplingState(abc.ABC):
    """
    A class to represent the state of sampling.
    """

    def __init__(self) -> None:
        self.low_threshold = INVALID_THRESHOLD
        self.high_threshold = INVALID_THRESHOLD

    @final
    def get_high_confidence_thresholds(self) -> tuple[float, float]:
        """
        Get the low and high thresholds.

        Returns:
            tuple: A tuple containing the low and high thresholds.
        """
        return self.low_threshold, self.high_threshold

    @final
    def has_consumed_samples(self) -> bool:
        """
        Check if the sampling state has consumed samples.

        Returns:
            bool: True if samples have been consumed, False otherwise.
        """
        return self.low_threshold != INVALID_THRESHOLD and self.high_threshold != INVALID_THRESHOLD

    @final
    def compute_low_confidence_threshold(self) -> float:
        """
        Get the low confidence threshold.

        Returns:
            float: The low confidence threshold.
        """
        assert self.has_consumed_samples(), "Sampling state has not consumed samples."
        low_confidence_threshold = self.compute_low_confidence_threshold_impl()
        if (
            (low_confidence_threshold < self.low_threshold)
            or (low_confidence_threshold > self.high_threshold)
            or (low_confidence_threshold == INVALID_THRESHOLD)
        ):
            return (self.low_threshold + self.high_threshold) / 2
        return low_confidence_threshold

    @abc.abstractmethod
    def compute_thresholds_with_new_samples(
        self,
        oracle_results: pd.Series,
        proxy_score: pd.Series,
        sampling_result: SamplingResult,
    ) -> None:
        """
        Compute the low and high thresholds based on new samples.

        Args:
            oracle_results (pd.Series): The oracle results.
            proxy_score (pd.Series): The proxy score.
            sampling_result (SamplingResult): The sampling result.

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def compute_low_confidence_threshold_impl(self) -> float:
        """
        Compute the low confidence threshold.

        Returns:
            float: The low confidence threshold.
        """
        pass
