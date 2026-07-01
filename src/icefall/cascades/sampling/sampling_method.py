import abc
from collections import namedtuple
from typing import final

import pandas as pd

SamplingResult = namedtuple("SamplingResult", ["sample_selections", "correction_factors"])


class SamplingMethod(abc.ABC):
    """
    Abstract base class for sampling methods.
    """

    def __init__(self, batch_size: int):
        """
        Initialize the sampling method with a batch size.

        Args:
            batch_size: The size of the batch to sample.
        """
        self.batch_size = batch_size

    @final
    def compute_sample(
        self, input_selections: list[int], proxy_score: pd.Series, budget: int
    ) -> SamplingResult:
        """
        Compute the sample based on the input selections and proxy score.
        Args:
            input_selections: The input selections to sample from.
            proxy_score: The proxy score to use for sampling.
            budget: The budget for sampling.

        Returns:
            A SamplingResult containing the sampled indices and correction factors.
        """

        n_samples = min(len(input_selections), self.batch_size, budget)
        sample_selections, correction_factors = self.sample_impl(
            input_selections, proxy_score, n_samples
        )
        return SamplingResult(sample_selections, correction_factors)

    @abc.abstractmethod
    def sample_impl(
        self, input_selections: list[int], proxy_score: pd.Series, budget: int
    ) -> tuple[list[int], list[float]]:
        """
        Abstract method to implement the sampling logic.

        Args:
            input_selections: The input selections to sample from.
            proxy_score: The proxy score to use for sampling.
            budget: The budget for sampling.

        Returns:
            A tuple containing the sampled indices and correction factors.
        """
        pass
