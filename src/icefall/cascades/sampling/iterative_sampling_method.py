import numpy as np
import pandas as pd

from icefall.cascades.sampling.sampling_method import SamplingMethod


class IterativeSamplingMethod(SamplingMethod):
    def __init__(self, batch_size: int, importance_sampling_weight: float = 0.5):
        """
        Initialize the IterativeSamplingMethod with a batch size and uniform weight.

        Args:
            batch_size: The size of the batch to sample.
            importance_sampling_weight: The weight for uniform sampling.
        """
        super().__init__(batch_size)
        self.importance_sampling_weight = importance_sampling_weight

    def sample_impl(
        self, input_selections: list[int], proxy_score: pd.Series, sample_size: int
    ) -> tuple[list[int], list[float]]:
        """
        Implement the sampling logic for iterative sampling.

        Args:
            input_selections: The input selections to sample from.
            proxy_score: The proxy score to use for sampling.
            sample_size: The size of the sample to draw.

        Returns:
            A SamplingResult containing the sampled indices and correction factors.
        """
        scores = proxy_score[input_selections]
        scores = scores.astype(float)
        n_scores = len(scores)
        w = np.sqrt(scores)
        # Defensive mixing of uniform weights.
        w = (
            self.importance_sampling_weight * w / np.sum(w)
            + (1 - self.importance_sampling_weight) * np.ones(n_scores) / n_scores
        )
        w /= sum(w)

        size = min(sample_size, len(scores))
        sample_index = np.random.choice(input_selections, size=size, p=w, replace=False)
        correction_factors = (1 / n_scores) / w[sample_index]

        return list(sample_index), list(correction_factors)
