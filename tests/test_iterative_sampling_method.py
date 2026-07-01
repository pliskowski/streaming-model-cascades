import unittest

import numpy as np
import pandas as pd

from icefall.cascades.sampling.iterative_sampling_method import (
    IterativeSamplingMethod,
)


class TestIterativeSamplingMethod(unittest.TestCase):
    def setUp(self):
        """Setup test data before each test method is run."""
        # Setup test data
        self.batch_size = 30
        self.importance_sampling_weight = 0.5
        self.sampling_method = IterativeSamplingMethod(
            self.batch_size, self.importance_sampling_weight
        )

        # Create input selection and proxy scores
        self.input_selections = list(range(100))

        # Create proxy scores where the first half has lower scores
        self.proxy_scores = pd.Series(np.ones(100))
        for i in range(50):
            self.proxy_scores[i] = 0.5  # Lower scores for first half
        for i in range(50, 100):
            self.proxy_scores[i] = 2.0  # Higher scores for second half

    def test_batch_size_limits(self):
        """Test batch size limits against budget and input size."""
        # Test with different budgets
        budget1 = 10
        indices1, factors1 = self.sampling_method.compute_sample(
            self.input_selections, self.proxy_scores, budget1
        )
        self.assertEqual(len(indices1), 10)

        # Test with larger budget
        budget2 = 50
        indices2, factors2 = self.sampling_method.compute_sample(
            self.input_selections, self.proxy_scores, budget2
        )
        self.assertEqual(len(indices2), 30)  # Limited by batch_size

        # Test with budget > input size
        self.sampling_method.batch_size = 150
        budget3 = 150
        indices3, factors3 = self.sampling_method.compute_sample(
            self.input_selections, self.proxy_scores, budget3
        )
        self.assertEqual(len(indices3), 100)  # Capped at input size

    def test_correction_factors(self):
        """Test correction factors computation."""
        # Sample with a budget
        budget = 30
        indices, factors = self.sampling_method.compute_sample(
            self.input_selections, self.proxy_scores, budget
        )

        # Verify correction factors are positive
        for factor in factors:
            self.assertGreater(factor, 0.0)

        # Verify indices are within bounds
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 100)


if __name__ == "__main__":
    unittest.main()
