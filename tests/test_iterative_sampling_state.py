import unittest

import pandas as pd

from icefall.cascades.sampling.iterative_sampling_state import IterativeSamplingState
from icefall.cascades.sampling.sampling_method import SamplingResult
from icefall.cascades.sampling.sampling_state import INVALID_THRESHOLD

# Default max samples for testing
DEFAULT_MAX_SAMPLES = 1048576


class TestIterativeSamplingState(unittest.TestCase):
    """
    Unit tests for the IterativeSamplingState class.
    These tests are based on the test cases in the provided C++ code.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        # Create sampling state with default parameters
        self.sampling_state = IterativeSamplingState(
            recall_target=0.8,
            precision_target=0.8,
            failure_probability=0.2,
            max_samples=1048576,  # Using a large default value
        )

        # Create sample selection of 50 items
        self.sample_size = 50
        self.sample_selections = list(range(self.sample_size))

        # Create correction factors (all 1.0 for simplicity)
        self.correction_factors = [1.0] * self.sample_size

        # Initialize oracle results and confidence scores (will be populated in tests)
        self.oracle_results = pd.Series([0] * self.sample_size)
        self.confidence_scores = pd.Series([0.0] * self.sample_size)

        # Create a sampling result
        self.sampling_result = SamplingResult(
            sample_selections=self.sample_selections,
            correction_factors=self.correction_factors,
        )

    def setup_sample_pattern(self, split_point: float = 0.5, mixed_region_width: float = 0.05):
        """
        Set up a simple pattern for testing:
        - High scores (above splitPoint + mixedRegionWidth) are always positive
        - Low scores (below splitPoint - mixedRegionWidth) are always negative
        - Scores within splitPoint ±mixedRegionWidth are mixed for more realistic testing
        """
        for row_idx in range(self.sample_size):
            score = (row_idx + 1) / self.sample_size  # 0.02 to 1
            self.confidence_scores[row_idx] = score

            if score > split_point + mixed_region_width:
                self.oracle_results[row_idx] = 1  # Always positive
            elif score < split_point - mixed_region_width:
                self.oracle_results[row_idx] = 0  # Always negative
            else:
                # Mixed area near the split point
                self.oracle_results[row_idx] = 1 if row_idx % 2 == 0 else 0

    def test_compute_threshold(self):
        """
        Test computation of thresholds with a pattern where split is at 0.7 with a narrow mixed region.
        """
        # Set up a pattern where split is at 0.7 with a narrow mixed region
        self.setup_sample_pattern(0.7, 0.03)

        # Compute thresholds
        self.sampling_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=self.sampling_result,
        )

        # Get the computed thresholds
        low_threshold, high_threshold = self.sampling_state.get_high_confidence_thresholds()

        # In the C++ test, it expects:
        # EXPECT_EQ(thresholds.second, 0.74);
        # EXPECT_EQ(thresholds.first, 0.68);
        # The Python implementation may have slight differences due to the statistical corrections
        self.assertGreater(high_threshold, low_threshold)
        self.assertAlmostEqual(high_threshold, 0.74, delta=1e-3)
        self.assertAlmostEqual(low_threshold, 0.68, delta=1e-3)

    def test_compute_threshold_low_threshold(self):
        """
        Test computation of thresholds with a pattern where split is at 0.4 with a wider mixed region.
        """
        # Set up a pattern where split is at 0.4 with a wider mixed region
        self.setup_sample_pattern(0.4, 0.1)

        # Compute thresholds
        self.sampling_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=self.sampling_result,
        )

        # Get the computed thresholds
        low_threshold, high_threshold = self.sampling_state.get_high_confidence_thresholds()

        # In the C++ test, it expects:
        # EXPECT_EQ(thresholds.second, 0.46);
        # EXPECT_EQ(thresholds.first, 0.4);
        self.assertGreater(high_threshold, low_threshold)
        self.assertAlmostEqual(high_threshold, 0.46, delta=1e-3)
        self.assertAlmostEqual(low_threshold, 0.4, delta=1e-3)

    def test_compute_threshold_for_low_confidence_rows(self):
        """
        Test computation of threshold for low confidence rows.
        """
        # Set up a pattern where high scores are positives with default mixed region
        self.setup_sample_pattern(0.5)  # Using default mixedRegionWidth of 0.05

        # First compute thresholds with samples
        self.sampling_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=self.sampling_result,
        )

        # Then compute threshold for low confidence rows
        low_conf_threshold = self.sampling_state.compute_low_confidence_threshold()

        self.assertAlmostEqual(low_conf_threshold, 0.45, delta=1e-3)

    def test_compute_threshold_with_no_samples(self):
        """
        Test computation of thresholds with no samples.
        """
        # Create empty sample selection
        empty_selections = []
        empty_correction_factors = []
        empty_sampling_result = SamplingResult(
            sample_selections=empty_selections,
            correction_factors=empty_correction_factors,
        )

        # Compute thresholds with empty sample
        self.sampling_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=empty_sampling_result,
        )

        # Get the computed thresholds
        low_threshold, high_threshold = self.sampling_state.get_high_confidence_thresholds()

        # With no samples, thresholds should default to safe values
        # Default low and high thresholds might be different from C++ implementation
        self.assertEqual(low_threshold, INVALID_THRESHOLD)
        self.assertEqual(high_threshold, INVALID_THRESHOLD)
        self.assertFalse(self.sampling_state.has_consumed_samples())

    def test_thresholds_with_multiple_batches(self):
        """
        Test computation of thresholds with multiple batches of samples.
        """
        # First batch of samples with split at 0.6 and small mixed region
        self.setup_sample_pattern(0.6, 0.02)
        self.sampling_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=self.sampling_result,
        )

        thresholds1_low, thresholds1_high = self.sampling_state.get_high_confidence_thresholds()

        # Create a new batch with different split point and larger mixed region
        self.setup_sample_pattern(0.4, 0.08)
        self.sampling_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=self.sampling_result,
        )

        thresholds2_low, thresholds2_high = self.sampling_state.get_high_confidence_thresholds()

        # Thresholds should adapt based on the combined data from both batches
        self.assertNotEqual(thresholds1_low, thresholds2_low)
        self.assertNotEqual(thresholds1_high, thresholds2_high)

        # The new thresholds should reflect the complex combined pattern
        # with wide ranges between split points
        self.assertAlmostEqual(thresholds2_high, 0.54, delta=1e-3)
        self.assertAlmostEqual(thresholds2_low, 0.44, delta=1e-3)

    def test_too_many_samples_behavior(self):
        """
        Test behavior when the number of samples exceeds the maximum.
        """
        # First batch - create samples with high scores (0.8-1.0) all positive
        self.setup_sample_pattern(0.9, 0.05)

        # Create a new sampling state with a smaller max_samples value for testing
        small_samples_state = IterativeSamplingState(
            recall_target=0.8,
            precision_target=0.8,
            failure_probability=0.2,
            max_samples=51,  # Small value to force limit to be exceeded
        )

        # Process first batch
        small_samples_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=self.sampling_result,
        )

        # Get high confidence thresholds after first batch
        thresholds_after_first_low, thresholds_after_first_high = (
            small_samples_state.get_high_confidence_thresholds()
        )

        # Second batch - create samples with medium scores (0.4-0.6)
        self.setup_sample_pattern(0.5, 0.05)

        # Process second batch - this should exceed the MAX_SAMPLES limit and discard ALL first batch samples
        small_samples_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=self.sampling_result,
        )

        # Verify thresholds are now based ONLY on the second batch
        thresholds_after_second_low, thresholds_after_second_high = (
            small_samples_state.get_high_confidence_thresholds()
        )

        # The threshold should now be influenced only by second batch (0.4-0.6)
        # and not by the first batch (0.8-1.0) that should have been completely discarded
        self.assertLess(thresholds_after_second_high, thresholds_after_first_high)
        self.assertLess(thresholds_after_second_low, thresholds_after_first_low)

        # Third batch with low scores (0.1-0.3)
        self.setup_sample_pattern(0.2, 0.05)

        # Process third batch - should discard ALL second batch samples
        small_samples_state.compute_thresholds_with_new_samples(
            oracle_results=self.oracle_results,
            proxy_score=self.confidence_scores,
            sampling_result=self.sampling_result,
        )

        # Verify thresholds are now based ONLY on the third batch
        thresholds_after_third_low, thresholds_after_third_high = (
            small_samples_state.get_high_confidence_thresholds()
        )

        # The threshold should now be influenced only by third batch (0.1-0.3)
        # and not by any previous batches that should have been discarded
        self.assertLess(thresholds_after_third_high, thresholds_after_second_high)
        self.assertLess(thresholds_after_third_low, thresholds_after_second_low)

        # Verify by computing threshold for low confidence rows
        low_conf_threshold = small_samples_state.compute_low_confidence_threshold()

        # In the C++ test, it expects:
        # EXPECT_EQ(lowConfThreshold, 0.36);
        self.assertAlmostEqual(low_conf_threshold, 0.36, delta=1e-3)

    def test_override_corrected_recall_threshold_param(self):
        """
        Tests the impact of `override_corrected_recall_threshold`.
        """
        self.setup_sample_pattern(0.5, 0.4)

        # Case 1: override_corrected_recall_threshold = True
        state_override_true = IterativeSamplingState(
            recall_target=0.9,
            precision_target=0.9,
            failure_probability=0.2,
            override_corrected_recall_threshold=True,
        )
        state_override_true.compute_thresholds_with_new_samples(
            self.oracle_results, self.confidence_scores, self.sampling_result
        )
        (
            low_thresh_true,
            high_thresh_true,
        ) = state_override_true.get_high_confidence_thresholds()

        # Case 2: override_corrected_recall_threshold = False
        state_override_false = IterativeSamplingState(
            recall_target=0.9,
            precision_target=0.9,
            failure_probability=0.2,
            override_corrected_recall_threshold=False,
        )
        state_override_false.compute_thresholds_with_new_samples(
            self.oracle_results, self.confidence_scores, self.sampling_result
        )
        (
            low_thresh_false,
            high_thresh_false,
        ) = state_override_false.get_high_confidence_thresholds()

        # With override=True, corrected recall target is clipped to be >=
        # recall_target. This usually means a lower recall threshold
        # (low_thresh) to accept more positives.
        self.assertLess(low_thresh_true, low_thresh_false)
        # The high threshold for precision should not be significantly
        # affected.
        self.assertAlmostEqual(high_thresh_true, high_thresh_false, delta=0.1)

    def test_replace_small_ub_with_lb_param(self):
        """
        Tests the impact of `replace_small_ub_with_lb`.
        """
        # Use a pattern that creates a clear separation between positives
        # and negatives
        self.setup_sample_pattern(split_point=0.5, mixed_region_width=0)

        # Create a scenario where tau_pos is likely to be less than
        # tau_neg_prime by setting a high recall target and a moderately
        # low precision target.
        recall_target = 0.9
        precision_target = 0.7

        # Case 1: replace_small_ub_with_lb = True (default)
        state_replace_true = IterativeSamplingState(
            recall_target=recall_target,
            precision_target=precision_target,
            failure_probability=0.2,
            replace_small_ub_with_lb=True,
        )
        state_replace_true.compute_thresholds_with_new_samples(
            self.oracle_results, self.confidence_scores, self.sampling_result
        )
        (
            low_thresh_true,
            high_thresh_true,
        ) = state_replace_true.get_high_confidence_thresholds()

        # Case 2: replace_small_ub_with_lb = False
        state_replace_false = IterativeSamplingState(
            recall_target=recall_target,
            precision_target=precision_target,
            failure_probability=0.2,
            replace_small_ub_with_lb=False,
        )
        state_replace_false.compute_thresholds_with_new_samples(
            self.oracle_results, self.confidence_scores, self.sampling_result
        )
        (
            low_thresh_false,
            high_thresh_false,
        ) = state_replace_false.get_high_confidence_thresholds()

        # In the scenario where precision threshold (tau_pos) < recall
        # threshold (tau_neg_prime):
        # With replace=True, high_threshold is max(tau_pos, tau_neg_prime).
        # With replace=False, a balanced threshold is computed for both
        # high and low. This should result in smaller thresholds for both high and low
        self.assertGreater(high_thresh_true, high_thresh_false)
        self.assertGreater(low_thresh_true, low_thresh_false)

        # With replace=False and tau_pos < tau_neg_prime, low and high
        # thresholds are the same.
        self.assertAlmostEqual(low_thresh_false, high_thresh_false)

    def test_corrected_delta_param(self):
        """
        Tests the impact of `corrected_delta` on clipping the corrected
        recall target.
        """
        # A pattern where corrected recall can be high
        self.setup_sample_pattern(split_point=0.3, mixed_region_width=0.2)

        # Case 1: smaller corrected_delta
        state_small_delta = IterativeSamplingState(
            recall_target=0.6,
            precision_target=0.6,
            failure_probability=0.1,
            override_corrected_recall_threshold=True,
            corrected_delta=0.01,  # small delta
        )
        state_small_delta.compute_thresholds_with_new_samples(
            self.oracle_results, self.confidence_scores, self.sampling_result
        )
        (
            low_thresh_small,
            high_thresh_small,
        ) = state_small_delta.get_high_confidence_thresholds()

        # Case 2: larger corrected_delta
        state_large_delta = IterativeSamplingState(
            recall_target=0.6,
            precision_target=0.6,
            failure_probability=0.1,
            override_corrected_recall_threshold=True,
            corrected_delta=0.2,  # large delta
        )
        state_large_delta.compute_thresholds_with_new_samples(
            self.oracle_results, self.confidence_scores, self.sampling_result
        )
        (
            low_thresh_large,
            high_thresh_large,
        ) = state_large_delta.get_high_confidence_thresholds()

        # A larger delta allows for a higher corrected recall target, which
        # generally leads to a lower recall threshold (low_thresh) to
        # include more items.
        self.assertLess(low_thresh_large, low_thresh_small)
        self.assertNotEqual(low_thresh_large, low_thresh_small)

    def test_fix_recall_target_correction(self):
        """
        Test recall target correction is clipped as expected by
        fix_recall_target_correction logic.
        """
        # Setup a pattern where recall correction would overshoot
        self.setup_sample_pattern(split_point=0.3, mixed_region_width=0.2)
        # Use a low recall/precision target to force correction to overshoot
        state = IterativeSamplingState(
            recall_target=0.5,
            precision_target=0.5,
            failure_probability=0.1,
            override_corrected_recall_threshold=False,
            corrected_delta=0.05,
        )
        state.compute_thresholds_with_new_samples(
            self.oracle_results, self.confidence_scores, self.sampling_result
        )
        low_thresh, high_thresh = state.get_high_confidence_thresholds()
        self.assertAlmostEqual(high_thresh, 0.44)
        self.assertAlmostEqual(low_thresh, 0.44)

        state = IterativeSamplingState(
            recall_target=0.5,
            precision_target=0.5,
            failure_probability=0.1,
            override_corrected_recall_threshold=True,
            corrected_delta=0.05,
        )
        state.compute_thresholds_with_new_samples(
            self.oracle_results, self.confidence_scores, self.sampling_result
        )
        low_thresh, high_thresh = state.get_high_confidence_thresholds()
        self.assertAlmostEqual(high_thresh, 0.6)
        self.assertAlmostEqual(low_thresh, 0.6)


if __name__ == "__main__":
    unittest.main()
