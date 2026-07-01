import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

from icefall.cascades.sampling.sampling_method import SamplingResult
from icefall.cascades.sampling.sampling_state import SamplingState


class IterativeSamplingState(SamplingState):
    def __init__(
        self,
        recall_target: float,
        precision_target: float,
        failure_probability: float,
        max_samples: int = 1048576,
        fix_recall_target_correction: bool = False,
        override_corrected_recall_threshold: bool = False,
        replace_small_ub_with_lb: bool = True,
        corrected_delta: float = 0.05,
    ) -> None:
        """
        Initialize the IterativeSamplingState with recall and precision targets.

        Args:
            recall_target (float): The target recall.
            precision_target (float): The target precision.
            failure_probability (float): The probability of failure.
            max_samples (int): The maximum number of samples to consider.
            override_corrected_recall_threshold (bool): If True, override the corrected recall threshold with min(threshold, corrected_threshold)
            replace_small_ub_with_lb (bool): If True, replace small upper bound with max(lower bound, upper bound).
        """

        super().__init__()
        self.recall_target = recall_target
        self.precision_target = precision_target
        self.failure_probability = failure_probability
        self.max_samples = max_samples
        self.sample_info = pd.DataFrame(
            {
                "score": [],
                "oracle_result": [],
                "correction_factor": [],
            },
        )
        self.sample_info = self.sample_info.astype(
            {
                "score": float,
                "oracle_result": bool,
                "correction_factor": float,
            }
        )
        self._upper_bound_history: list[float] = []
        self._lower_bound_history: list[float] = []
        self.fix_recall_target_correction = fix_recall_target_correction
        self.override_corrected_recall_threshold = override_corrected_recall_threshold
        self.replace_small_ub_with_lb = replace_small_ub_with_lb
        self.corrected_delta = corrected_delta if corrected_delta is not None else 0.05

    @staticmethod
    def _UB(mean: float, std_dev: float, s: int, delta: float) -> float:
        return float(mean + (std_dev / (s**0.5)) * ((2 * np.log(1 / delta)) ** 0.5))

    @staticmethod
    def _LB(mean: float, std_dev: float, s: int, delta: float) -> float:
        return float(mean - (std_dev / (s**0.5)) * ((2 * np.log(1 / delta)) ** 0.5))

    def _calculate_tau_neg(self, recall_target: float) -> float:
        _, tpr, thresholds = roc_curve(
            y_true=self.sample_info["oracle_result"],
            y_score=self.sample_info["score"],
            sample_weight=self.sample_info["correction_factor"],
            drop_intermediate=False,
        )
        # Drop threshold=infinity, and corresponding TPR value.
        thresholds = thresholds[1:]
        tpr = tpr[1:]
        for i in range(len(tpr)):
            if tpr[i] >= recall_target:
                # Off-by-one adjustment for strict/open interval at the recall cutoff.
                if i + 1 < len(thresholds):
                    logging.debug(
                        f"Threshold for the recall target thresholds[i+1]: {thresholds[i + 1]}"
                    )
                    logging.debug(f"TPR[i+1]: {tpr[i + 1]}")
                    return thresholds[i + 1]
        return 0.0

    def compute_balanced_threshold(self):
        precisions, recalls, thresholds = precision_recall_curve(
            self.sample_info["oracle_result"], self.sample_info["score"]
        )
        eps = 1e-10  # Small epsilon to avoid division by zero
        ratio_array = recalls / (precisions + eps)
        # compute the one that is closest to the target ratio of recall/precision
        return thresholds[
            np.argmin(np.abs(ratio_array - self.recall_target / self.precision_target))
        ]

    def compute_thresholds_with_new_samples(
        self,
        oracle_results: pd.Series,
        proxy_score: pd.Series,
        sampling_result: SamplingResult,
    ) -> None:

        current_sample_size = len(sampling_result.sample_selections)
        if current_sample_size == 0:
            return
        if len(self.sample_info) + current_sample_size > self.max_samples:
            self.sample_info = pd.DataFrame(
                {
                    "score": [],
                    "oracle_result": [],
                    "correction_factor": [],
                }
            )
            self.sample_info = self.sample_info.astype(
                {
                    "score": float,
                    "oracle_result": bool,
                    "correction_factor": float,
                }
            )

        self.sample_info = pd.concat(
            [
                self.sample_info,
                pd.DataFrame(
                    {
                        "score": proxy_score[sampling_result.sample_selections].astype(float),
                        "oracle_result": oracle_results[sampling_result.sample_selections].astype(
                            bool
                        ),
                        "correction_factor": np.array(sampling_result.correction_factors).astype(
                            float
                        ),
                    }
                ),
            ],
            ignore_index=True,
        )
        self.sample_info = self.sample_info.astype(
            {
                "score": float,
                "oracle_result": bool,
                "correction_factor": float,
            }
        )
        self.sample_info.sort_values(by="score", ascending=False, inplace=True)

        sample_size = len(self.sample_info)

        # Find tau_negative based on recall
        tau_neg_0 = self._calculate_tau_neg(self.recall_target)

        # Do a statistical correction to get a new target recall
        Z1 = []
        Z2 = []
        for _, row in self.sample_info.iterrows():
            z = int(row["oracle_result"]) * row["correction_factor"]
            if row["score"] >= tau_neg_0:
                Z1.append(z)
                if self.fix_recall_target_correction:
                    Z2.append(0)
            else:
                Z2.append(z)
                if self.fix_recall_target_correction:
                    Z1.append(0)

        mean_z1 = float(np.mean(Z1)) if Z1 else 0.0
        std_z1 = float(np.std(Z1)) if Z1 else 0.0
        mean_z2 = float(np.mean(Z2)) if Z2 else 0.0
        std_z2 = float(np.std(Z2)) if Z2 else 0.0

        ub_z1 = self._UB(mean_z1, std_z1, sample_size, self.failure_probability / 2)
        lb_z2 = self._LB(mean_z2, std_z2, sample_size, self.failure_probability / 2)
        if ub_z1 + lb_z2 == 0:  # Avoid division by zero
            corrected_recall_target = 1.0
        else:
            corrected_recall_target = ub_z1 / (ub_z1 + lb_z2)
        corrected_recall_target = min(1, corrected_recall_target)
        logging.debug("Corrected recall target before clipping: %s", corrected_recall_target)

        # After clipping, such that it does not go too much above recall target
        min_recall_clip = self.recall_target if self.override_corrected_recall_threshold else 0.0
        corrected_recall_target = np.clip(
            corrected_recall_target,
            min_recall_clip,
            self.recall_target + self.corrected_delta
            if self.override_corrected_recall_threshold
            else 1.0,
        )
        logging.debug("Corrected recall target after clipping: %s", corrected_recall_target)

        tau_neg_prime = self._calculate_tau_neg(corrected_recall_target)
        logging.debug("Tau negative prime: %s", tau_neg_prime)

        # Do a statistical correction to get a target satisfying precision
        candidate_thresholds: list[float] = [1.0]
        outputs_grouped_by_proxy_score = (
            self.sample_info.groupby("score")["oracle_result"]
            .agg(["sum", "count"])
            .sort_index(ascending=False)
        )
        cum_stats = outputs_grouped_by_proxy_score.expanding(1).sum()
        cum_stats["mean"] = cum_stats["sum"] / cum_stats["count"]
        cum_stats["diff"] = cum_stats["count"] - cum_stats["sum"]
        cum_stats["std"] = np.sqrt(
            (
                cum_stats["diff"] * (cum_stats["sum"] / cum_stats["count"]) ** 2
                + cum_stats["sum"] * (cum_stats["diff"] / cum_stats["count"]) ** 2
            )
            / cum_stats["count"]
        )
        cum_stats = cum_stats.reset_index()

        for _, row in cum_stats.iterrows():
            p_l = self._LB(
                row["mean"],
                row["std"],
                row["count"],
                self.failure_probability / len(self.sample_info),
            )
            if p_l > self.precision_target:
                candidate_thresholds.append(row["score"])

        tau_pos = min(candidate_thresholds)
        best_combination = (  # (upper_bound, lower_bound)
            max(tau_neg_prime, tau_pos) if self.replace_small_ub_with_lb else tau_pos,
            tau_neg_prime,
        )
        logging.debug("Best candidate threshold (ub, lb): %s", best_combination)

        if not self.replace_small_ub_with_lb and best_combination[0] < best_combination[1]:
            # if the lb > ub and it is not replaced, then we will select the middle point as both the upper bound and
            # lower bound. Since mid < lower bound (recall threshold), that means we are being more conservative on
            # marking FALSE, which will lead to an increase in recall. Since mid > upper bound (precision threshold),
            # that means we are being more conservative on marking TRUE, which will lead to an increase in precision.
            # mid = np.mean(best_combination)
            # best_combination = (mid, mid)
            threshold = self.compute_balanced_threshold()
            logging.debug("Using balanced threshold: %s", threshold)
            best_combination = (threshold, threshold)

        upper_bound, lower_bound = best_combination
        self.low_threshold = lower_bound
        self.high_threshold = upper_bound
        self._upper_bound_history.append(upper_bound)
        self._lower_bound_history.append(lower_bound)

    def compute_low_confidence_threshold_impl(self) -> float:
        y_score = self.sample_info["score"]
        y_score_values = np.sort(np.unique(y_score))
        y_true = self.sample_info["oracle_result"]
        thresholds = (y_score_values[:-1] + y_score_values[1:]) / 2
        f1_scores = [
            f1_score(y_true=y_true, y_pred=y_score > threshold) for threshold in thresholds
        ]
        i = np.argmax(f1_scores)
        best_threshold = thresholds[i]
        return best_threshold
