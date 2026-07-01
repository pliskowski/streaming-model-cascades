from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from pandas import CategoricalDtype
from pygam import LogisticGAM, s
from pygam.utils import OptimizationError
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer

from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
from icefall.cascades.model_cascade import ModelCascade
from icefall.cascades.sampling.sampling_method import SamplingMethod, SamplingResult
from icefall.cascades.sampling.sampling_state import SamplingState

ModelMethodType = Literal["naive_ci", "monotonic_inc", "custom", "identity", "platt", "isotonic"]
OptimizationMetricType = Literal["f1", "accuracy"]
SamplingStrategyType = Literal["uniform", "info_valued", "info_decay", "class_only"]
InfoMethodType = Literal["ci_width", "fisher"]


def _calculate_cost_left(score: pd.Series, left: float, scale: float) -> float:
    """Calculate the normalized cost for scores at or above the left threshold.

    Args:
        score: Series of proxy scores.
        left: Left threshold value.
        scale: Scaling factor for normalization.

    Returns:
        Normalized cost (proportion of scores >= left threshold).
    """
    n = (score >= left).sum()
    return n / scale


def _calculate_error_left(score: pd.Series, beta: float, left: float, scale: float = 1.0) -> float:
    """Calculate the normalized error for scores below the left threshold.

    Computes weighted error for false negatives (scores below threshold that should
    have been classified as positive).

    Args:
        score: Series of proxy scores representing probabilities.
        beta: Weight parameter for balancing error components.
        left: Left threshold value.
        scale: Scaling factor for normalization. Defaults to 1.0.

    Returns:
        Normalized weighted error for scores below the threshold.
    """
    below_threshold = score < left
    errors = (1 - beta) * score * below_threshold
    return errors.sum() / scale


def _func_left(
    left: float,
    score_left: pd.Series,
    alpha: float,
    beta: float,
    error_scaling: float,
    cost_scaling: float,
) -> float:
    """Objective function for optimizing the left threshold.

    Computes a weighted combination of error and cost for the left threshold,
    balancing between accuracy and computational cost.

    Args:
        left: Left threshold value to evaluate.
        score_left: Series of proxy scores below 0.5.
        alpha: Weight for error vs. cost trade-off (0=cost only, 1=error only).
        beta: Weight parameter for balancing error components.
        error_scaling: Scaling factor for error normalization.
        cost_scaling: Scaling factor for cost normalization.

    Returns:
        Weighted sum of normalized error and cost.
    """
    error = _calculate_error_left(score=score_left, beta=beta, left=left, scale=error_scaling)
    cost = _calculate_cost_left(score=score_left, left=left, scale=cost_scaling)
    metric_sum = alpha * error + (1 - alpha) * cost
    return metric_sum


def _calculate_cost_right(score: pd.Series, right: float, scale: float) -> float:
    """Calculate the normalized cost for scores below the right threshold.

    Args:
        score: Series of proxy scores.
        right: Right threshold value.
        scale: Scaling factor for normalization.

    Returns:
        Normalized cost (proportion of scores < right threshold).
    """
    n = (score < right).sum()
    return n / scale


def _calculate_error_right(
    score: pd.Series, beta: float, right: float, scale: float = 1.0
) -> float:
    """Calculate the normalized error for scores at or above the right threshold.

    Computes weighted error for false positives (scores above threshold that should
    have been classified as negative).

    Args:
        score: Series of proxy scores representing probabilities.
        beta: Weight parameter for balancing error components.
        right: Right threshold value.
        scale: Scaling factor for normalization. Defaults to 1.0.

    Returns:
        Normalized weighted error for scores at or above the threshold.
    """
    above_threshold = score >= right
    errors = beta * (1 - score) * above_threshold
    return errors.sum() / scale


def _func_right(
    right: float,
    score_right: pd.Series,
    alpha: float,
    beta: float,
    error_scaling: float,
    cost_scaling: float,
) -> float:
    """Objective function for optimizing the right threshold.

    Computes a weighted combination of error and cost for the right threshold,
    balancing between accuracy and computational cost.

    Args:
        right: Right threshold value to evaluate.
        score_right: Series of proxy scores at or above 0.5.
        alpha: Weight for error vs. cost trade-off (0=cost only, 1=error only).
        beta: Weight parameter for balancing error components.
        error_scaling: Scaling factor for error normalization.
        cost_scaling: Scaling factor for cost normalization.

    Returns:
        Weighted sum of normalized error and cost.
    """
    error = _calculate_error_right(score=score_right, beta=beta, right=right, scale=error_scaling)
    cost = _calculate_cost_right(score=score_right, right=right, scale=cost_scaling)
    metric_sum = alpha * error + (1 - alpha) * cost
    return metric_sum


def _calculate_cost(scores: pd.Series, left: float, right: float, scale: float) -> float:
    """Calculate the normalized cost for scores between the two thresholds.

    The cost represents the proportion of samples that fall in the uncertain region
    and require oracle evaluation.

    Args:
        scores: Series of proxy scores.
        left: Left threshold value.
        right: Right threshold value.
        scale: Scaling factor for normalization.

    Returns:
        Normalized cost (proportion of scores in [left, right) range).
    """
    n = scores.between(left=left, right=right, inclusive="left").sum()
    return n / scale


def _calculate_f1(scores: pd.Series, left: float, right: float, beta: float) -> float:
    """Calculate the expected F-beta score for given thresholds.

    Computes the weighted harmonic mean of precision and recall based on proxy scores
    and threshold values. The calculation assumes:
    - Scores < left threshold: classified as negative (can produce FN)
    - Scores >= right threshold: classified as positive (can produce FP)
    - Scores in [left, right): sent to oracle (always correct)

    Args:
        scores: Series of proxy scores representing probabilities.
        left: Left threshold value (reject below this).
        right: Right threshold value (accept at or above this).
        beta: Weight parameter for F-beta score (beta=1 for F1, beta>1 favors recall).

    Returns:
        Expected F-beta score for the given thresholds.

    Note:
        For scores less than the left threshold, they will all be labeled
        negative, so we can only generate True-Negative and False-Negative counts from that
        region. For the scores greater than or equal to the upper threshold, they will all be labeled
        positive, so we can only generate True-Positives and False-Positives from that
        region. For the scores between the two thresholds, we will call the oracle, so they
        will only be True-Positives or True-Negatives.

    Note: This is a simplified version of the F-beta score calculation, using plug-in estimates. A more accurate estimate might use MCMC methods.
    """
    lower_scores = scores[scores < left]
    inner_scores = scores[scores.between(left, right, inclusive="left")]
    upper_scores = scores[scores >= right]

    # lower_expected_tn = sum(1 - lower_scores)
    lower_expected_fn = sum(lower_scores)

    inner_expected_tp = sum(inner_scores)
    # inner_expected_tn = sum(1 - inner_scores)

    upper_expected_tp = sum(upper_scores)
    upper_expected_fp = sum(1 - upper_scores)

    expected_tp = inner_expected_tp + upper_expected_tp
    if expected_tp == 0.0:
        expected_f1_score = 0.0
    else:
        expected_f1_score = (
            (1 + beta**2)
            * expected_tp
            / ((1 + beta**2) * expected_tp + lower_expected_fn + beta**2 * upper_expected_fp)
        )
    return expected_f1_score


def _func_accuracy(
    x: np.ndarray,
    scores: pd.Series,
    alpha: float,
    beta: float,
    error_scaling: float,
    cost_scaling: float,
) -> float:
    """Objective function for optimizing both thresholds using accuracy-based error.

    Combines the left and right threshold optimization objectives for finding
    optimal thresholds that balance accuracy and cost.

    Args:
        x: Array containing [left, right] threshold values.
        scores: Series of proxy scores.
        alpha: Weight for error vs. cost trade-off (0=cost only, 1=error only).
        beta: Weight parameter for balancing error components.
        error_scaling: Scaling factor for error normalization.
        cost_scaling: Scaling factor for cost normalization.

    Returns:
        Combined objective value (sum of left and right objectives).
    """
    left, right = x

    result_left = _func_left(
        left=left,
        score_left=scores[scores < 0.5],
        alpha=alpha,
        beta=beta,
        error_scaling=error_scaling,
        cost_scaling=cost_scaling,
    )
    result_right = _func_right(
        right=right,
        score_right=scores[scores >= 0.5],
        alpha=alpha,
        beta=beta,
        error_scaling=error_scaling,
        cost_scaling=cost_scaling,
    )
    return result_left + result_right


def _func_f1(
    x: np.ndarray,
    scores: pd.Series,
    alpha: float,
    beta: float,
    error_scaling: float,
    cost_scaling: float,
) -> float:
    """Objective function for optimizing both thresholds using F-beta score.

    Optimizes thresholds to maximize F-beta score while minimizing oracle calls,
    balancing between precision, recall, and computational cost.

    Args:
        x: Array containing [left, right] threshold values.
        scores: Series of proxy scores.
        alpha: Weight for error vs. cost trade-off (0=cost only, 1=error only).
        beta: Weight parameter for F-beta score (beta=1 for F1).
        error_scaling: Scaling factor for error normalization.
        cost_scaling: Scaling factor for cost normalization.

    Returns:
        Weighted sum of normalized error (1 - F-beta) and cost.
    """
    left, right = x

    cost = _calculate_cost(scores=scores, left=left, right=right, scale=cost_scaling)
    error = 1 - _calculate_f1(scores=scores, left=left, right=right, beta=beta)
    error /= error_scaling

    return alpha * error + (1 - alpha) * cost


def _estimate_bounds_f1(scores: pd.Series, alpha: float, beta: float) -> tuple[float, float]:
    """Estimate optimal left and right thresholds by minimizing scaled sum of error and cost.

    Uses Nelder-Mead optimization to find threshold values that optimize the
    trade-off between error and cost

    Args:
        scores: Series of proxy scores representing probabilities.
        alpha: Weight for error vs. cost trade-off (0=cost only, 1=error only).
        beta: Weight parameter for balancing error components.

    Returns:
        Tuple of (left_threshold, right_threshold) values.
    """
    error_scale = 1 - _calculate_f1(scores=scores, left=0.5, right=0.5, beta=beta)
    cost_scale = scores.shape[0]

    partial_kwargs = dict(
        scores=scores,
        alpha=alpha,
        beta=beta,
        error_scaling=error_scale,
        cost_scaling=cost_scale,
    )
    partial_func = partial(_func_f1, **partial_kwargs)

    def convert_input(y: np.ndarray) -> np.ndarray:
        x0 = y[0]
        # This maps y[1] from [0, 1] to the [x0, 1.0] range
        x1 = x0 + (1.0 - x0) * y[1]
        return np.array([x0, x1])

    def wrapper_func(y: np.ndarray) -> float:
        return partial_func(convert_input(y))

    minimize_result = differential_evolution(func=wrapper_func, bounds=[[0, 1.0], [0, 1.0]])
    lb, ub = convert_input(minimize_result.x)
    return lb, ub


def _estimate_bounds_accuracy(scores: pd.Series, alpha: float, beta: float) -> tuple[float, float]:
    """Estimate optimal left and right thresholds by minimizing scaled sum of error and cost.

    Args:
        scores: Series of proxy scores representing probabilities.
        alpha: Weight for error vs. cost trade-off (0=cost only, 1=error only).
        beta: Weight parameter for error calculation.

    Returns:
        Tuple of (left_threshold, right_threshold) values.
    """
    score_left = scores[scores < 0.5]
    max_error_left = _calculate_error_left(score_left, beta=beta, left=0.5)
    score_right = scores[scores >= 0.5]
    max_error_right = _calculate_error_right(score_right, beta=beta, right=0.5)
    error_scale = max_error_left + max_error_right
    cost_scale = scores.shape[0]

    partial_kwargs = dict(
        scores=scores,
        alpha=alpha,
        beta=beta,
        error_scaling=error_scale,
        cost_scaling=cost_scale,
    )
    partial_func = partial(_func_accuracy, **partial_kwargs)
    minimize_result = differential_evolution(
        func=partial_func,
        bounds=[[0, 0.5], [0.5, 1.0]],
    )

    return minimize_result.x[0], minimize_result.x[1]


class RegularizedLogisticRegression:
    """Implements L2-Regularized Logistic Regression without an intercept term.,

    Uses the SciPy L-BFGS-B optimization method.
    Assumes input X is a 2D NumPy array and y is a 1D NumPy array of {0, 1}.
    """

    def __init__(
        self,
        mu_param: np.array,
        lambda_param: float = 1.0,
        max_iterations: int = 10_000,
    ):
        """
        Initializes the solver parameters.

        Note: The learning_rate is handled internally by the L-BFGS-B solver
        and is no longer a parameter here.
        """
        self.mu_param = mu_param
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations
        self.coef_: np.ndarray | None = None

        self.covariance_matrix_: np.ndarray | None = None

    @staticmethod
    def _objective_function_and_gradient(
        coef: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        mu_param: np.ndarray,
        lambda_param: float,
    ) -> tuple[float, float]:
        """Calculates the L2-Regularized Log-Loss Cost and its Gradient.

        This function is the required interface for scipy.optimize.minimize.
        It takes the current weights as the first argument and returns (cost, gradient).
        """
        m = X.shape[0]

        # 1. Calculate Hypothesis (Predicted Probabilities)
        z = X @ coef
        h = expit(np.clip(z, -5, 5))

        # 2. Calculate Cost (Log-Loss + L2 Penalty)
        epsilon = 1e-15

        # Log-Loss Term (Binary Cross-Entropy)
        log_loss = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

        # L2 Regularization Term (Penalty applied to all weights, as no intercept)
        l2_penalty = (lambda_param / (2 * m)) * np.sum((coef - mu_param) ** 2)

        cost = log_loss + l2_penalty

        # 3. Calculate Gradient

        # Prediction error
        error = h - y

        # Gradient of Log-Loss: (1/m) * X.T @ error
        log_loss_gradient = (1 / m) * (X.T @ error)

        # Gradient of L2 Penalty: (lambda/m) * theta
        regularization_gradient = (lambda_param / m) * (coef - mu_param)

        # The full gradient for all weights
        gradient = log_loss_gradient + regularization_gradient

        return cost, gradient

    def _compute_cov(self, X: np.ndarray, y: np.ndarray) -> None:
        m, n = X.shape

        # 1. Get predictions at the optimum
        z = X @ self.coef_
        h = expit(np.clip(z, -5, 5))

        # 2. Compute the Hessian of the Log-Likelihood (Fisher Information)
        # H_likelihood = (1/m) * X.T @ S @ X, where S is diag(h * (1-h))
        # We compute this efficiently without creating the full diagonal matrix S
        S_vector = h * (1 - h)
        # Using broadcasting for efficiency: X.T @ (S * X)
        hessian_likelihood = X.T @ (S_vector[:, None] * X)

        # 3. Compute Hessian of the L2 Penalty
        # The second derivative of (lambda/2m) * sum(theta^2) is (lambda/m) * Identity
        hessian_penalty = (self.lambda_param) * np.eye(n)

        # 4. Total Hessian
        hessian_total = hessian_likelihood + hessian_penalty

        # 5. Covariance is the Inverse Hessian
        self.covariance_matrix_ = np.linalg.pinv(hessian_total)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model to the data using the L-BFGS-B optimization solver.

        Args:
            X (np.ndarray): The 2D design matrix (m samples, n features).
            y (np.ndarray): The 1D target vector (m samples) with values {0, 1}.
        """
        m, n = X.shape

        # Initialize weights (theta) randomly
        initial_weights = np.random.randn(n) * 0.01

        # Call SciPy's minimize function
        # method='L-BFGS-B' is an excellent choice for this type of problem.
        # jac=True tells the solver that our objective function returns both the cost (f)
        # and the gradient (g), which drastically speeds up convergence.
        optimization_result = minimize(
            fun=self._objective_function_and_gradient,
            x0=initial_weights,
            args=(
                X,
                y,
                self.mu_param,
                self.lambda_param,
            ),  # Extra arguments to pass to the objective function
            method="L-BFGS-B",
            jac=True,  # Specifies that the function returns the gradient
            options={"maxiter": self.max_iterations},
        )

        # Store the optimal weights and final cost
        self.coef_ = optimization_result.x

        if not optimization_result.success:
            print(
                f"Warning: Optimization failed to converge. Message: {optimization_result.message}"
            )

        self._compute_cov(X, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for the given data.

        Args:
            X (np.ndarray): The 2D design matrix.

        Returns:
            np.ndarray: The predicted probabilities of P(y=1|X).
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        z = X @ self.coef_
        return expit(z)

    def confidence_intervals(self, X: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        z = X @ self.coef_
        se_z = np.sqrt(np.sum((X @ self.covariance_matrix_) * X, axis=1))

        if quantiles.shape[0] != z.shape[0]:
            z = z[:, np.newaxis]
            se_z = se_z[:, np.newaxis]

        z_ci = norm.ppf(q=quantiles, loc=z, scale=se_z)
        return expit(z_ci)


class GamCalDynamicCascadeSamplingMethod(SamplingMethod):
    """Simple uniform random sampling method for the GAM calibration cascade.

    Samples indices uniformly at random without replacement from the pool of
    low-confidence selections that require oracle evaluation.
    """

    def sample_impl(
        self, input_selections: list[int], proxy_score: pd.Series, sample_size: int
    ) -> tuple[list[int], list[float]]:
        """Sample indices uniformly at random without replacement.

        Args:
            input_selections: List of available indices to sample from.
            proxy_score: Series of proxy scores (not used in this implementation).
            sample_size: Number of samples to draw.

        Returns:
            Tuple of (sampled_indices, correction_factors). The correction_factors
            list is empty as uniform sampling doesn't require corrections.
        """
        size = min(sample_size, len(proxy_score))
        sample_index = np.random.choice(input_selections, size=size, replace=False)
        # Correction factor not used.
        return list(sample_index), []


class InfoValuedSamplingMethod(SamplingMethod):
    """Sampling proportional to U(i) = w_class * V_class(i) + lambda * V_info(i).

    V_class: classification uncertainty (1 - max(g, 1-g)) where g is calibrated prob.
    V_info: calibrator uncertainty (CI width or Fisher information gain).

    Falls back to uniform sampling when no GAM model has been trained yet.
    """

    def __init__(
        self,
        batch_size: int,
        sampling_state: "GamCalDynamicCascadeSamplingState",
        info_lambda: float = 0.5,
        info_method: InfoMethodType = "ci_width",
        class_value_weight: float = 1.0,
    ):
        super().__init__(batch_size=batch_size)
        self.sampling_state = sampling_state
        self.info_lambda = info_lambda
        self.info_method = info_method
        self.class_value_weight = class_value_weight

    def sample_impl(
        self, input_selections: list[int], proxy_score: pd.Series, sample_size: int
    ) -> tuple[list[int], list[float]]:
        size = min(sample_size, len(input_selections))
        if size <= 0:
            return [], []

        if self.sampling_state.model_ is None:
            sample_index = np.random.choice(input_selections, size=size, replace=False)
            return list(sample_index), []

        candidates = np.array(input_selections)
        raw_scores = proxy_score.iloc[candidates].values

        calibrated = self.sampling_state.calibrate_proxy_scores(
            pd.Series(raw_scores, index=candidates), q=0
        )
        g = np.clip(calibrated.values, 1e-8, 1 - 1e-8)
        v_class = 1.0 - np.maximum(g, 1.0 - g)

        if self.info_lambda > 0:
            v_info = self.sampling_state.compute_info_value(raw_scores, method=self.info_method)
            v_info_max = v_info.max()
            if v_info_max > 0:
                v_info = v_info / v_info_max
        else:
            v_info = np.zeros(len(candidates))

        utility = self.class_value_weight * v_class + self.info_lambda * v_info
        utility = np.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0)
        utility = np.maximum(utility, 1e-10)
        probs = utility / utility.sum()

        sample_index = np.random.choice(candidates, size=size, replace=False, p=probs)
        return list(sample_index), []


class GamCalDynamicCascadeSamplingState(SamplingState):
    """Maintains state for GAM-based calibration and threshold optimization.

    This class manages the training and calibration of Generalized Additive Models (GAMs)
    that map proxy scores to calibrated probabilities. It also optimizes decision thresholds
    to balance accuracy and computational cost.

    Attributes:
        smooth_lambda: Smoothing parameter for the GAM model.
        model_method: Calibration method ("naive_ci" or "monotonic_inc").
        opt_method: Optimization method for thresholds ("f1" or "accuracy").
        min_train: Minimum samples per class before training a model.
        low_threshold: Lower decision threshold (reject below this).
        high_threshold: Upper decision threshold (accept at or above this).
        model_: Trained LogisticGAM model for calibration.
        models_: Bootstrap ensemble of GAMs (used with "monotonic_inc" method).
        last_trained: Number of samples when model was last trained.
        sample_info: DataFrame storing accumulated oracle samples and proxy scores.
    """

    def __init__(
        self,
        smooth_lambda: float,
        model_method: ModelMethodType,
        opt_method: OptimizationMetricType,
        min_train: int = 10,
        adaptive_lambda: bool = False,
    ) -> None:
        """Initialize the sampling state.

        Args:
            smooth_lambda: Smoothing parameter (lambda) for GAM regularization.
                Used as the initial/fixed lambda depending on adaptive_lambda.
            model_method: Calibration method to use:
                - "naive_ci": Uses GAM confidence intervals for calibration
                - "monotonic_inc": Uses bootstrap ensemble with logit-space calibration
                - "custom": Uses regularized logistic regression with splines
                - "platt": 2-parameter Platt scaling (sklearn LogisticRegression)
                - "isotonic": Non-parametric monotonic isotonic regression
            opt_method: Threshold optimization method:
                - "f1": Optimize F-beta score
                - "accuracy": Optimize classification accuracy
            min_train: Minimum number of samples per class required before training.
                Prevents overfitting when data is limited.
            adaptive_lambda: If True, use GCV-based grid search to select
                lambda at each retraining event instead of using smooth_lambda.
                Only affects naive_ci and monotonic_inc methods.
        """
        super().__init__()
        self.smooth_lambda = smooth_lambda
        self.model_method = model_method
        self.opt_method = opt_method
        self.min_train = min_train
        self.adaptive_lambda = adaptive_lambda

        self.low_threshold = 0.0
        self.high_threshold = 1.0

        self.model_: LogisticGAM | RegularizedLogisticRegression | object | None = None
        self.models_: list[LogisticGAM] | None = None
        # The number of observations accumulated since the last model training.
        self.last_trained: int = -1

        # The accumulating oracle sampled data, and corresponding proxy scores.
        self.sample_info = pd.DataFrame(
            data=dict(proxy_score=pd.Series(dtype=float), oracle_results=pd.Series(dtype=bool))
        )

        self.fisher_inv_: np.ndarray | None = None

    def update_sample_info(
        self,
        oracle_results: pd.Series,
        proxy_score: pd.Series,
        sampling_result: SamplingResult,
    ) -> None:
        """Update the accumulated sample information with new oracle results.

        Appends newly sampled oracle labels and their corresponding proxy scores to
        the training dataset, maintaining sorted order by proxy score.

        Args:
            oracle_results: Series of ground truth labels from oracle evaluation.
            proxy_score: Series of proxy model scores for all samples.
            sampling_result: SamplingResult containing indices of newly sampled items.
        """
        # Note how large the sample selection is
        current_sample_size = len(sampling_result.sample_selections)
        if current_sample_size == 0:
            return

        self.sample_info = pd.concat(
            [
                self.sample_info,
                pd.DataFrame(
                    data=dict(
                        proxy_score=proxy_score[sampling_result.sample_selections]
                        .rename("proxy_score")
                        .astype(float),
                        oracle_results=oracle_results[sampling_result.sample_selections]
                        .rename("oracle_results")
                        .astype(float),
                    )
                ),
            ],
            ignore_index=True,
        ).sort_values(by="proxy_score", ascending=False)

    def compute_low_confidence_threshold_impl(self) -> float:
        # Fixed midpoint; could be tuned from recall/precision trade-offs.
        return 0.5

    def check_if_train_model(self) -> bool:
        """Check whether enough new samples have been collected to retrain the model.

        Uses a simple heuristic: retrain when the dataset size has doubled since
        the last training.

        Returns:
            True if model should be retrained, False otherwise.
        """
        if (
            self.sample_info["oracle_results"]
            .astype(CategoricalDtype(categories=[0.0, 1.0]))
            .value_counts()
            .min()
            < self.min_train
        ):
            return False

        return len(self.sample_info) >= 2 * self.last_trained

    def calibrate_proxy_scores_naive_ci(self, scores: pd.Series, q: pd.Series | int) -> pd.Series:
        """Calibrate proxy scores using GAM confidence intervals.

        Draws q uniformly in [0, 1] per record (see harness ``quantile_distribution``)
        and maps it through the GAM mean and approximate one-sigma band in probability
        space. This is the default calibration path used for the paper experiments.
        """
        x_mat = scores.to_numpy().reshape((-1, 1))
        mu_and_mu_plus_sigma = self.model_.confidence_intervals(
            x_mat, quantiles=[0.5, 0.8413447460685429]
        )
        mu = mu_and_mu_plus_sigma[:, 0]
        sigma = mu_and_mu_plus_sigma[:, 1] - mu_and_mu_plus_sigma[:, 0]
        calibrated_scores = pd.Series(data=mu + q * sigma, index=scores.index, name=scores.name)
        return calibrated_scores

    def calibrate_proxy_scores_monotonic_inc(
        self, scores: pd.Series, q: pd.Series | int
    ) -> pd.Series:
        """Calibrate proxy scores using bootstrap ensemble in logit space."""
        x_mat = scores.to_numpy().reshape((-1, 1))
        mu = self.model_.predict_proba(x_mat)
        boot_mus = np.vstack([model.predict_proba(x_mat) for model in self.models_])

        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        boot_mus = np.clip(boot_mus, 1e-8, 1 - 1e-8)

        logit_boot_mus = logit(boot_mus)
        logit_boot_mus -= logit_boot_mus.mean(axis=0)
        calibrated_scores = expit(logit(mu) + np.quantile(logit_boot_mus, q=q.to_numpy()))
        return pd.Series(calibrated_scores, index=scores.index)

    def calibrate_proxy_scores_custom(self, scores: pd.Series, q: pd.Series | int) -> pd.Series:
        log_odds_max = 5
        log_odds_min = -log_odds_max
        lo_train = np.clip(logit(scores.astype(float)), log_odds_min, log_odds_max)

        x_mat = self.spline_transformer_.transform(lo_train.to_numpy().reshape((-1, 1)))
        calibrated_scores = self.model_.confidence_intervals(X=x_mat, quantiles=q)
        return pd.Series(calibrated_scores, index=scores.index)

    def calibrate_proxy_scores_platt(self, scores: pd.Series, q: pd.Series | int) -> pd.Series:
        x_mat = scores.to_numpy().reshape((-1, 1))
        calibrated = self.model_.predict_proba(x_mat)[:, 1]
        return pd.Series(calibrated, index=scores.index)

    def calibrate_proxy_scores_isotonic(self, scores: pd.Series, q: pd.Series | int) -> pd.Series:
        calibrated = self.model_.predict(scores.to_numpy())
        calibrated = np.clip(calibrated, 1e-8, 1 - 1e-8)
        return pd.Series(calibrated, index=scores.index)

    def calibrate_proxy_scores(self, scores: pd.Series, q: pd.Series | int) -> pd.Series:
        """Calibrate proxy scores using the configured calibration method.

        Dispatches to the appropriate calibration method based on model_method setting.
        Returns uncalibrated scores if no model has been trained yet.

        Args:
            scores: Series of raw proxy scores to calibrate.
            q: Quantile parameter(s) for confidence adjustment.

        Returns:
            Series of calibrated scores (or original scores if model is None).

        Raises:
            ValueError: If model_method is not supported.
        """
        if self.model_ is None:
            return scores

        match self.model_method:
            case "naive_ci":
                calibrated_scores = self.calibrate_proxy_scores_naive_ci(scores=scores, q=q)
            case "monotonic_inc":
                calibrated_scores = self.calibrate_proxy_scores_monotonic_inc(scores=scores, q=q)
            case "custom":
                calibrated_scores = self.calibrate_proxy_scores_custom(scores=scores, q=q)
            case "platt":
                calibrated_scores = self.calibrate_proxy_scores_platt(scores=scores, q=q)
            case "isotonic":
                calibrated_scores = self.calibrate_proxy_scores_isotonic(scores=scores, q=q)
            case "identity":
                calibrated_scores = scores
            case _:
                raise ValueError(f"Unsupported {self.model_method=}")

        return calibrated_scores

    def train_model_naive_ci(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            model_ = LogisticGAM(s(0, constraints="monotonic_inc"), lam=self.smooth_lambda)
            if self.adaptive_lambda:
                model_.gridsearch(X=x_train, y=y_train, lam=np.logspace(-3, 3, 11))
            else:
                model_.fit(X=x_train, y=y_train)
            self.model_ = model_
        except OptimizationError:
            return

    def train_model_monotonic_inc(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        models_ = []

        try:
            for _ in range(100):
                boot_sample_info = self.sample_info.sample(frac=1.0, replace=True)
                boot_x_train = boot_sample_info[["proxy_score"]].to_numpy()
                boot_y_train = boot_sample_info["oracle_results"].to_numpy()
                boot_model = LogisticGAM(s(0, constraints="monotonic_inc"), lam=self.smooth_lambda)
                boot_model.fit(X=boot_x_train, y=boot_y_train)
                models_.append(boot_model)

            model_ = LogisticGAM(s(0, constraints="monotonic_inc"), lam=self.smooth_lambda)
            model_.fit(X=x_train, y=y_train)

            self.model_ = model_
            self.models_ = models_
        except OptimizationError:
            return

    def train_model_custom(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        p = 20
        degree = 3
        log_odds_max = 5
        log_odds_min = -log_odds_max
        lo_train = np.clip(logit(x_train), log_odds_min, log_odds_max)

        self.spline_transformer_ = SplineTransformer(
            knots=np.linspace(log_odds_min, log_odds_max, p + 2).reshape((-1, 1)),
            include_bias=True,
            extrapolation="linear",
            degree=degree,
        )
        X = self.spline_transformer_.fit_transform(lo_train.reshape((-1, 1)))

        p1 = X.shape[1]
        mu_coef = np.linspace(
            log_odds_min * (p1 - 1) / (p1 - degree),
            log_odds_max * (p1 - 1) / (p1 - degree),
            p1,
        )
        self.model_ = RegularizedLogisticRegression(
            mu_param=mu_coef, lambda_param=self.smooth_lambda
        )
        self.model_.fit(X, y_train)

    def train_model_platt(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        y = y_train.astype(int)
        if len(np.unique(y)) < 2:
            return
        self.model_ = LogisticRegression(
            C=1.0 / max(self.smooth_lambda, 1e-8),
            solver="lbfgs",
            max_iter=1000,
        )
        self.model_.fit(x_train, y)

    def train_model_isotonic(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression

        y = y_train.astype(float)
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model_.fit(x_train.ravel(), y)

    def train_model(self) -> None:
        """Train or retrain the GAM calibration model on accumulated samples.

        Checks that sufficient samples exist per class before training. Uses different
        training procedures depending on the calibration method:
        - naive_ci: Trains a single monotonic GAM
        - monotonic_inc: Trains a GAM plus 100 bootstrap replicates for uncertainty

        Returns early without training if:
        - Insufficient samples per class (< min_train)
        - Optimization fails (OptimizationError)
        """
        # Check that we have enough data to train a model.
        x_train = self.sample_info[["proxy_score"]].to_numpy()
        y_train = self.sample_info["oracle_results"].to_numpy()

        match self.model_method:
            case "naive_ci":
                self.train_model_naive_ci(x_train=x_train, y_train=y_train)
            case "monotonic_inc":
                self.train_model_monotonic_inc(x_train=x_train, y_train=y_train)
            case "custom":
                self.train_model_custom(x_train=x_train, y_train=y_train)
            case "platt":
                self.train_model_platt(x_train=x_train, y_train=y_train)
            case "isotonic":
                self.train_model_isotonic(x_train=x_train, y_train=y_train)
            case "identity":
                pass
            case _:
                raise ValueError(f"Unsupported {self.model_method=}")

        self.last_trained = 2 ** int(np.log2(x_train.shape[0]))

    def update_thresholds(self, scores: pd.Series, alpha: float, beta: float):
        """Optimize and update decision thresholds based on calibrated scores.

        Uses numerical optimization to find threshold values that balance the trade-off
        between accuracy/F-score and computational cost. The optimization method is
        determined by the opt_method setting.

        Args:
            scores: Series of calibrated proxy scores.
            alpha: Weight for error vs. cost trade-off (0=cost only, 1=error only).
            beta: Weight parameter for F-beta score or error calculation.

        Raises:
            ValueError: If opt_method is not supported.
        """
        if self.model_ is None and self.model_method != "identity":
            return

        match self.opt_method:
            case "f1":
                estimate_bounds = _estimate_bounds_f1
            case "accuracy":
                estimate_bounds = _estimate_bounds_accuracy
            case _:
                raise ValueError(f"Unsupported {self.opt_method=}")

        self.low_threshold, self.high_threshold = estimate_bounds(
            scores=scores, alpha=alpha, beta=beta
        )

        # Calculate estimated delegation rate (proportion of scores in uncertain region)
        delegation_rate = scores.between(
            self.low_threshold, self.high_threshold, inclusive="left"
        ).mean()

        logger.info(
            f"After training on {self.last_trained} observations, "
            f"low={self.low_threshold:.4f}, high={self.high_threshold:.4f}, "
            f"estimated_delegation_rate={delegation_rate:.4f}"
        )

    def compute_ci_width(self, scores: np.ndarray) -> np.ndarray:
        """Compute GAM confidence interval width (68% CI) at given proxy scores.

        Returns zeros when no model is trained or the method doesn't support CIs.
        """
        if self.model_ is None:
            return np.zeros(len(scores))

        x_mat = np.asarray(scores, dtype=float).reshape((-1, 1))

        if self.model_method in ("naive_ci", "monotonic_inc"):
            ci = self.model_.confidence_intervals(x_mat, quantiles=[0.1587, 0.8413])
            return np.maximum(ci[:, 1] - ci[:, 0], 0.0)
        elif self.model_method == "custom":
            lo = np.clip(logit(np.clip(scores.astype(float), 1e-8, 1 - 1e-8)), -5, 5)
            x_spline = self.spline_transformer_.transform(lo.reshape((-1, 1)))
            q_arr = np.array([0.1587, 0.8413])
            ci = self.model_.confidence_intervals(X=x_spline, quantiles=q_arr)
            return np.maximum(ci[:, 1] - ci[:, 0], 0.0)
        else:
            return np.zeros(len(scores))

    def compute_fisher_info(self) -> None:
        """Cache the inverse Fisher information matrix from the trained GAM.

        For naive_ci: uses pygam's ``model_.statistics_['cov']``.
        For custom: uses ``model_.covariance_matrix_``.
        """
        if self.model_ is None:
            self.fisher_inv_ = None
            return

        if self.model_method in ("naive_ci", "monotonic_inc"):
            stats = getattr(self.model_, "statistics_", None)
            if stats is not None and "cov" in stats:
                self.fisher_inv_ = np.asarray(stats["cov"])
            else:
                self.fisher_inv_ = None
        elif self.model_method == "custom":
            self.fisher_inv_ = self.model_.covariance_matrix_
        else:
            self.fisher_inv_ = None

    def compute_fisher_info_value(self, scores: np.ndarray) -> np.ndarray:
        """Compute Fisher-based information value: phi(s)^T F^{-1} phi(s) * p(1-p).

        Returns zeros when the Fisher inverse is unavailable.
        """
        if self.model_ is None or not hasattr(self, "fisher_inv_") or self.fisher_inv_ is None:
            return np.zeros(len(scores))

        x_mat = np.asarray(scores, dtype=float).reshape((-1, 1))

        if self.model_method in ("naive_ci", "monotonic_inc"):
            phi = self.model_._modelmat(x_mat)
            if hasattr(phi, "toarray"):
                phi = phi.toarray()
            phi = np.asarray(phi, dtype=float)
            p = self.model_.predict_proba(x_mat)
        elif self.model_method == "custom":
            lo = np.clip(logit(np.clip(scores.astype(float), 1e-8, 1 - 1e-8)), -5, 5)
            phi = self.spline_transformer_.transform(lo.reshape((-1, 1)))
            phi = np.asarray(phi, dtype=float)
            p = self.model_.predict_proba(phi)
        else:
            return np.zeros(len(scores))

        p = np.clip(p, 1e-8, 1 - 1e-8)
        phi_Finv = phi @ self.fisher_inv_
        quadratic = np.sum(phi_Finv * phi, axis=1)
        return np.maximum(quadratic * p * (1 - p), 0.0)

    def compute_info_value(
        self, scores: np.ndarray, method: InfoMethodType = "ci_width"
    ) -> np.ndarray:
        """Compute information value using the specified method."""
        if method == "ci_width":
            return self.compute_ci_width(scores)
        elif method == "fisher":
            return self.compute_fisher_info_value(scores)
        else:
            raise ValueError(f"Unknown info method: {method}")

    def compute_thresholds_with_new_samples(
        self,
        oracle_results: pd.Series,
        proxy_score: pd.Series,
        sampling_result: SamplingResult,
    ) -> None:
        # Not used.
        raise NotImplementedError()


class GamCalDynamicCascade(ModelCascade):
    """Dynamic cascade system using GAM-based proxy score calibration.

    This cascade implements an adaptive decision system that balances accuracy and
    computational cost by:
    1. Using a cheap proxy model for initial scoring
    2. Calibrating proxy scores using Generalized Additive Models (GAMs) trained on
       oracle samples
    3. Dynamically optimizing decision thresholds to minimize oracle calls while
       maintaining target accuracy/F-score

    The system iteratively:
    - Samples low-confidence predictions for oracle evaluation
    - Trains/updates GAM calibration models
    - Optimizes decision thresholds
    - Makes high-confidence predictions without oracle calls

    Attributes:
        oracle_executor: Precomputed oracle labels/scores.
        proxy_executor: Precomputed proxy scores.
        smooth_lambda: Smoothing parameter for GAM regularization.
        alpha: Weight for error vs. cost trade-off (0=cost only, 1=error only).
        beta: Weight parameter for F-beta score (beta=1 for F1, beta>1 favors recall).
        batch_size: Number of samples to evaluate per oracle call.
        budget_percentage: Maximum fraction of samples that can be sent to oracle.
        cascade_output_column: Name for the output column in results DataFrame.
        sampling_method: Method for selecting samples to send to oracle.
        sampling_state: State manager for calibration and threshold optimization.
    """

    def __init__(
        self,
        oracle_executor: PrecomputedModelExecutor,
        proxy_executor: PrecomputedModelExecutor,
        smooth_lambda: float | None,
        model_method: ModelMethodType,
        opt_method: OptimizationMetricType,
        alpha: float,
        beta: float,
        batch_size: int,
        budget_percentage: float,
        cascade_output_column: str = "cascade_output",
        fixed_quantile: float | None = None,
        quantile_range: tuple[float, float] = (0.0, 1.0),
        quantile_distribution: str = "uniform",
        quantile_per_batch: bool = False,
        quantile_df: float = 3.0,
        expanded_sampling_fraction: float = 0.0,
        alpha_decay_gamma: float = 0.0,
        alpha_floor: float = 0.0,
        adaptive_lambda: bool = False,
        sampling_strategy: SamplingStrategyType = "uniform",
        info_lambda: float = 0.0,
        info_method: InfoMethodType = "ci_width",
        info_candidate_cap: int = 200,
        class_value_weight: float = 1.0,
    ):
        self.oracle_executor = oracle_executor
        self.proxy_executor = proxy_executor
        self.smooth_lambda = smooth_lambda
        self.model_method = model_method
        self.opt_method = opt_method
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.budget_percentage = budget_percentage
        self.cascade_output_column = cascade_output_column
        self.fixed_quantile = fixed_quantile
        self.quantile_range = quantile_range
        self.quantile_distribution = quantile_distribution
        self.quantile_per_batch = quantile_per_batch
        self.quantile_df = quantile_df
        self.expanded_sampling_fraction = expanded_sampling_fraction
        self.alpha_decay_gamma = alpha_decay_gamma
        self.alpha_floor = alpha_floor
        self.adaptive_lambda = adaptive_lambda
        self.sampling_strategy: SamplingStrategyType = sampling_strategy
        self.info_lambda = info_lambda
        self.info_method: InfoMethodType = info_method
        self.info_candidate_cap = info_candidate_cap
        self.class_value_weight = class_value_weight

        if smooth_lambda is None and not adaptive_lambda:
            raise NotImplementedError("Dynamic lambda exploration not currently implemented")

        self.sampling_state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=self.smooth_lambda if self.smooth_lambda is not None else 1.0,
            model_method=self.model_method,
            opt_method=self.opt_method,
            adaptive_lambda=self.adaptive_lambda,
        )

        if sampling_strategy in ("info_valued", "info_decay", "class_only"):
            effective_lambda = info_lambda if sampling_strategy != "class_only" else 0.0
            self.sampling_method: SamplingMethod = InfoValuedSamplingMethod(
                batch_size=self.batch_size,
                sampling_state=self.sampling_state,
                info_lambda=effective_lambda,
                info_method=self.info_method,
                class_value_weight=self.class_value_weight,
            )
        else:
            self.sampling_method = GamCalDynamicCascadeSamplingMethod(batch_size=self.batch_size)

    def sample_more_rows_for_oracle(
        self,
        rows: pd.DataFrame,
        proxy_scores: pd.Series,
        budget: int,
        low_confidence_selections: list[int],
    ) -> tuple[list[int], pd.Series]:
        """Sample more rows for the oracle based on the low confidence selections.

        Args:
            rows (pd.DataFrame): The rows to sample from.
            proxy_scores (pd.Series): The proxy scores to use for sampling.
            budget (int): The budget for sampling.
            low_confidence_selections (list[int]): The low confidence selections to sample from.

        Returns:
            tuple[list[int], pd.Series]: A tuple containing the sampled indices and the corresponding proxy scores.
        """

        sampling_result = self.sampling_method.compute_sample(
            input_selections=low_confidence_selections,
            proxy_score=proxy_scores,
            budget=budget,
        )
        sampling_selections = sampling_result.sample_selections

        oracle_results = self.oracle_executor.execute(rows, sampling_selections)
        oracle_results = oracle_results[self.oracle_executor.output_columns[0]]

        self.sampling_state.update_sample_info(
            oracle_results=oracle_results,
            proxy_score=proxy_scores,
            sampling_result=sampling_result,
        )

        return sampling_selections, oracle_results

    def _build_info_candidate_pool(
        self,
        input_selections: list[int],
        low_confidence_selections: list[int],
        cap: int,
    ) -> list[int]:
        """Build a merged candidate pool from uncertain and confident regions.

        Uses a 75/25 split (uncertain / confident) up to `cap` candidates.
        """
        lc_set = set(low_confidence_selections)
        uncertain = list(lc_set)
        confident = [i for i in input_selections if i not in lc_set]

        max_uncertain = int(cap * 0.75)
        max_confident = cap - max_uncertain

        if len(uncertain) > max_uncertain:
            uncertain = list(np.random.choice(uncertain, size=max_uncertain, replace=False))
        if len(confident) > max_confident:
            confident = list(np.random.choice(confident, size=max_confident, replace=False))

        pool = uncertain + confident
        return pool if pool else low_confidence_selections

    def execute(self, rows: pd.DataFrame) -> pd.DataFrame:
        # Initialize input selections.
        input_selections = list(range(len(rows)))

        # Generate raw proxy scores for all rows passed in.
        proxy_result = self.proxy_executor.execute(input_data=rows, selections=input_selections)
        proxy_scores = proxy_result.iloc[:, 0]

        # Initialize budget.
        budget = max(1, int(len(rows) * self.budget_percentage))

        # Initialize output results.
        results = pd.Series([None] * len(rows), dtype=object)

        if self.fixed_quantile is not None:
            q = pd.Series(np.full(len(rows), self.fixed_quantile))
        else:
            n = len(rows)
            draw_n = 1 if self.quantile_per_batch else n
            if self.quantile_distribution == "uniform":
                lo, hi = self.quantile_range
                raw = np.random.uniform(low=lo, high=hi, size=draw_n)
            elif self.quantile_distribution == "t":
                from scipy.stats import t as t_dist

                raw = t_dist.rvs(df=self.quantile_df, size=draw_n)
            elif self.quantile_distribution == "normal":
                raw = np.random.standard_normal(size=draw_n)
            elif self.quantile_distribution == "cauchy":
                raw = np.random.standard_cauchy(size=draw_n)
            else:
                raise ValueError(f"Unknown quantile_distribution: {self.quantile_distribution}")
            if self.quantile_per_batch:
                raw = np.full(n, raw[0])
            q = pd.Series(raw)

        # Initialize calibrated proxy scores.
        calibrated_proxy_scores = self.sampling_state.calibrate_proxy_scores(
            scores=proxy_scores, q=q
        )

        # Calculate low confidence selection.
        low_confidence_selections = self.compute_low_confidence_selections(
            current_selections=input_selections,
            proxy_scores=calibrated_proxy_scores,
            thresholds=self.sampling_state.get_high_confidence_thresholds(),
        )

        use_info_sampling = self.sampling_strategy in ("info_valued", "info_decay", "class_only")
        initial_budget = budget

        while budget > 0 and len(low_confidence_selections) > 0:
            # --- Lambda decay for info_decay strategy (budget-based) ---
            if self.sampling_strategy == "info_decay" and isinstance(
                self.sampling_method, InfoValuedSamplingMethod
            ):
                self.sampling_method.info_lambda = self.info_lambda * (
                    budget / max(1, initial_budget)
                )

            # --- Build candidate pool ---
            if use_info_sampling:
                candidate_pool = self._build_info_candidate_pool(
                    input_selections=input_selections,
                    low_confidence_selections=low_confidence_selections,
                    cap=self.info_candidate_cap,
                )
                expanded_budget = 0
            else:
                candidate_pool = low_confidence_selections
                if self.expanded_sampling_fraction > 0:
                    per_iter = min(budget, self.batch_size)
                    expanded_budget = max(1, int(per_iter * self.expanded_sampling_fraction))
                else:
                    expanded_budget = 0

            sampling_selections, oracle_results = self.sample_more_rows_for_oracle(
                rows=rows,
                proxy_scores=proxy_scores,
                budget=budget,
                low_confidence_selections=candidate_pool,
            )

            budget -= len(sampling_selections)

            input_selections = self.subtract_sample_from_selection(
                current_selections=input_selections,
                sampling_selections=sampling_selections,
            )

            self.populate_sample_results_from_oracle(
                results=results,
                oracle_results=oracle_results,
                sampling_selections=sampling_selections,
            )

            # Expanded uniform sampling (unused when info-valued strategies are active).
            if expanded_budget > 0 and budget > 0:
                lc_set = set(low_confidence_selections)
                sampled_set = set(sampling_selections)
                high_conf_remaining = [
                    i for i in input_selections if i not in lc_set and i not in sampled_set
                ]
                if len(high_conf_remaining) > 0:
                    n_expanded = min(expanded_budget, len(high_conf_remaining), budget)
                    expanded_indices = list(
                        np.random.choice(high_conf_remaining, size=n_expanded, replace=False)
                    )
                    exp_oracle = self.oracle_executor.execute(rows, expanded_indices)
                    exp_oracle = exp_oracle[self.oracle_executor.output_columns[0]]
                    exp_sr = SamplingResult(
                        sample_selections=expanded_indices, correction_factors=[]
                    )
                    self.sampling_state.update_sample_info(
                        oracle_results=exp_oracle,
                        proxy_score=proxy_scores,
                        sampling_result=exp_sr,
                    )
                    self.populate_sample_results_from_oracle(
                        results=results,
                        oracle_results=exp_oracle,
                        sampling_selections=expanded_indices,
                    )
                    budget -= n_expanded
                    input_selections = self.subtract_sample_from_selection(
                        current_selections=input_selections,
                        sampling_selections=expanded_indices,
                    )

            if self.sampling_state.check_if_train_model():
                self.sampling_state.train_model()
                calibrated_proxy_scores = self.sampling_state.calibrate_proxy_scores(
                    scores=proxy_scores, q=q
                )
                if self.alpha_decay_gamma > 0:
                    n_t = len(self.sampling_state.sample_info)
                    n_min = self.sampling_state.min_train
                    alpha_t = max(
                        self.alpha_floor,
                        self.alpha * (n_min / max(n_min, n_t)) ** self.alpha_decay_gamma,
                    )
                else:
                    alpha_t = self.alpha
                self.sampling_state.update_thresholds(
                    scores=calibrated_proxy_scores, alpha=alpha_t, beta=self.beta
                )
                if use_info_sampling and self.info_method == "fisher":
                    self.sampling_state.compute_fisher_info()

            # Update low confidence rows.
            low_confidence_selections = self.compute_low_confidence_selections(
                current_selections=input_selections,
                proxy_scores=calibrated_proxy_scores,
                thresholds=self.sampling_state.get_high_confidence_thresholds(),
            )

        # For high confidence rows, fill in 0 or 1 without calling the oracle.
        self.populate_high_confidence_results_from_proxy(
            results=results,
            proxy_results=calibrated_proxy_scores,
            thresholds=self.sampling_state.get_high_confidence_thresholds(),
            current_selections=input_selections,
            low_confidence_selections=low_confidence_selections,
        )

        # If there are still low confidence rows, evaluate them using a simple rule.
        if len(low_confidence_selections) > 0:
            self.populate_low_confidence_results_from_proxy(
                results=results,
                proxy_scores=calibrated_proxy_scores,
                threshold=self.sampling_state.compute_low_confidence_threshold_impl(),
                low_confidence_selections=low_confidence_selections,
            )

        return pd.concat([rows, results.rename(self.cascade_output_column).astype(bool)], axis=1)
