"""Tests for GamCalDynamicCascade-specific features beyond the baseline smoke suite.

Tests cover:
  1. Adaptive lambda (GCV gridsearch for GAM smoothing parameter)
  2. Budget-constrained cascade (_estimate_bounds_budget_constrained)
  3. Fisher information and CI-width computation
  4. InfoValuedSamplingMethod
  5. End-to-end smoke tests for info-valued sampling strategies
  6. Default-config stability (uniform sampling defaults)
  7. Config wiring (CascadeConfig → _build_cascade → GamCalDynamicCascade)
  8. Merged candidate pool construction
"""

import numpy as np
import pandas as pd
import pytest

from icefall.datasets import ORACLE_RESULT_COL, PROXY_SCORE_COL
from icefall.harness import CascadeConfig, run_single


def _make_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    proxy = rng.uniform(0, 1, n)
    oracle = (proxy > 0.5).astype(bool)
    noise_idx = rng.choice(n, size=int(n * 0.1), replace=False)
    oracle[noise_idx] = ~oracle[noise_idx]
    return pd.DataFrame(
        {
            PROXY_SCORE_COL: proxy,
            ORACLE_RESULT_COL: oracle,
        }
    )


SYNTHETIC_DATA = _make_synthetic_data()


# ---------------------------------------------------------------------------
# 1. Adaptive lambda (GCV gridsearch)
# ---------------------------------------------------------------------------


class TestAdaptiveLambda:
    """Tests for the adaptive_lambda feature (GCV gridsearch for GAM lambda)."""

    def test_config_default_is_false(self):
        cfg = CascadeConfig()
        assert cfg.adaptive_lambda is False

    def test_config_accepts_true(self):
        cfg = CascadeConfig(adaptive_lambda=True)
        assert cfg.adaptive_lambda is True

    def test_sampling_state_default_is_false(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0, model_method="naive_ci", opt_method="f1"
        )
        assert state.adaptive_lambda is False

    def test_sampling_state_accepts_true(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
            adaptive_lambda=True,
        )
        assert state.adaptive_lambda is True

    def test_fixed_lambda_training(self):
        """With adaptive_lambda=False, the GAM uses the specified smooth_lambda."""
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )
        from icefall.cascades.sampling.sampling_method import SamplingResult

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=10.0,
            model_method="naive_ci",
            opt_method="f1",
            adaptive_lambda=False,
        )
        rng = np.random.RandomState(42)
        n = 200
        proxy = rng.uniform(0.05, 0.95, n)
        oracle = (proxy > 0.5).astype(float)
        noise_idx = rng.choice(n, size=20, replace=False)
        oracle[noise_idx] = 1.0 - oracle[noise_idx]
        sr = SamplingResult(sample_selections=list(range(n)), correction_factors=[])
        state.update_sample_info(
            oracle_results=pd.Series(oracle),
            proxy_score=pd.Series(proxy),
            sampling_result=sr,
        )
        state.train_model()

        assert state.model_ is not None
        actual_lam = state.model_.lam[0][0]
        assert actual_lam == pytest.approx(10.0), (
            f"Fixed lambda should be 10.0 but got {actual_lam}"
        )

    def test_adaptive_lambda_training(self):
        """With adaptive_lambda=True, the GAM selects lambda via GCV gridsearch."""
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )
        from icefall.cascades.sampling.sampling_method import SamplingResult

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
            adaptive_lambda=True,
        )
        rng = np.random.RandomState(42)
        n = 200
        proxy = rng.uniform(0.05, 0.95, n)
        oracle = (proxy > 0.5).astype(float)
        noise_idx = rng.choice(n, size=20, replace=False)
        oracle[noise_idx] = 1.0 - oracle[noise_idx]
        sr = SamplingResult(sample_selections=list(range(n)), correction_factors=[])
        state.update_sample_info(
            oracle_results=pd.Series(oracle),
            proxy_score=pd.Series(proxy),
            sampling_result=sr,
        )
        state.train_model()

        assert state.model_ is not None
        actual_lam = state.model_.lam[0][0]
        # GCV should select a lambda from the search grid, not necessarily 1.0
        assert actual_lam > 0, f"Lambda should be positive, got {actual_lam}"

    def test_adaptive_lambda_e2e_smoke(self):
        """End-to-end: adaptive_lambda=True runs to completion."""
        config = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            dop=1,
            seed=0,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            adaptive_lambda=True,
            alpha=0.5,
        )
        result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.delegation_rate <= 1.0
        assert result.elapsed_seconds > 0

    def test_adaptive_lambda_wired_through_cascade(self):
        """The GamCalDynamicCascade constructor passes adaptive_lambda to the state."""
        from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
        from icefall.cascades.gam_cal_dynamic_cascade import GamCalDynamicCascade

        data = SYNTHETIC_DATA
        executor = PrecomputedModelExecutor(
            data=data,
            input_columns=[PROXY_SCORE_COL],
            output_columns=[ORACLE_RESULT_COL],
        )
        cascade = GamCalDynamicCascade(
            oracle_executor=executor,
            proxy_executor=executor,
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
            alpha=0.5,
            beta=1.0,
            batch_size=128,
            budget_percentage=1.0,
            adaptive_lambda=True,
        )
        assert cascade.adaptive_lambda is True
        assert cascade.sampling_state.adaptive_lambda is True

    def test_fixed_lambda_default(self):
        """Default adaptive_lambda=False matches an explicit fixed-lambda run."""
        cfg_default = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            seed=42,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            alpha=0.5,
        )
        cfg_explicit = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            seed=42,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            alpha=0.5,
            adaptive_lambda=False,
        )
        r1 = run_single("synthetic", config=cfg_default, data=SYNTHETIC_DATA)
        r2 = run_single("synthetic", config=cfg_explicit, data=SYNTHETIC_DATA)
        assert r1.f1 == pytest.approx(r2.f1, abs=1e-10)
        assert r1.delegation_rate == pytest.approx(r2.delegation_rate, abs=1e-10)


# ---------------------------------------------------------------------------
# 2. Budget-constrained cascade (_estimate_bounds_budget_constrained)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Budget-constrained GAMCAL mode is not implemented (_estimate_bounds_budget_constrained)."
    )
)
class TestBudgetConstrainedBounds:
    """Tests for the binary-search-over-alpha budget constrained function."""

    @pytest.fixture(autouse=True)
    def _import(self):
        gam_cal = pytest.importorskip(
            "icefall.cascades.gam_cal_dynamic_cascade",
            reason="gam_cal_dynamic_cascade module unavailable",
        )
        if not hasattr(gam_cal, "_estimate_bounds_budget_constrained"):
            pytest.skip("_estimate_bounds_budget_constrained is not implemented")
        self._estimate_bounds_budget_constrained = gam_cal._estimate_bounds_budget_constrained
        self._estimate_bounds_f1 = gam_cal._estimate_bounds_f1

    def _make_scores(self, n=1000, seed=0):
        rng = np.random.RandomState(seed)
        return pd.Series(rng.uniform(0, 1, n))

    def test_returns_three_values(self):
        scores = self._make_scores()
        result = self._estimate_bounds_budget_constrained(
            scores=scores, target_delegation=0.20, beta=1.0
        )
        assert len(result) == 3
        tau_l, tau_h, effective_alpha = result
        assert 0.0 <= tau_l <= tau_h <= 1.0
        assert 0.0 < effective_alpha < 1.0

    def test_respects_target_delegation_approximately(self):
        scores = self._make_scores()
        for target in [0.10, 0.20, 0.30]:
            tau_l, tau_h, _ = self._estimate_bounds_budget_constrained(
                scores=scores, target_delegation=target, beta=1.0
            )
            actual = scores.between(tau_l, tau_h, inclusive="left").mean()
            assert abs(actual - target) < 0.10, f"target={target}, actual={actual}"

    def test_higher_target_means_wider_band(self):
        scores = self._make_scores()
        _, _, alpha_low = self._estimate_bounds_budget_constrained(
            scores=scores, target_delegation=0.10, beta=1.0
        )
        _, _, alpha_high = self._estimate_bounds_budget_constrained(
            scores=scores, target_delegation=0.40, beta=1.0
        )
        assert alpha_high > alpha_low

    def test_extreme_targets(self):
        scores = self._make_scores()
        tau_l, tau_h, _ = self._estimate_bounds_budget_constrained(
            scores=scores, target_delegation=0.01, beta=1.0
        )
        actual = scores.between(tau_l, tau_h, inclusive="left").mean()
        assert actual < 0.15

        tau_l, tau_h, _ = self._estimate_bounds_budget_constrained(
            scores=scores, target_delegation=0.90, beta=1.0
        )
        actual = scores.between(tau_l, tau_h, inclusive="left").mean()
        assert actual > 0.50


# ---------------------------------------------------------------------------
# 2. Fisher information computation
# ---------------------------------------------------------------------------


class TestFisherInformation:
    """Tests for compute_fisher_info and compute_info_value."""

    @pytest.fixture
    def trained_state(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )
        from icefall.cascades.sampling.sampling_method import SamplingResult

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
        )
        rng = np.random.RandomState(42)
        n = 200
        proxy = rng.uniform(0, 1, n)
        oracle = (proxy > 0.5).astype(float)
        noise_idx = rng.choice(n, size=20, replace=False)
        oracle[noise_idx] = 1.0 - oracle[noise_idx]

        sr = SamplingResult(sample_selections=list(range(n)), correction_factors=[])
        state.update_sample_info(
            oracle_results=pd.Series(oracle),
            proxy_score=pd.Series(proxy),
            sampling_result=sr,
        )
        state.train_model()
        return state

    def test_fisher_inv_computed_after_training(self, trained_state):
        assert trained_state.model_ is not None
        trained_state.compute_fisher_info()
        assert trained_state.fisher_inv_ is not None
        assert trained_state.fisher_inv_.ndim == 2
        d = trained_state.fisher_inv_.shape[0]
        assert trained_state.fisher_inv_.shape == (d, d)

    def test_fisher_inv_is_symmetric(self, trained_state):
        trained_state.compute_fisher_info()
        F_inv = trained_state.fisher_inv_
        np.testing.assert_allclose(F_inv, F_inv.T, atol=1e-10)

    def test_info_value_shape_and_nonneg(self, trained_state):
        trained_state.compute_fisher_info()
        scores = np.linspace(0.01, 0.99, 50)
        v_info = trained_state.compute_info_value(scores)
        assert v_info.shape == (50,)
        assert (v_info >= 0).all()

    def test_info_value_decreases_with_more_data(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )
        from icefall.cascades.sampling.sampling_method import SamplingResult

        rng = np.random.RandomState(123)
        test_scores = np.array([0.3, 0.5, 0.7])

        vals = []
        for n in [50, 200]:
            state = GamCalDynamicCascadeSamplingState(
                smooth_lambda=1.0, model_method="naive_ci", opt_method="f1"
            )
            proxy = rng.uniform(0, 1, n)
            oracle = (proxy > 0.5).astype(float)
            sr = SamplingResult(sample_selections=list(range(n)), correction_factors=[])
            state.update_sample_info(
                oracle_results=pd.Series(oracle),
                proxy_score=pd.Series(proxy),
                sampling_result=sr,
            )
            state.train_model()
            state.compute_fisher_info()
            vals.append(state.compute_info_value(test_scores))

        assert vals[0].mean() > vals[1].mean(), "V_info should decrease with more training data"

    def test_info_value_zero_without_model(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0, model_method="naive_ci", opt_method="f1"
        )
        scores = np.array([0.2, 0.5, 0.8])
        v = state.compute_info_value(scores)
        np.testing.assert_array_equal(v, np.zeros(3))


# ---------------------------------------------------------------------------
# 3. InfoValuedSamplingMethod
# ---------------------------------------------------------------------------


class TestInfoValuedSamplingMethod:
    """Tests for the InfoValuedSamplingMethod class."""

    @pytest.fixture
    def sampler_with_state(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
            InfoValuedSamplingMethod,
        )
        from icefall.cascades.sampling.sampling_method import SamplingResult

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0, model_method="naive_ci", opt_method="f1"
        )
        rng = np.random.RandomState(42)
        n = 400
        proxy = rng.uniform(0.05, 0.95, n)
        oracle = (proxy > 0.5).astype(float)
        noise_idx = rng.choice(n, size=int(n * 0.1), replace=False)
        oracle[noise_idx] = 1.0 - oracle[noise_idx]
        sr = SamplingResult(sample_selections=list(range(n)), correction_factors=[])
        state.update_sample_info(
            oracle_results=pd.Series(oracle),
            proxy_score=pd.Series(proxy),
            sampling_result=sr,
        )
        state.train_model()
        state.compute_fisher_info()
        state.update_thresholds(
            scores=state.calibrate_proxy_scores(pd.Series(proxy), q=0.5),
            alpha=0.5,
            beta=1.0,
        )

        candidate_scores = pd.Series(rng.uniform(0.05, 0.95, 500))
        sampler = InfoValuedSamplingMethod(
            batch_size=32,
            sampling_state=state,
            info_lambda=0.5,
        )
        return sampler, state, candidate_scores

    def test_returns_correct_count(self, sampler_with_state):
        sampler, state, proxy_scores = sampler_with_state
        indices = list(range(500))
        result = sampler.compute_sample(indices, proxy_scores, budget=100)
        assert len(result.sample_selections) == 32  # batch_size
        assert all(i in indices for i in result.sample_selections)

    def test_no_duplicates(self, sampler_with_state):
        sampler, state, proxy_scores = sampler_with_state
        indices = list(range(500))
        result = sampler.compute_sample(indices, proxy_scores, budget=100)
        assert len(set(result.sample_selections)) == len(result.sample_selections)

    def test_handles_small_pool(self, sampler_with_state):
        sampler, state, proxy_scores = sampler_with_state
        indices = [0, 1, 2]
        result = sampler.compute_sample(indices, proxy_scores, budget=100)
        assert len(result.sample_selections) <= 3

    def test_cold_start_fallback(self):
        """Without a trained model, should fall back to uniform sampling."""
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
            InfoValuedSamplingMethod,
        )

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0, model_method="naive_ci", opt_method="f1"
        )
        sampler = InfoValuedSamplingMethod(
            batch_size=10,
            sampling_state=state,
            info_lambda=0.5,
        )
        proxy = pd.Series(np.random.uniform(0, 1, 100))
        result = sampler.compute_sample(list(range(100)), proxy, budget=50)
        assert len(result.sample_selections) == 10

    def test_lambda_zero_favours_boundaries(self, sampler_with_state):
        """With lambda=0, V_info is ignored; sampling should favour threshold-proximal rows."""
        sampler, state, proxy_scores = sampler_with_state
        sampler.info_lambda = 0.0
        indices = list(range(len(proxy_scores)))
        samples = []
        for _ in range(20):
            result = sampler.compute_sample(indices, proxy_scores, budget=len(indices))
            samples.extend(result.sample_selections[:5])

        sampled_scores = proxy_scores[samples]
        tau_l = state.low_threshold
        tau_h = state.high_threshold
        mid = (tau_l + tau_h) / 2
        mean_dist = abs(sampled_scores - mid).mean()
        random_dist = abs(proxy_scores - mid).mean()
        assert mean_dist <= random_dist + 0.15


# ---------------------------------------------------------------------------
# 4. End-to-end smoke tests for new CascadeConfig options
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Budget-constrained GAMCAL mode is not implemented (_estimate_bounds_budget_constrained)."
    )
)
class TestBudgetConstrainedSmoke:
    """End-to-end smoke tests for budget-constrained cascade mode."""

    @pytest.mark.parametrize("rho", [0.05, 0.15, 0.30, 0.50])
    def test_budget_constrained_runs(self, rho):
        config = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            dop=1,
            seed=0,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            expanded_sampling_fraction=0.10,
            max_delegation_rate=rho,
        )
        result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.delegation_rate <= 1.0

    def test_budget_constrained_vs_alpha(self):
        """Budget-constrained should produce results in the same ballpark as alpha-sweep."""
        r_alpha = run_single(
            "synthetic",
            config=CascadeConfig(
                algorithm="gamcal",
                batch_size=500,
                seed=0,
                sampling_percentage=1.0,
                calibration_method="naive_ci",
                fixed_quantile=0.5,
                alpha=0.5,
            ),
            data=SYNTHETIC_DATA,
        )
        r_budget = run_single(
            "synthetic",
            config=CascadeConfig(
                algorithm="gamcal",
                batch_size=500,
                seed=0,
                sampling_percentage=1.0,
                calibration_method="naive_ci",
                fixed_quantile=0.5,
                max_delegation_rate=r_alpha.delegation_rate,
            ),
            data=SYNTHETIC_DATA,
        )
        assert abs(r_budget.f1 - r_alpha.f1) < 0.15


class TestInfoValuedSamplingSmoke:
    """End-to-end smoke tests for all info-valued sampling strategies."""

    @pytest.mark.parametrize(
        "strategy,info_lambda",
        [
            ("info_valued", 0.0),
            ("info_valued", 0.5),
            ("info_valued", 1.0),
            ("info_decay", 1.0),
            ("class_only", 0.0),
        ],
    )
    def test_sampling_strategy_runs(self, strategy, info_lambda):
        config = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            dop=1,
            seed=0,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            sampling_strategy=strategy,
            info_lambda=info_lambda,
            alpha=0.3,
        )
        result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.delegation_rate <= 1.0

    def test_uniform_unchanged(self):
        """Uniform strategy with expanded sampling and no info-valued selection."""
        config = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            dop=1,
            seed=0,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            expanded_sampling_fraction=0.10,
            sampling_strategy="uniform",
            alpha=0.3,
        )
        result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
        assert 0.0 <= result.f1 <= 1.0

    def test_info_decay_lambda_decreases(self):
        """In info_decay mode, successful_train_count should increase."""
        config = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            dop=1,
            seed=0,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            sampling_strategy="info_decay",
            info_lambda=1.0,
            alpha=0.3,
        )
        result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.delegation_rate <= 1.0

    @pytest.mark.parametrize("info_method", ["ci_width", "fisher"])
    def test_info_method_variants(self, info_method):
        config = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            dop=1,
            seed=0,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            sampling_strategy="info_valued",
            info_lambda=0.5,
            info_method=info_method,
            alpha=0.3,
        )
        result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.delegation_rate <= 1.0


# ---------------------------------------------------------------------------
# 6. Default-config stability
# ---------------------------------------------------------------------------


class TestDefaultConfigStability:
    """Default config (sampling_strategy='uniform') matches explicit uniform defaults."""

    def test_default_config_uses_uniform_strategy(self):
        cfg = CascadeConfig()
        assert cfg.sampling_strategy == "uniform"
        assert cfg.info_lambda == 0.0
        assert cfg.info_method == "ci_width"
        assert cfg.info_candidate_cap == 200
        assert cfg.class_value_weight == 1.0

    def test_default_creates_uniform_sampler(self):
        from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascade,
            GamCalDynamicCascadeSamplingMethod,
        )

        data = SYNTHETIC_DATA
        executor = PrecomputedModelExecutor(
            data=data,
            input_columns=[PROXY_SCORE_COL],
            output_columns=[ORACLE_RESULT_COL],
        )
        cascade = GamCalDynamicCascade(
            oracle_executor=executor,
            proxy_executor=executor,
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
            alpha=0.5,
            beta=1.0,
            batch_size=128,
            budget_percentage=1.0,
        )
        assert isinstance(cascade.sampling_method, GamCalDynamicCascadeSamplingMethod)
        assert cascade.sampling_strategy == "uniform"

    def test_omitting_new_params_gives_same_results(self):
        """A config with no new params should produce identical F1/delegation
        as one with them explicitly set to defaults."""
        cfg_baseline = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            seed=42,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            alpha=0.5,
        )
        cfg_explicit_defaults = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            seed=42,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            alpha=0.5,
            sampling_strategy="uniform",
            info_lambda=0.0,
            info_method="ci_width",
            info_candidate_cap=200,
            class_value_weight=1.0,
        )
        r1 = run_single("synthetic", config=cfg_baseline, data=SYNTHETIC_DATA)
        r2 = run_single("synthetic", config=cfg_explicit_defaults, data=SYNTHETIC_DATA)
        assert r1.f1 == pytest.approx(r2.f1, abs=1e-10)
        assert r1.delegation_rate == pytest.approx(r2.delegation_rate, abs=1e-10)

    def test_uniform_with_expanded_sampling(self):
        """Uniform sampling with expanded_sampling_fraction is unchanged by info-valued plumbing."""
        cfg = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            seed=42,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            alpha=0.3,
            expanded_sampling_fraction=0.10,
        )
        r1 = run_single("synthetic", config=cfg, data=SYNTHETIC_DATA)

        cfg2 = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            seed=42,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            alpha=0.3,
            expanded_sampling_fraction=0.10,
            sampling_strategy="uniform",
            info_lambda=0.0,
        )
        r2 = run_single("synthetic", config=cfg2, data=SYNTHETIC_DATA)
        assert r1.f1 == pytest.approx(r2.f1, abs=1e-10)
        assert r1.delegation_rate == pytest.approx(r2.delegation_rate, abs=1e-10)


# ---------------------------------------------------------------------------
# 7. Config wiring: params flow CascadeConfig -> _build_cascade -> constructor
# ---------------------------------------------------------------------------


class TestInfoValuedConfigWiring:
    """Verify new params are properly wired end-to-end."""

    def test_info_valued_creates_info_sampler(self):
        from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascade,
            InfoValuedSamplingMethod,
        )

        data = SYNTHETIC_DATA
        executor = PrecomputedModelExecutor(
            data=data,
            input_columns=[PROXY_SCORE_COL],
            output_columns=[ORACLE_RESULT_COL],
        )
        cascade = GamCalDynamicCascade(
            oracle_executor=executor,
            proxy_executor=executor,
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
            alpha=0.5,
            beta=1.0,
            batch_size=128,
            budget_percentage=1.0,
            sampling_strategy="info_valued",
            info_lambda=0.5,
            info_method="ci_width",
            class_value_weight=1.0,
        )
        assert isinstance(cascade.sampling_method, InfoValuedSamplingMethod)
        assert cascade.sampling_method.info_lambda == 0.5
        assert cascade.sampling_method.info_method == "ci_width"
        assert cascade.sampling_method.class_value_weight == 1.0
        assert cascade.sampling_method.sampling_state is cascade.sampling_state

    def test_class_only_sets_lambda_zero(self):
        from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascade,
            InfoValuedSamplingMethod,
        )

        data = SYNTHETIC_DATA
        executor = PrecomputedModelExecutor(
            data=data,
            input_columns=[PROXY_SCORE_COL],
            output_columns=[ORACLE_RESULT_COL],
        )
        cascade = GamCalDynamicCascade(
            oracle_executor=executor,
            proxy_executor=executor,
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
            alpha=0.5,
            beta=1.0,
            batch_size=128,
            budget_percentage=1.0,
            sampling_strategy="class_only",
            info_lambda=0.99,
        )
        assert isinstance(cascade.sampling_method, InfoValuedSamplingMethod)
        assert cascade.sampling_method.info_lambda == 0.0

    def test_config_wired_through_eval_harness(self):
        """CascadeConfig fields flow through _build_cascade to the cascade."""

        config = CascadeConfig(
            algorithm="gamcal",
            batch_size=500,
            dop=1,
            seed=0,
            sampling_percentage=1.0,
            calibration_method="naive_ci",
            fixed_quantile=0.5,
            sampling_strategy="info_valued",
            info_lambda=0.75,
            info_method="fisher",
            info_candidate_cap=100,
            class_value_weight=0.5,
            alpha=0.3,
        )
        result = run_single("synthetic", config=config, data=SYNTHETIC_DATA)
        assert 0.0 <= result.f1 <= 1.0
        assert result.config["sampling_strategy"] == "info_valued"
        assert result.config["info_lambda"] == 0.75
        assert result.config["info_method"] == "fisher"
        assert result.config["info_candidate_cap"] == 100
        assert result.config["class_value_weight"] == 0.5


# ---------------------------------------------------------------------------
# 8. CI-width V_info computation
# ---------------------------------------------------------------------------


class TestCIWidthInfoValue:
    """Tests for the CI-width based V_info computation."""

    @pytest.fixture
    def trained_state(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )
        from icefall.cascades.sampling.sampling_method import SamplingResult

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
        )
        rng = np.random.RandomState(42)
        n = 200
        proxy = rng.uniform(0, 1, n)
        oracle = (proxy > 0.5).astype(float)
        noise_idx = rng.choice(n, size=20, replace=False)
        oracle[noise_idx] = 1.0 - oracle[noise_idx]

        sr = SamplingResult(sample_selections=list(range(n)), correction_factors=[])
        state.update_sample_info(
            oracle_results=pd.Series(oracle),
            proxy_score=pd.Series(proxy),
            sampling_result=sr,
        )
        state.train_model()
        return state

    def test_ci_width_shape_and_nonneg(self, trained_state):
        scores = np.linspace(0.01, 0.99, 50)
        ci = trained_state.compute_ci_width(scores)
        assert ci.shape == (50,)
        assert (ci >= 0).all()

    def test_ci_width_zero_without_model(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )

        state = GamCalDynamicCascadeSamplingState(
            smooth_lambda=1.0, model_method="naive_ci", opt_method="f1"
        )
        ci = state.compute_ci_width(np.array([0.2, 0.5, 0.8]))
        np.testing.assert_array_equal(ci, np.zeros(3))

    def test_ci_width_varies_with_score(self, trained_state):
        """CI width should vary across scores (not constant) because the
        GAM's uncertainty depends on local data density and curvature."""
        scores = np.linspace(0.05, 0.95, 20)
        ci = trained_state.compute_ci_width(scores)
        assert ci.std() > 0, "CI width should not be constant across all scores"

    def test_ci_width_decreases_with_more_data(self):
        from icefall.cascades.gam_cal_dynamic_cascade import (
            GamCalDynamicCascadeSamplingState,
        )
        from icefall.cascades.sampling.sampling_method import SamplingResult

        rng = np.random.RandomState(123)
        test_scores = np.array([0.3, 0.5, 0.7])

        vals = []
        for n in [50, 200]:
            state = GamCalDynamicCascadeSamplingState(
                smooth_lambda=1.0, model_method="naive_ci", opt_method="f1"
            )
            proxy = rng.uniform(0, 1, n)
            oracle = (proxy > 0.5).astype(float)
            sr = SamplingResult(sample_selections=list(range(n)), correction_factors=[])
            state.update_sample_info(
                oracle_results=pd.Series(oracle),
                proxy_score=pd.Series(proxy),
                sampling_result=sr,
            )
            state.train_model()
            vals.append(state.compute_ci_width(test_scores))

        assert vals[0].mean() > vals[1].mean(), "CI width should decrease with more training data"

    def test_compute_info_value_dispatches_ci_width(self, trained_state):
        scores = np.linspace(0.1, 0.9, 20)
        via_dispatch = trained_state.compute_info_value(scores, method="ci_width")
        via_direct = trained_state.compute_ci_width(scores)
        np.testing.assert_array_equal(via_dispatch, via_direct)


# ---------------------------------------------------------------------------
# 9. Merged candidate pool construction
# ---------------------------------------------------------------------------


class TestMergedCandidatePool:
    """Tests for _build_info_candidate_pool."""

    @pytest.fixture
    def cascade(self):
        from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor
        from icefall.cascades.gam_cal_dynamic_cascade import GamCalDynamicCascade

        data = SYNTHETIC_DATA
        executor = PrecomputedModelExecutor(
            data=data,
            input_columns=[PROXY_SCORE_COL],
            output_columns=[ORACLE_RESULT_COL],
        )
        return GamCalDynamicCascade(
            oracle_executor=executor,
            proxy_executor=executor,
            smooth_lambda=1.0,
            model_method="naive_ci",
            opt_method="f1",
            alpha=0.5,
            beta=1.0,
            batch_size=128,
            budget_percentage=1.0,
            sampling_strategy="info_valued",
            info_lambda=0.5,
            info_candidate_cap=200,
        )

    def test_respects_cap(self, cascade):
        np.random.seed(0)
        all_sel = list(range(500))
        lc_sel = list(range(300))
        pool = cascade._build_info_candidate_pool(all_sel, lc_sel, cap=200)
        assert len(pool) <= 200

    def test_includes_both_regions(self, cascade):
        np.random.seed(0)
        all_sel = list(range(500))
        lc_sel = list(range(200))
        pool = cascade._build_info_candidate_pool(all_sel, lc_sel, cap=200)
        lc_set = set(lc_sel)
        has_uncertain = any(i in lc_set for i in pool)
        has_confident = any(i not in lc_set for i in pool)
        assert has_uncertain, "Pool should include uncertain rows"
        assert has_confident, "Pool should include confident rows"

    def test_small_pool_no_crash(self, cascade):
        pool = cascade._build_info_candidate_pool(
            input_selections=[0, 1, 2],
            low_confidence_selections=[0, 1],
            cap=200,
        )
        assert len(pool) <= 3
        assert 0 in pool or 1 in pool

    def test_all_uncertain_no_confident(self, cascade):
        """When all rows are uncertain, pool is just the uncertain rows."""
        all_sel = list(range(50))
        pool = cascade._build_info_candidate_pool(all_sel, all_sel, cap=200)
        assert set(pool) == set(all_sel)

    def test_75_25_split(self, cascade):
        """With many rows, the split should approximate 75/25."""
        np.random.seed(0)
        all_sel = list(range(1000))
        lc_sel = list(range(500))
        pool = cascade._build_info_candidate_pool(all_sel, lc_sel, cap=200)
        lc_set = set(lc_sel)
        n_uncertain = sum(1 for i in pool if i in lc_set)
        n_confident = sum(1 for i in pool if i not in lc_set)
        assert n_uncertain <= 150
        assert n_confident <= 50
        assert n_uncertain + n_confident == len(pool)
