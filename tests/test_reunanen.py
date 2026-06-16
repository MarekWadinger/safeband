"""Tests for the Reunanen et al. (2020) online autoencoder detector."""

import math

import numpy as np
import pandas as pd

from functions.evaluate import progressive_val_predict
from functions.reunanen import ReunanenScorer

EPS = 1e-6
GRAD_ATOL = 1e-7


def make_calibrated_scorer(
    n_points: int = 100,
    M: int = 200,
    seed: int = 42,
) -> ReunanenScorer:
    """Build a scorer calibrated on a noiseless two-feature sine stream."""
    scorer = ReunanenScorer(n_hidden=2, M=M, seed=seed)
    for i in range(n_points):
        scorer.learn_one({"a": math.sin(i / 5), "b": math.cos(i / 5)})
    assert scorer.calibrated
    return scorer


def numerical_gradient(
    scorer: ReunanenScorer,
    v_scaled: np.ndarray,
    param_name: str,
) -> np.ndarray:
    """Differentiate the L1 cost w.r.t. a parameter by central differences."""
    param = getattr(scorer, param_name)
    grad = np.zeros_like(param)
    iterator = np.nditer(param, flags=["multi_index"])
    for _ in iterator:
        idx = iterator.multi_index
        original = param[idx]
        param[idx] = original + EPS
        _, z = scorer._forward(v_scaled)
        cost_plus = scorer._cost(v_scaled, z)
        param[idx] = original - EPS
        _, z = scorer._forward(v_scaled)
        cost_minus = scorer._cost(v_scaled, z)
        param[idx] = original
        grad[idx] = (cost_plus - cost_minus) / (2 * EPS)
    return grad


class TestGradients:
    """The analytic L1 backprop matches numerical finite differences."""

    def test_gradients_match_finite_differences(self) -> None:
        """All three parameter gradients agree with central differences."""
        scorer = ReunanenScorer(n_hidden=3, seed=1)
        scorer._initialize({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0})
        rng = np.random.default_rng(7)
        v_scaled = rng.uniform(0.1, 0.9, 4)

        grad_w, grad_b, grad_b_z = scorer._gradients(v_scaled)

        for name, analytic in [
            ("_w", grad_w),
            ("_b", grad_b),
            ("_b_z", grad_b_z),
        ]:
            numeric = numerical_gradient(scorer, v_scaled, name)
            assert np.allclose(analytic, numeric, atol=GRAD_ATOL), name

    def test_sgd_step_reduces_cost(self) -> None:
        """Repeated SGD steps on one point reduce its reconstruction cost."""
        scorer = ReunanenScorer(n_hidden=2, seed=3)
        scorer._initialize({"a": 0.0, "b": 0.0})
        v_scaled = np.array([0.2, 0.8])
        _, z = scorer._forward(v_scaled)
        cost_before = scorer._cost(v_scaled, z)
        for _ in range(20):
            scorer._sgd_step(v_scaled)
        _, z = scorer._forward(v_scaled)
        assert scorer._cost(v_scaled, z) < cost_before


class TestCalibrationStateMachine:
    """Algorithm 1: limits phase, training phase, then detection."""

    def test_starts_in_limits_phase_and_predicts_false(self) -> None:
        """A fresh scorer is uncalibrated and never flags points."""
        scorer = ReunanenScorer(seed=0)
        assert scorer._phase == "limits"
        assert not scorer.calibrated
        assert scorer.predict_one({"a": 1e9, "b": -1e9}) is False

    def test_constant_stream_stays_in_limits_phase(self) -> None:
        """Without feature variance the scaling limits never spread."""
        scorer = ReunanenScorer(seed=0)
        for _ in range(10):
            scorer.learn_one({"a": 1.0, "b": 2.0})
        assert scorer._phase == "limits"

    def test_transitions_to_training_once_limits_spread(self) -> None:
        """Phase 2 starts as soon as every feature has min < max."""
        scorer = ReunanenScorer(seed=0)
        scorer.learn_one({"a": 1.0, "b": 2.0})
        assert scorer._phase == "limits"
        scorer.learn_one({"a": 2.0, "b": 1.0})
        assert scorer._phase == "train"
        assert not scorer.calibrated

    def test_patience_ends_training_within_budget(self) -> None:
        """Training stops via the patience heuristic and within M points."""
        scorer = ReunanenScorer(n_hidden=2, M=100, seed=0)
        for i in range(100):
            scorer.learn_one({"a": math.sin(i / 5), "b": math.cos(i / 5)})
            if scorer.calibrated:
                break
        assert scorer.calibrated
        assert scorer._x_old is not None

    def test_budget_exhaustion_forces_detection_phase(self) -> None:
        """A degenerate stream exhausting M still ends calibration."""
        scorer = ReunanenScorer(M=5, seed=0)
        for _ in range(5):
            scorer.learn_one({"a": 1.0, "b": 2.0})
        assert scorer.calibrated

    def test_improvement_resets_patience(self) -> None:
        """A cost decrease larger than decmin resets the patience counter."""
        scorer = ReunanenScorer(n_hidden=2, M=10_000, seed=0)
        scorer.learn_one({"a": 0.0, "b": 1.0})
        scorer.learn_one({"a": 1.0, "b": 0.0})
        assert scorer._phase == "train"
        scorer.learn_one({"a": 0.5, "b": 0.5})
        # The first training point always improves on r_min = inf.
        assert scorer._patience == scorer._patience_reset
        assert scorer._r_min < float("inf")


class TestScalingLimits:
    """Eq. 15: limits update online but are frozen on detected outliers."""

    def test_limits_frozen_on_detected_outlier(self) -> None:
        """A point flagged as outlier must not move the scaling limits."""
        scorer = make_calibrated_scorer()
        # Force the outlier branch: with mu = sigma = 0 any positive
        # reconstruction cost exceeds the threshold.
        scorer._mu = 0.0
        scorer._s = 0.0
        x_min = scorer._x_min.copy()
        x_max = scorer._x_max.copy()
        scorer.learn_one({"a": 5.0, "b": -5.0})
        assert np.array_equal(scorer._x_min, x_min)
        assert np.array_equal(scorer._x_max, x_max)

    def test_limits_updated_on_inlier(self) -> None:
        """A point below the threshold extends the scaling limits."""
        scorer = make_calibrated_scorer()
        # Force the inlier branch: a huge mean makes nothing an outlier.
        scorer._mu = 1e6
        scorer._s = 0.0
        scorer.learn_one({"a": 6.0, "b": -6.0})
        assert scorer._x_max[0] == 6.0
        assert scorer._x_min[1] == -6.0


class TestIdenticalPointSkip:
    """Algorithm 2 lines 3-4: identical consecutive points are skipped."""

    def test_repeated_point_leaves_state_unchanged(self) -> None:
        """The second occurrence triggers no SGD, EWMA, or limit update."""
        scorer = make_calibrated_scorer()
        x = {"a": 0.3, "b": 0.7}
        scorer.learn_one(x)
        w = scorer._w.copy()
        b = scorer._b.copy()
        b_z = scorer._b_z.copy()
        mu, s = scorer._mu, scorer._s
        x_min = scorer._x_min.copy()
        x_max = scorer._x_max.copy()

        scorer.learn_one(x)

        assert np.array_equal(scorer._w, w)
        assert np.array_equal(scorer._b, b)
        assert np.array_equal(scorer._b_z, b_z)
        assert scorer._mu == mu
        assert scorer._s == s
        assert np.array_equal(scorer._x_min, x_min)
        assert np.array_equal(scorer._x_max, x_max)

    def test_distinct_point_updates_state(self) -> None:
        """A non-identical point does update the EWMA statistics."""
        scorer = make_calibrated_scorer()
        scorer.learn_one({"a": 0.3, "b": 0.7})
        mu = scorer._mu
        scorer.learn_one({"a": 0.4, "b": 0.6})
        assert scorer._mu != mu


class TestSpikeDetection:
    """The EWMA threshold flags an injected spike on a synthetic stream."""

    def test_spike_flagged_normal_not(self) -> None:
        """After learning a noisy sine stream, only the spike is flagged."""
        rng = np.random.default_rng(0)
        scorer = ReunanenScorer(n_hidden=2, seed=0)
        x = None
        for i in range(400):
            x = {
                "a": math.sin(i / 10) + rng.normal(0.0, 0.05),
                "b": math.cos(i / 10) + rng.normal(0.0, 0.05),
            }
            scorer.learn_one(x)
        assert scorer.calibrated
        assert x is not None
        spike = {"a": x["a"] + 10.0, "b": x["b"] - 10.0}
        assert scorer.predict_one(spike) is True
        normal = {"a": math.sin(40.0), "b": math.cos(40.0)}
        assert scorer.predict_one(normal) is False

    def test_score_one_is_normalised_and_stateless(self) -> None:
        """Scores are cost / d and repeated calls do not change state."""
        scorer = make_calibrated_scorer()
        x = {"a": 0.1, "b": 0.9}
        first = scorer.score_one(x)
        assert 0.0 <= first <= 1.0
        assert scorer.score_one(x) == first


class TestProgressiveValPredict:
    """The scorer plugs into the repository's prequential harness."""

    def test_runs_inside_progressive_val_predict(self) -> None:
        """One pass over a synthetic DataFrame flags the injected spike."""
        rng = np.random.default_rng(3)
        n = 200
        spike = 150
        idx = np.arange(n)
        x_values = np.sin(idx / 10) + rng.normal(0.0, 0.05, n)
        x_values[spike] += 10.0
        dataset = pd.DataFrame(
            {
                "x": x_values,
                "y": np.cos(idx / 10) + rng.normal(0.0, 0.05, n),
            },
        )
        scorer = ReunanenScorer(n_hidden=2, M=500, seed=0)

        y_pred, meta = progressive_val_predict(
            scorer,
            dataset,
            print_final=False,
        )

        assert len(y_pred) == n
        assert all(isinstance(p, bool) for p in y_pred)
        assert y_pred[spike] is True
        assert meta == {}


class TestPatienceFloor:
    """Calibration patience must never drop below 1 on wide inputs."""

    def test_wide_input_keeps_patience_at_least_one(self) -> None:
        """A high-dimensional input cannot drive M*decmin/d below 1."""
        # M*decmin = 5 here; with d=40 features the raw value 5/40 = 0.125
        # would end calibration after a single point. The floor keeps it
        # at >= 1.
        scorer = ReunanenScorer(n_hidden=3, M=500, decmin=0.01, seed=0)
        d = 40
        x = {f"f{i}": float(i % 3) for i in range(d)}
        scorer.learn_one(x)
        assert scorer._patience_reset >= 1
        assert scorer._patience_reset == max(1, round(500 * 0.01 / d))

    def test_narrow_input_unchanged(self) -> None:
        """Low-dimensional inputs keep the rounded P_r value."""
        scorer = ReunanenScorer(n_hidden=2, M=200, decmin=0.01, seed=0)
        scorer.learn_one({"a": 0.0, "b": 1.0})
        assert scorer._patience_reset == max(1, round(200 * 0.01 / 2))
