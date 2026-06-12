"""Regression tests for the Gaussian anomaly scorers."""

import datetime as dt
import math
import sys
from pathlib import Path

import pytest
from river import compose, preprocessing
from river.proba import Gaussian
from river.utils import Rolling, TimeRolling

sys.path.insert(1, str(Path(__file__).parent.parent))

from functions.anomaly import (
    ConditionalGaussianScorer,
    GaussianScorer,
    TimeRollingBuffer,
)
from functions.proba import MultivariateGaussian

SAMPLES = [
    {"a": 0.1, "b": 0.6},
    {"a": 0.5, "b": 0.2},
    {"a": 0.3, "b": 0.9},
    {"a": 0.7, "b": 0.4},
    {"a": 0.2, "b": 0.8},
]


def make_multivariate_scorer(seed: int | None = 42) -> GaussianScorer:
    """Build a multivariate GaussianScorer fitted on the shared samples."""
    scorer = GaussianScorer(
        Rolling(MultivariateGaussian(seed=seed), 5),
        grace_period=2,
        protect_anomaly_detector=False,
    )
    for sample in SAMPLES:
        scorer.learn_one(sample)
    return scorer


def make_conditional_scorer(
    physical_limits: dict[str, tuple[float, float]] | None = None,
    learn_on_physical_violation: bool = False,
) -> ConditionalGaussianScorer:
    """Build a ConditionalGaussianScorer fitted on the shared samples."""
    scorer = ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=42), 5),
        grace_period=2,
        protect_anomaly_detector=False,
        physical_limits=physical_limits,
        learn_on_physical_violation=learn_on_physical_violation,
    )
    for sample in SAMPLES:
        scorer.learn_one(sample)
    return scorer


class TestBufferLength:
    """Tests for the anomaly buffer length without monkey-patching river."""

    def test_time_rolling_is_not_monkey_patched(self) -> None:
        """The river TimeRolling class gains no __len__ from our import."""
        assert "__len__" not in vars(TimeRolling)

    def test_buffer_reports_store_length(self) -> None:
        """A timedelta protection window yields a sized buffer."""
        scorer = GaussianScorer(
            TimeRolling(Gaussian(), period=dt.timedelta(days=1)),
            grace_period=dt.timedelta(hours=1),
        )
        assert isinstance(scorer.buffer, TimeRollingBuffer)
        assert len(scorer.buffer) == 0
        # river's TimeRolling compares against a naive sentinel timestamp,
        # so strip the timezone the same way the server's preprocess does.
        t = dt.datetime(2022, 1, 1, tzinfo=dt.UTC).replace(tzinfo=None)
        scorer.learn_one(1.0, t=t)
        assert len(scorer.buffer) == 1


class TestReproducibleScores:
    """Repeated scoring without learning must return identical results."""

    def test_multivariate_score_one_is_reproducible(self) -> None:
        """Seeded multivariate CDF scores match across repeated calls."""
        scorer = make_multivariate_scorer()
        x = {"a": 0.9, "b": 0.1}
        assert scorer.score_one(x) == scorer.score_one(x)

    def test_multivariate_limit_one_is_reproducible(self) -> None:
        """Multivariate limits match across repeated calls."""
        scorer = make_multivariate_scorer()
        x = {"a": 0.9, "b": 0.1}
        assert scorer.limit_one(x) == scorer.limit_one(x)

    def test_univariate_score_one_is_reproducible(self) -> None:
        """Univariate scores match across repeated calls."""
        scorer = GaussianScorer(
            Rolling(Gaussian(), 5),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        for value in [1.0, 2.0, 1.5, 1.2]:
            scorer.learn_one(value)
        assert scorer.score_one(1.8) == scorer.score_one(1.8)
        assert scorer.limit_one() == scorer.limit_one()

    def test_conditional_score_one_is_reproducible(self) -> None:
        """Conditional scores match across repeated calls."""
        scorer = make_conditional_scorer()
        x = {"a": 0.9, "b": 0.1}
        assert scorer.score_one(x) == scorer.score_one(x)
        assert scorer.limit_one(x) == scorer.limit_one(x)


class TestUnscorableInput:
    """Explicit contract when no conditional score can be computed."""

    def test_empty_input_scores_neutral(self) -> None:
        """An input without features yields the neutral score 0.5."""
        scorer = make_conditional_scorer()
        assert scorer.score_one({}) == 0.5

    def test_empty_input_is_not_flagged(self) -> None:
        """An input without features is not predicted anomalous."""
        scorer = make_conditional_scorer()
        assert scorer.predict_one({}) == 0
        assert scorer.get_root_cause() is None


class TestScoresOne:
    """Public per-signal conditional scores keyed by feature name."""

    def test_keys_match_input_features(self) -> None:
        """scores_one keys follow the input observation's features."""
        scorer = make_conditional_scorer()
        x = {"a": 0.9, "b": 0.1}
        assert list(scorer.scores_one(x)) == list(x)

    def test_values_match_positional_scores(self) -> None:
        """scores_one values equal _scores_one in positional order."""
        scorer = make_conditional_scorer()
        x = {"a": 0.9, "b": 0.1}
        assert list(scorer.scores_one(x).values()) == scorer._scores_one(x)

    def test_unfitted_scorer_returns_neutral_scores(self) -> None:
        """Before any learning, every feature scores the neutral 0.5."""
        scorer = ConditionalGaussianScorer(
            Rolling(MultivariateGaussian(seed=42), 5),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        assert scorer.scores_one({"a": 1.0, "b": 2.0}) == {
            "a": 0.5,
            "b": 0.5,
        }


class TestRankRootCauses:
    """Ranked root-cause candidates beyond the argmax diagnostic."""

    def test_ordering_by_deviation_from_center(self) -> None:
        """Features are sorted by |score - 0.5| in decreasing order."""
        scorer = make_conditional_scorer()
        x = {"a": 0.9, "b": 0.1}
        scores = scorer.scores_one(x)
        expected = sorted(
            scores,
            key=lambda name: abs(scores[name] - 0.5),
            reverse=True,
        )
        assert scorer.rank_root_causes(x) == expected

    def test_top_k_truncates_ranking(self) -> None:
        """A small k truncates the ranking; oversized k returns all."""
        scorer = make_conditional_scorer()
        x = {"a": 0.9, "b": 0.1}
        full = scorer.rank_root_causes(x)
        assert scorer.rank_root_causes(x, k=1) == full[:1]
        assert scorer.rank_root_causes(x, k=10) == full

    def test_first_ranked_matches_get_root_cause(self) -> None:
        """Top-ranked feature agrees with the argmax-based root cause."""
        scorer = make_conditional_scorer()
        x = {"a": 0.4, "b": 5.0}
        assert scorer.predict_one(x) == 1
        assert scorer.rank_root_causes(x, k=1) == [scorer.get_root_cause()]


class TestDriftDetected:
    """Public regime-change indicator on the protected scorer."""

    def test_visible_around_changepoint(self) -> None:
        """drift_detected flips once anomalies dominate the buffer."""
        scorer = GaussianScorer(Rolling(Gaussian(), 3), grace_period=2)
        for value in [1.0, 1.1, 0.9]:
            scorer.learn_one(value)
        assert scorer.drift_detected is False
        for _ in range(3):
            scorer.learn_one(10.0)
        assert scorer.drift_detected is True

    def test_matches_internal_signal(self) -> None:
        """The public property wraps the internal drift method."""
        scorer = GaussianScorer(Rolling(Gaussian(), 3), grace_period=2)
        for value in [1.0, 1.1, 0.9, 10.0, 10.0, 10.0]:
            scorer.learn_one(value)
            assert scorer.drift_detected == scorer._drift_detected()

    def test_false_without_protection(self) -> None:
        """An unprotected scorer keeps no buffer and reports no drift."""
        scorer = GaussianScorer(
            Rolling(Gaussian(), 3),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        scorer.learn_one(1.0)
        assert scorer.drift_detected is False


class TestGetLimits:
    """Public per-signal dynamic limits keyed by feature name."""

    def test_keys_and_values_match_limit_one(self) -> None:
        """get_limits regroups limit_one's (upper, lower) per feature."""
        scorer = make_conditional_scorer()
        x = {"a": 0.4, "b": 0.5}
        ths, tls = scorer.limit_one(x)
        limits = scorer.get_limits(x)
        assert set(limits) == set(ths) == set(tls)
        for key, (lower, upper) in limits.items():
            assert lower == tls[key]
            assert upper == ths[key]
            assert lower < upper

    def test_unfitted_scorer_returns_nan_limits(self) -> None:
        """Before any learning, limits are NaN for every feature."""
        scorer = ConditionalGaussianScorer(
            Rolling(MultivariateGaussian(seed=42), 5),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        limits = scorer.get_limits({"a": 1.0, "b": 2.0})
        assert set(limits) == {"a", "b"}
        assert all(
            math.isnan(lower) and math.isnan(upper)
            for lower, upper in limits.values()
        )


class TestScorerInPipeline:
    """The scorers keep consistent state inside a river Pipeline."""

    def test_limit_one_before_fit_returns_nan_limits(self) -> None:
        """limit_one called before any learning returns NaN limits."""
        scorer = ConditionalGaussianScorer(
            Rolling(MultivariateGaussian(seed=42), 5),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        ths, tls = scorer.limit_one({"a": 1.0, "b": 2.0})
        assert set(ths) == set(tls) == {"a", "b"}
        assert all(math.isnan(v) for v in ths.values())
        assert all(math.isnan(v) for v in tls.values())

    def test_process_one_keeps_pipeline_state_consistent(self) -> None:
        """process_one on the wrapped scorer does not corrupt its state."""
        scorer = ConditionalGaussianScorer(
            Rolling(MultivariateGaussian(seed=42), 10),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        pipeline = compose.Pipeline(
            ("scale", preprocessing.StandardScaler()),
            ("scorer", scorer),
        )
        for sample in SAMPLES:
            pipeline.learn_one(sample)
        n_before = scorer.gaussian.n_samples

        x = {"a": 0.4, "b": 0.5}
        x_scaled = pipeline["scale"].transform_one(x)
        is_anomaly, ths, tls = scorer.process_one(x_scaled)

        assert is_anomaly in (0, 1)
        assert isinstance(ths, dict)
        assert isinstance(tls, dict)
        assert set(ths) == set(tls) == {"a", "b"}
        # A normal sample is learned; state advances by exactly one.
        assert scorer.gaussian.n_samples == n_before + (1 - is_anomaly)
        # The pipeline keeps scoring without errors after process_one.
        assert 0.0 <= pipeline.score_one(x) <= 1.0


def make_univariate_scorer(
    physical_limits: tuple[float, float] | None = None,
    learn_on_physical_violation: bool = False,
) -> GaussianScorer:
    """Build a univariate GaussianScorer fitted on a few samples."""
    scorer = GaussianScorer(
        Rolling(Gaussian(), 5),
        grace_period=2,
        protect_anomaly_detector=False,
        physical_limits=physical_limits,
        learn_on_physical_violation=learn_on_physical_violation,
    )
    for value in [1.0, 2.0, 1.5, 1.2]:
        scorer.learn_one(value)
    return scorer


class TestPhysicalLimitsUnivariate:
    """Static physical bounds on the univariate scorer."""

    def test_invalid_bounds_raise(self) -> None:
        """Bounds with low >= high are rejected at construction."""
        with pytest.raises(ValueError, match="low < high"):
            GaussianScorer(Gaussian(), physical_limits=(2.0, 1.0))

    def test_limits_clipped_into_physical_bounds(self) -> None:
        """Reported limits are the learned limits clipped to the bounds."""
        free = make_univariate_scorer()
        bounded = make_univariate_scorer(physical_limits=(1.0, 1.6))
        high_free, low_free = free.limit_one()
        high, low = bounded.limit_one()
        assert isinstance(high_free, float)
        assert isinstance(low_free, float)
        assert high == min(high_free, 1.6)
        assert low == max(low_free, 1.0)

    def test_violation_forced_anomaly_during_grace_period(self) -> None:
        """A physically impossible value is flagged before any learning."""
        scorer = GaussianScorer(
            Rolling(Gaussian(), 5),
            grace_period=100,
            protect_anomaly_detector=False,
            physical_limits=(0.0, 10.0),
        )
        assert scorer.predict_one(11.0) == 1
        assert scorer.predict_one(5.0) == 0

    def test_violation_forced_anomaly_after_grace_period(self) -> None:
        """A statistically unremarkable value is flagged when impossible."""
        free = make_univariate_scorer()
        bounded = make_univariate_scorer(physical_limits=(0.0, 1.7))
        assert free.predict_one(1.8) == 0
        assert bounded.predict_one(1.8) == 1

    def test_violations_excluded_from_learning_by_default(self) -> None:
        """Physically impossible samples do not update the distribution."""
        scorer = make_univariate_scorer(physical_limits=(0.0, 2.5))
        n = scorer.gaussian.n_samples
        scorer.learn_one(99.0)
        assert scorer.gaussian.n_samples == n

    def test_learn_on_physical_violation_keeps_learning(self) -> None:
        """With the opt-in flag, impossible samples are still learned."""
        scorer = make_univariate_scorer(
            physical_limits=(0.0, 2.5),
            learn_on_physical_violation=True,
        )
        n = scorer.gaussian.n_samples
        scorer.learn_one(99.0)
        assert scorer.gaussian.n_samples == n + 1

    def test_in_bounds_samples_still_learned(self) -> None:
        """Samples inside the physical bounds keep updating the model."""
        scorer = make_univariate_scorer(physical_limits=(0.0, 2.5))
        n = scorer.gaussian.n_samples
        scorer.learn_one(1.3)
        assert scorer.gaussian.n_samples == n + 1


class TestPhysicalLimitsConditional:
    """Per-feature static physical bounds on the conditional scorer."""

    def test_invalid_bounds_raise(self) -> None:
        """Per-feature bounds with low >= high are rejected."""
        with pytest.raises(ValueError, match="low < high"):
            ConditionalGaussianScorer(
                Rolling(MultivariateGaussian(seed=42), 5),
                grace_period=2,
                physical_limits={"a": (1.0, 1.0)},
            )

    def test_violation_forces_anomaly_during_grace_period(self) -> None:
        """A violating feature is flagged before any learning."""
        scorer = ConditionalGaussianScorer(
            Rolling(MultivariateGaussian(seed=42), 5),
            grace_period=100,
            protect_anomaly_detector=False,
            physical_limits={"b": (0.0, 1.0)},
        )
        assert scorer.predict_one({"a": 0.1, "b": 5.0}) == 1
        assert scorer.get_root_cause() == "b"
        assert scorer.predict_one({"a": 0.1, "b": 0.5}) == 0

    def test_root_cause_prefers_physically_violated_feature(self) -> None:
        """The violated feature wins over a statistically worse one."""
        scorer = make_conditional_scorer(physical_limits={"b": (0.0, 1.0)})
        # 'a' deviates far more statistically, but only 'b' breaks its
        # physical bound and must be attributed as root cause.
        x = {"a": 50.0, "b": 1.1}
        assert scorer.predict_one(x) == 1
        assert scorer.get_root_cause() == "b"

    def test_root_cause_picks_most_violated_feature(self) -> None:
        """With several violations, the worst offender is attributed."""
        scorer = make_conditional_scorer(
            physical_limits={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        assert scorer.predict_one({"a": 1.2, "b": 9.0}) == 1
        assert scorer.get_root_cause() == "b"

    def test_limits_clipped_per_feature(self) -> None:
        """Only features with configured bounds get their limits clipped."""
        free = make_conditional_scorer()
        bounded = make_conditional_scorer(
            physical_limits={"b": (0.3, 0.6)},
            learn_on_physical_violation=True,
        )
        x = {"a": 0.4, "b": 0.5}
        free_limits = free.get_limits(x)
        limits = bounded.get_limits(x)
        assert limits["b"][0] == max(free_limits["b"][0], 0.3)
        assert limits["b"][1] == min(free_limits["b"][1], 0.6)
        assert limits["a"] == free_limits["a"]

    def test_learning_exclusion_flag(self) -> None:
        """Impossible samples are skipped unless learning is opted in."""
        # The window must be larger than the fitted sample count so a
        # learned observation is visible as an n_samples increment.
        scorer = ConditionalGaussianScorer(
            Rolling(MultivariateGaussian(seed=42), 10),
            grace_period=2,
            protect_anomaly_detector=False,
            physical_limits={"b": (0.0, 1.0)},
        )
        for sample in SAMPLES:
            scorer.learn_one(sample)
        n = scorer.gaussian.n_samples
        scorer.learn_one({"a": 0.5, "b": 2.0})
        assert scorer.gaussian.n_samples == n
        scorer.learn_one({"a": 0.5, "b": 0.5})
        assert scorer.gaussian.n_samples == n + 1
