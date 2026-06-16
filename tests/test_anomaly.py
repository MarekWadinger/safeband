"""Regression tests for the Gaussian anomaly scorers."""

import collections
import datetime as dt
import math
import pickle
import sys
from pathlib import Path

import pytest
from river import anomaly, compose, preprocessing
from river.proba import Gaussian
from river.utils import Rolling, TimeRolling

sys.path.insert(1, str(Path(__file__).parent.parent))

from functions.anomaly import (
    AdaptiveThresholdFilter,
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
        # Explicit t_a so the buffer accumulates several flags; the
        # paper default (t_e/4) would re-adapt before three flags pile
        # up, which this test is not exercising.
        scorer = GaussianScorer(Rolling(Gaussian(), 3), grace_period=2, t_a=3)
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


class _ScoreEchoDetector(anomaly.base.AnomalyDetector):
    """Minimal detector whose anomaly score is the sample itself."""

    def __init__(self) -> None:
        self.learned: list[float] = []

    # ty: the deliberate narrowing to plain floats is the point of the
    # echo detector; the filter passes samples through unchanged but may
    # also thread a timestamp ``t`` for the time-based buffer.
    def learn_one(  # ty: ignore[invalid-method-override]
        self,
        x: float,
        **_kwargs: object,
    ) -> None:
        self.learned.append(x)

    def score_one(self, x: float) -> float:  # ty: ignore[invalid-method-override]
        return x


class TestAdaptiveThresholdFilter:
    """Standalone protection filter wrapping an arbitrary detector."""

    def make_filter(
        self,
        t_a: dt.timedelta | int = 4,
        *,
        threshold: float = 0.5,
        protect_anomaly_detector: bool = True,
    ) -> tuple[AdaptiveThresholdFilter, _ScoreEchoDetector]:
        """Build a filter around a learn-recording echo detector."""
        detector = _ScoreEchoDetector()
        filt = AdaptiveThresholdFilter(
            detector,
            threshold=threshold,
            t_a=t_a,
            protect_anomaly_detector=protect_anomaly_detector,
        )
        return filt, detector

    def test_learns_normal_samples(self) -> None:
        """Samples classified as normal reach the wrapped detector."""
        filt, detector = self.make_filter()
        for x in [0.1, 0.2, 0.3]:
            filt.learn_one(x)
        assert detector.learned == [0.1, 0.2, 0.3]

    def test_skips_anomalous_sample(self) -> None:
        """A sporadic anomaly never reaches the wrapped detector."""
        filt, detector = self.make_filter()
        for x in [0.1, 0.2, 0.3]:
            filt.learn_one(x)
        filt.learn_one(0.9)
        assert detector.learned == [0.1, 0.2, 0.3]
        assert filt.drift_detected is False

    def test_readapts_on_changepoint(self) -> None:
        """Learning resumes once anomalies dominate the buffer."""
        filt, detector = self.make_filter()
        for x in [0.1, 0.1, 0.1, 0.1]:
            filt.learn_one(x)
        # Buffer fills with anomaly flags: 1/4 and 2/4 stay at or
        # below the threshold, 3/4 exceeds it and re-enables learning.
        filt.learn_one(0.9)
        filt.learn_one(0.9)
        assert detector.learned == [0.1, 0.1, 0.1, 0.1]
        filt.learn_one(0.9)
        assert filt.drift_detected is True
        assert detector.learned[-1] == 0.9

    def test_count_based_buffer(self) -> None:
        """An int adaptation period yields a bounded deque buffer."""
        filt, _ = self.make_filter(t_a=4)
        assert isinstance(filt.buffer, collections.deque)
        assert filt.buffer.maxlen == 4
        filt.learn_one(0.1)
        assert len(filt.buffer) == 1

    def test_time_based_buffer(self) -> None:
        """A timedelta adaptation period yields a time rolling buffer."""
        filt, detector = self.make_filter(t_a=dt.timedelta(hours=1))
        assert isinstance(filt.buffer, TimeRollingBuffer)
        # river TimeRolling compares timestamps internally; use naive
        # datetimes (as the server's preprocess produces) to avoid
        # mixing offset-aware and offset-naive values.
        t = dt.datetime(2024, 1, 1, tzinfo=dt.UTC).replace(tzinfo=None)
        filt.learn_one(0.1, t=t)
        assert len(filt.buffer) == 1
        assert detector.learned == [0.1]

    def test_time_based_buffer_evicts_by_time(self) -> None:
        """Flags older than t_a are evicted so drift reflects the window."""
        filt, _ = self.make_filter(t_a=dt.timedelta(hours=1))
        base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC).replace(tzinfo=None)
        # Three anomalies inside the window: drift fires (3/3 > thresh).
        for i in range(3):
            filt.learn_one(0.9, t=base + dt.timedelta(minutes=i))
        assert len(filt.buffer) == 3
        assert filt.drift_detected is True
        # A normal sample more than an hour later evicts all old flags;
        # only the recent (single, normal) flag remains in the window.
        filt.learn_one(0.1, t=base + dt.timedelta(hours=2))
        assert len(filt.buffer) == 1
        assert filt.drift_detected is False

    def test_time_based_buffer_requires_timestamp(self) -> None:
        """A timedelta t_a without a timestamp raises a clear error."""
        filt, _ = self.make_filter(t_a=dt.timedelta(hours=1))
        with pytest.raises(ValueError, match="timestamped samples"):
            filt.learn_one(0.1)

    def test_unprotected_filter_learns_everything(self) -> None:
        """Without protection every sample is learned and none gated."""
        filt, detector = self.make_filter(protect_anomaly_detector=False)
        filt.learn_one(0.9)
        assert detector.learned == [0.9]
        assert len(filt.buffer) == 0

    def test_predict_one_falls_back_to_classify(self) -> None:
        """Without detector predict_one the score is thresholded."""
        filt, _ = self.make_filter()
        assert filt.predict_one(0.9) == 1
        assert filt.predict_one(0.1) == 0

    def test_predict_one_prefers_detector_predict(self) -> None:
        """A wrapped scorer's own two-sided decision rule drives gating."""
        scorer = GaussianScorer(
            Rolling(Gaussian(), 5),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        filt = AdaptiveThresholdFilter(scorer, t_a=4)
        for x in [1.0, 0.0, 1.0, 0.0]:
            filt.learn_one(x)
        # The raw CDF score of a low outlier is ~0, below the filter's
        # threshold; only the scorer's two-sided rule flags it.
        assert filt.score_one(-10.0) < filt.threshold
        assert filt.predict_one(-10.0) == 1


class TestSingleProtectionLayer:
    """The filter is the only learn gate of a protected scorer."""

    def make_trained_scorer(self) -> GaussianScorer:
        """Build a protected scorer trained on three normal samples."""
        # Explicit t_a=3 so the changepoint buffer holds three flags;
        # the paper default (t_e/4 == 1) would re-adapt on the first
        # anomaly, which these tests are not exercising.
        scorer = GaussianScorer(
            Rolling(Gaussian(), 3),
            grace_period=2,
            t_a=3,
        )
        for value in [1.0, 1.1, 0.9]:
            scorer.learn_one(value)
        return scorer

    def test_process_one_predicts_and_learns_exactly_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """process_one neither double-predicts nor double-gates."""
        scorer = self.make_trained_scorer()
        predict_calls: list[float] = []
        learn_calls: list[float] = []
        original_predict = scorer.predict_one
        original_learn = scorer._learn_one

        def spy_predict(x: float) -> int:
            predict_calls.append(x)
            return original_predict(x)

        def spy_learn(
            x: float,
            **kwargs: dt.datetime | float | None,
        ) -> GaussianScorer:
            learn_calls.append(x)
            return original_learn(x, **kwargs)

        monkeypatch.setattr(scorer, "predict_one", spy_predict)
        monkeypatch.setattr(scorer, "_learn_one", spy_learn)
        scorer.process_one(1.0)
        assert predict_calls == [1.0]
        assert learn_calls == [1.0]

    def test_process_one_records_anomalies_for_readaptation(self) -> None:
        """Anomalies seen via process_one drive changepoint adaptation."""
        scorer = self.make_trained_scorer()
        assert scorer.process_one(10.0)[0] == 1
        assert scorer.process_one(10.0)[0] == 1
        # Two anomalies fill 2/3 of the buffer; the distribution is
        # untouched so far.
        assert scorer.gaussian.mu == pytest.approx(1.0)
        assert scorer.drift_detected is False
        # The third anomaly tips the buffer over the threshold: the
        # changepoint fires and the sample is learned (re-adaptation).
        assert scorer.process_one(10.0)[0] == 1
        assert scorer.drift_detected is True
        assert scorer.gaussian.mu == pytest.approx(4.0)

    def test_learn_one_skips_anomaly_without_changepoint(self) -> None:
        """A sporadic anomaly is buffered but not learned."""
        scorer = self.make_trained_scorer()
        scorer.learn_one(10.0)
        assert scorer.gaussian.mu == pytest.approx(1.0)
        assert list(scorer.buffer) == [0, 0, 1]


class TestGetLimits:
    """Public per-signal dynamic limits keyed by feature name."""

    def test_keys_and_values_match_limit_one(self) -> None:
        """get_limits and limit_one agree on the (upper, lower) order."""
        scorer = make_conditional_scorer()
        x = {"a": 0.4, "b": 0.5}
        ths, tls = scorer.limit_one(x)
        limits = scorer.get_limits(x)
        assert set(limits) == set(ths) == set(tls)
        for key, (upper, lower) in limits.items():
            assert upper == ths[key]
            assert lower == tls[key]
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
        # get_limits returns (upper, lower), matching limit_one.
        assert limits["b"][0] == min(free_limits["b"][0], 0.6)
        assert limits["b"][1] == max(free_limits["b"][1], 0.3)
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


class TestLegacyPickleProtection:
    """Models pickled before _protection existed degrade gracefully."""

    def test_buffer_and_drift_survive_missing_protection(self) -> None:
        """A model without _protection rebuilds it instead of raising."""
        scorer = GaussianScorer(Rolling(Gaussian(), 4), grace_period=2)
        for value in [1.0, 1.1, 0.9]:
            scorer.learn_one(value)
        # Simulate a legacy pickle lacking the protection attribute.
        del scorer._protection
        # Access must not raise AttributeError.
        assert isinstance(
            scorer.buffer,
            (collections.deque, TimeRollingBuffer),
        )
        assert scorer.drift_detected is False


class TestTaDefault:
    """The adaptation period t_a defaults to t_e / 4 (paper guidance)."""

    def test_int_default_is_quarter_of_window(self) -> None:
        """Count-based t_e defaults t_a to max(1, round(t_e / 4))."""
        scorer = GaussianScorer(Rolling(Gaussian(), 8), grace_period=2)
        assert scorer.t_a == 2

    def test_int_default_floored_at_one(self) -> None:
        """A tiny window still yields a usable (>= 1) adaptation period."""
        scorer = GaussianScorer(Rolling(Gaussian(), 2), grace_period=1)
        assert scorer.t_a == 1

    def test_timedelta_default_is_quarter_of_period(self) -> None:
        """Time-based t_e defaults t_a to t_e / 4."""
        period = dt.timedelta(hours=4)
        scorer = GaussianScorer(
            TimeRolling(Gaussian(), period=period),
            grace_period=dt.timedelta(hours=1),
        )
        assert scorer.t_a == period / 4

    def test_explicit_zero_is_honored(self) -> None:
        """t_a=0 (disable re-adaptation) is not overwritten by the default."""
        scorer = GaussianScorer(
            Rolling(Gaussian(), 8),
            grace_period=2,
            t_a=0,
        )
        assert scorer.t_a == 0


class TestGracePeriodImmutable:
    """Scoring past the grace window must not mutate grace_period."""

    def test_grace_period_unchanged_after_scoring(self) -> None:
        """grace_period survives scoring past the warm-up window."""
        scorer = ConditionalGaussianScorer(
            Rolling(MultivariateGaussian(seed=42), 5),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        for sample in SAMPLES:
            scorer.learn_one(sample)
        # Score well past the grace window several times.
        for _ in range(5):
            scorer.score_one({"a": 0.5, "b": 0.5})
        assert scorer.grace_period == 2

    def test_grace_period_unchanged_across_repickle(self) -> None:
        """A re-instantiated model keeps the same configured grace_period."""
        scorer = ConditionalGaussianScorer(
            Rolling(MultivariateGaussian(seed=42), 5),
            grace_period=2,
            protect_anomaly_detector=False,
        )
        for sample in SAMPLES:
            scorer.learn_one(sample)
        scorer.score_one({"a": 0.5, "b": 0.5})
        # Round-trip through pickle: the recovered model must keep its
        # configured grace_period so _warn_on_param_mismatch stays quiet.
        restored = pickle.loads(pickle.dumps(scorer))  # noqa: S301
        assert restored.grace_period == 2
