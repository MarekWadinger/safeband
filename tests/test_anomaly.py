"""Regression tests for the Gaussian anomaly scorers."""

import datetime as dt
import math
import sys
from pathlib import Path

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


def make_conditional_scorer() -> ConditionalGaussianScorer:
    """Build a ConditionalGaussianScorer fitted on the shared samples."""
    scorer = ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=42), 5),
        grace_period=2,
        protect_anomaly_detector=False,
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
