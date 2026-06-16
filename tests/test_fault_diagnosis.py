"""Tests for streaming sensor fault-type classification (IDEAS I7)."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from river.utils import Rolling
from scipy.stats import norm

sys.path.insert(1, str(Path(__file__).parent.parent))

from functions.anomaly import ConditionalGaussianScorer
from functions.fault_diagnosis import FaultLabel, SensorFaultClassifier
from functions.proba import MultivariateGaussian

N_SIGNALS = 4
SIGNALS = [f"s{i}" for i in range(N_SIGNALS)]
FAULTY = "s0"
HEALTHY = [s for s in SIGNALS if s != FAULTY]

N_TRAIN = 400
N_WARMUP = 150
N_FAULT = 250
WINDOW = 25
LONG_WINDOW = 100


def healthy_samples(
    n: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Correlated Gaussian stream: common latent factor + own noise."""
    latent = rng.standard_normal(n)
    noise = rng.standard_normal((n, N_SIGNALS))
    return [
        {s: 0.6 * latent[t] + 0.8 * noise[t, i] for i, s in enumerate(SIGNALS)}
        for t in range(n)
    ]


@pytest.fixture(scope="module")
def pipeline() -> tuple[
    ConditionalGaussianScorer,
    SensorFaultClassifier,
    np.random.Generator,
]:
    """Scorer trained on healthy data and a warmed-up classifier.

    The scorer is treated as read-only afterwards (no learning during
    fault injection — in deployment the protected scorer rejects
    anomalous samples), so tests deepcopy only the classifier.
    """
    rng = np.random.default_rng(7)
    samples = healthy_samples(N_TRAIN + N_WARMUP, rng)
    scorer = ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=42), N_TRAIN),
        grace_period=50,
        protect_anomaly_detector=False,
    )
    for x in samples[:N_TRAIN]:
        scorer.learn_one(x)
    clf = SensorFaultClassifier(window=WINDOW, long_window=LONG_WINDOW)
    for x in samples[N_TRAIN:]:
        clf.process_one(x, scorer.residuals_one(x))
    return scorer, clf, rng


@pytest.fixture
def fresh(
    pipeline: tuple[
        ConditionalGaussianScorer,
        SensorFaultClassifier,
        np.random.Generator,
    ],
) -> tuple[
    ConditionalGaussianScorer,
    SensorFaultClassifier,
    np.random.Generator,
]:
    """Per-test classifier copy plus a per-test reproducible rng."""
    scorer, clf, _ = pipeline
    return scorer, copy.deepcopy(clf), np.random.default_rng(11)


def run_stream(
    scorer: ConditionalGaussianScorer,
    clf: SensorFaultClassifier,
    stream: list[dict[str, float]],
) -> list[dict[str, FaultLabel]]:
    """Classify a stream against a fixed scorer; collect labels."""
    return [clf.process_one(x, scorer.residuals_one(x)) for x in stream]


def first_index(
    labels_seq: list[dict[str, FaultLabel]],
    name: str,
    label: str,
) -> int | None:
    """Return the first step at which ``name`` carries ``label``."""
    for i, labels in enumerate(labels_seq):
        if labels[name] == label:
            return i
    return None


def assert_healthy_normal(labels_seq: list[dict[str, FaultLabel]]) -> None:
    """All healthy signals must stay 'normal' on every step."""
    for labels in labels_seq:
        for s in HEALTHY:
            assert labels[s] == "normal"


def test_healthy_stream_stays_normal(fresh) -> None:  # noqa: ANN001
    """No fault labels at all on a continued healthy stream."""
    scorer, clf, rng = fresh
    labels_seq = run_stream(scorer, clf, healthy_samples(200, rng))
    for labels in labels_seq:
        assert all(label == "normal" for label in labels.values())


def test_bias_fault_labeled_within_delay(fresh) -> None:  # noqa: ANN001
    """A constant offset settles to 'bias'; healthy peers stay normal."""
    scorer, clf, rng = fresh
    stream = healthy_samples(N_FAULT, rng)
    for x in stream:
        x[FAULTY] += 4.0
    labels_seq = run_stream(scorer, clf, stream)

    detected = first_index(labels_seq, FAULTY, "bias")
    assert detected is not None
    assert detected <= 150
    tail = [labels[FAULTY] for labels in labels_seq[-50:]]
    assert tail.count("bias") / len(tail) >= 0.9
    assert_healthy_normal(labels_seq)


def test_drift_fault_labeled_within_delay(fresh) -> None:  # noqa: ANN001
    """A single-sensor ramp is labeled 'drift'; peers stay normal."""
    scorer, clf, rng = fresh
    stream = healthy_samples(N_FAULT, rng)
    for t, x in enumerate(stream):
        x[FAULTY] += 0.032 * t
    labels_seq = run_stream(scorer, clf, stream)

    detected = first_index(labels_seq, FAULTY, "drift")
    assert detected is not None
    assert detected <= 175
    tail = [labels[FAULTY] for labels in labels_seq[-100:]]
    assert tail.count("drift") / len(tail) >= 0.7
    assert_healthy_normal(labels_seq)


def test_accuracy_loss_labeled_within_delay(fresh) -> None:  # noqa: ANN001
    """Inflated noise with zero offset is 'accuracy_loss'."""
    scorer, clf, rng = fresh
    stream = healthy_samples(N_FAULT, rng)
    extra = rng.standard_normal(N_FAULT)
    for t, x in enumerate(stream):
        x[FAULTY] += 4.0 * extra[t]
    labels_seq = run_stream(scorer, clf, stream)

    detected = first_index(labels_seq, FAULTY, "accuracy_loss")
    assert detected is not None
    assert detected <= 100
    tail = [labels[FAULTY] for labels in labels_seq[-100:]]
    assert tail.count("accuracy_loss") / len(tail) >= 0.6
    assert_healthy_normal(labels_seq)


def test_freezing_near_conditional_mean_is_caught(fresh) -> None:  # noqa: ANN001
    """A value frozen at the signal mean is the scorer's blind spot.

    The per-signal conditional score stays mid-range (the
    ``cond_std -> 0`` branch never alarms a frozen-but-plausible
    value), yet the stuck-at test flags it within ``freeze_window``.
    """
    scorer, clf, rng = fresh
    frozen_at = float(scorer.gaussian.mu[FAULTY])
    stream = healthy_samples(N_FAULT, rng)
    for x in stream:
        x[FAULTY] = frozen_at
    labels_seq = run_stream(scorer, clf, stream)

    detected = first_index(labels_seq, FAULTY, "freezing")
    assert detected is not None
    assert detected <= WINDOW + 2
    for labels in labels_seq[detected:]:
        assert labels[FAULTY] == "freezing"
    assert_healthy_normal(labels_seq)

    # Blind-spot evidence: the scorer alone keeps the frozen signal
    # well inside its limits on the vast majority of steps.
    inside = [0.023 < scorer.scores_one(x)[FAULTY] < 0.977 for x in stream]
    assert sum(inside) / len(inside) >= 0.9


def test_regime_change_not_attributed_to_sensors(fresh) -> None:  # noqa: ANN001
    """A coordinated shift of ALL signals yields no bias/drift labels.

    The conditional means follow the peers, so per-signal residuals
    stay small — the scorer adapts (regime change) instead of blaming
    a sensor.
    """
    scorer, clf, rng = fresh
    stream = healthy_samples(N_FAULT, rng)
    for x in stream:
        for s in SIGNALS:
            x[s] += 3.0
    labels_seq = run_stream(scorer, clf, stream)

    for labels in labels_seq:
        for s in SIGNALS:
            assert labels[s] not in ("bias", "drift")


def test_drift_detected_flag_suppresses_residual_labels() -> None:
    """The scorer's changepoint flag suppresses bias/drift/accuracy."""
    flagged = SensorFaultClassifier(window=2, long_window=4)
    plain = SensorFaultClassifier(window=2, long_window=4)
    x = 0.0
    for _ in range(10):
        x = 1.0 - x
        suppressed = flagged.process_one(
            {"a": x},
            {"a": (8.0, 1.0)},
            drift_detected=True,
        )
        labeled = plain.process_one({"a": x}, {"a": (8.0, 1.0)})
    assert suppressed["a"] == "normal"
    assert labeled["a"] == "bias"


def test_freezing_is_not_suppressed_by_drift_flag() -> None:
    """Freezing is raw-innovation based and survives regime changes."""
    clf = SensorFaultClassifier(window=3)
    labels = {}
    for _ in range(6):
        labels = clf.process_one(
            {"a": 1.0},
            {"a": (0.0, 1.0)},
            drift_detected=True,
        )
    assert labels["a"] == "freezing"


def test_warmup_returns_normal_despite_evidence() -> None:
    """Residual-based labels need ``window`` updates of history."""
    clf = SensorFaultClassifier(window=10, long_window=20)
    x = 0.0
    for _ in range(9):
        x = 1.0 - x
        labels = clf.process_one({"a": x}, {"a": (9.0, 1.0)})
        assert labels["a"] == "normal"


def test_residuals_one_matches_conditional_moments(fresh) -> None:  # noqa: ANN001
    """Public accessor agrees with the per-signal conditional scores."""
    scorer, _, rng = fresh
    x = healthy_samples(1, rng)[0]
    residuals = scorer.residuals_one(x)
    scores = scorer.scores_one(x)
    assert set(residuals) == set(x)
    for s in SIGNALS:
        res, cond_std = residuals[s]
        assert cond_std > 0
        assert scores[s] == pytest.approx(norm.cdf(res / cond_std))


class TestFrozenResidualsExcludedFromBaseline:
    """Residual baselines must ignore the frozen-period constant residual."""

    def test_freeze_does_not_shift_residual_means(self) -> None:
        """A large residual during a freeze does not move short/long EWMAs."""
        clf = SensorFaultClassifier(window=3, long_window=6, freeze_window=3)
        # Healthy phase: a varying signal with near-zero residuals so the
        # baselines settle around zero.
        value = 0.0
        for _ in range(20):
            value = 1.0 - value
            clf.process_one({"s": value}, {"s": (0.0, 1.0)})
        diag_before = clf.diagnostics["s"]

        # Freeze: the raw value stays constant (triggers the freeze test)
        # while the reported residual is large. Once freeze_window
        # consecutive constant points are seen the signal is frozen and
        # the large residual must NOT be folded into the baselines.
        for _ in range(30):
            labels = clf.process_one({"s": 5.0}, {"s": (8.0, 1.0)})
        assert labels["s"] == "freezing"
        diag_after = clf.diagnostics["s"]
        # n grows only by the short pre-freeze ramp (<= freeze_window),
        # never by the long frozen tail: the 30 frozen residuals are
        # gated out. The buggy version folded every frozen residual and
        # would push n up by ~30.
        ramp = diag_after["n"] - diag_before["n"]
        assert ramp <= clf.freeze_window
        assert diag_after["n"] < diag_before["n"] + 30

    def test_recovery_after_freeze_is_not_spuriously_biased(self) -> None:
        """Healthy residuals after a freeze are labelled normal, not bias."""
        clf = SensorFaultClassifier(window=3, long_window=6, freeze_window=3)
        value = 0.0
        for _ in range(20):
            value = 1.0 - value
            clf.process_one({"s": value}, {"s": (0.0, 1.0)})
        # Freeze with a large constant residual.
        for _ in range(10):
            clf.process_one({"s": 5.0}, {"s": (8.0, 1.0)})
        # Recovery: the signal varies again with healthy residuals.
        value = 5.0
        labels: dict[str, FaultLabel] = {}
        for _ in range(10):
            value = 6.0 - value
            labels = clf.process_one({"s": value}, {"s": (0.0, 1.0)})
        assert labels["s"] == "normal"
