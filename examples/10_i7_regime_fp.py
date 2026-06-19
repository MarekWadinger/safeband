"""I7 false alarms during a genuine regime change, vs correlation.

Adversarial review (Claim 2) argued the "conditional residuals cancel a
coordinated regime shift" mechanism silently assumes strong inter-signal
correlation: when signals are weakly correlated the conditional mean
``E[x_i | x_rest]`` barely tracks its peers, so a coordinated shift
produces LARGE conditional residuals and the classifier raises false
bias/drift labels on healthy signals -- the opposite of the intended
"adapt, don't alarm" behaviour.

This script tests that directly on a synthetic correlated-Gaussian
carrier with a *tunable* pairwise correlation ``rho`` (signal =
``sqrt(rho)*latent + sqrt(1-rho)*idiosyncratic``). After a clean
training prefix, a coordinated step shift is applied to EVERY signal
simultaneously -- a legitimate regime change, so the correct label for
every signal is ``normal``. The false-positive rate is the fraction of
(signal, step) labelled non-``normal`` during the regime window.

Two questions:

1. Does FP rate fall as ``rho`` rises (mechanism 1 -- residual
   cancellation -- only works when correlation is high)?
2. Does the graded change-point suppression (mechanism 2,
   ``suppress_threshold_scale``) cover the low-correlation gap? Compared
   at scale 1.0 (no suppression), 5.0 (default) and inf (hard zero).

Artifact: ``examples/benchmarks/i7_regime_fp.csv`` -- FP rate (mean and
std over seeds) per (rho, suppress_threshold_scale).

Run::

    uv run python examples/10_i7_regime_fp.py

CPU-only; ~3 min.
"""

# IMPORTS
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from river.utils import Rolling

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

from safeband.anomaly import ConditionalGaussianScorer
from safeband.fault_diagnosis import SensorFaultClassifier
from safeband.proba import MultivariateGaussian

logger = logging.getLogger(__name__)

# CONSTANTS
RANDOM_STATE = 42
SIGNALS = ["s0", "s1", "s2", "s3", "s4"]
BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"

N_TRAIN = 2000
N_WARMUP = 400
N_REGIME = 400
WINDOW = 25
LONG_WINDOW = 400
# Coordinated step in marginal-sigma units. Swept: a small shift is
# cancelled by the conditional regression at every rho, so the gradient
# only appears once the un-cancelled remainder 3*(1-sum(beta)) crosses
# the 3-sigma mean_threshold (at low rho the regression cancels little).
SHIFTS = [3.0, 6.0, 10.0]
RHOS = [0.1, 0.3, 0.5, 0.7, 0.9]
SCALES = [1.0, 5.0, math.inf]
SEEDS = [0, 1, 2, 3, 4]


def make_stream(rho: float, seed: int, n: int) -> list[dict[str, float]]:
    """Return ``n`` rows of a correlated-Gaussian stream.

    Each signal is ``sqrt(rho)*latent + sqrt(1-rho)*idiosyncratic`` so
    every pair has correlation ``rho`` and unit marginal variance.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    idio = rng.standard_normal((n, len(SIGNALS)))
    a = math.sqrt(rho)
    b = math.sqrt(1 - rho)
    return [
        {s: a * latent[t] + b * idio[t, i] for i, s in enumerate(SIGNALS)}
        for t in range(n)
    ]


def fit_scorer(rows: list[dict[str, float]]) -> ConditionalGaussianScorer:
    """Fit a read-only ConditionalGaussianScorer on the clean prefix."""
    scorer = ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=RANDOM_STATE), N_TRAIN),
        grace_period=200,
        protect_anomaly_detector=False,
    )
    for x in rows[:N_TRAIN]:
        scorer.learn_one(x)
    return scorer


def regime_fp(
    rows: list[dict[str, float]],
    scorer: ConditionalGaussianScorer,
    scale: float,
    shift: float,
) -> float:
    """Return the false-positive rate during a coordinated regime shift.

    A constant ``SHIFT`` is added to every signal after the warm-up; the
    correct label for every signal is ``normal``, so any non-``normal``
    label is a false positive. The change-point flag is forced True
    during the regime window (a coordinated shift is exactly when the
    system-wide change-point test fires), isolating mechanism 1
    (residual cancellation) at ``scale=1.0`` from the added effect of
    mechanism 2 (graded suppression) at higher scales.
    """
    clf = SensorFaultClassifier(
        window=WINDOW,
        long_window=LONG_WINDOW,
        suppress_threshold_scale=scale,
    )
    base = rows[N_TRAIN:]
    fp = 0
    total = 0
    for i, raw in enumerate(base):
        x = dict(raw)
        regime = i >= N_WARMUP
        if regime:
            for s in SIGNALS:
                x[s] += shift
        labels = clf.process_one(
            x, scorer.residuals_one(x), drift_detected=regime
        )
        if regime:
            for s in SIGNALS:
                total += 1
                fp += int(labels[s] != "normal")
    return fp / total if total else float("nan")


def main() -> None:
    """Sweep rho x suppression scale; write the FP-rate artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records: list[dict[str, object]] = []
    for rho in RHOS:
        # Fit one read-only scorer per (rho, seed) and reuse it across
        # suppression scales -- the scorer is identical across scales.
        fitted = []
        for seed in SEEDS:
            rows = make_stream(rho, seed, N_TRAIN + N_WARMUP + N_REGIME)
            fitted.append((rows, fit_scorer(rows)))
        for shift in SHIFTS:
            for scale in SCALES:
                rates = [
                    regime_fp(rows, sc, scale, shift) for rows, sc in fitted
                ]
                arr = np.array(rates)
                records.append(
                    {
                        "rho": rho,
                        "shift": shift,
                        "suppress_scale": (
                            "inf" if scale == math.inf else scale
                        ),
                        "fp_rate_mean": round(float(arr.mean()), 4),
                        "fp_rate_std": round(float(arr.std()), 4),
                        "n_seeds": len(SEEDS),
                    }
                )
                logger.info(
                    "rho=%.1f shift=%.0f scale=%s: FP=%.4f +/- %.4f",
                    rho,
                    shift,
                    records[-1]["suppress_scale"],
                    records[-1]["fp_rate_mean"],
                    records[-1]["fp_rate_std"],
                )

    table = pd.DataFrame(records)
    logger.info(
        "\n=== Regime-change false-positive rate ===\n%s",
        table.to_string(index=False),
    )
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(BENCH_DIR / "i7_regime_fp.csv", index=False)
    logger.info("\nArtifact written to %s", BENCH_DIR / "i7_regime_fp.csv")


if __name__ == "__main__":
    main()
