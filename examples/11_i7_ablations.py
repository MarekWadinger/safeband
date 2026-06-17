"""Defensibility ablations for the I7 classifier (opponent R3 / R5).

Two characterisations adversarial review flagged as must-haves:

**R3 -- the co-occurring-fault dead-band.** During a detected
change-point the per-signal thresholds are scaled up by
``suppress_threshold_scale`` (default 5), so a genuine single-sensor
fault that co-occurs with a regime change is only caught if its
normalized residual exceeds ``scale * mean_threshold``. This maps the
detection rate of a bias of magnitude ``b`` (in conditional sigma)
against the suppression scale, making the [mean_threshold,
scale*mean_threshold] dead-band explicit.

**R5 -- false freezing on a genuinely quiescent (not stuck) signal.**
The freeze test's "absolute" floor is ``freeze_abs_scale * freeze_eps *
running_std``, so it shrinks as a signal settles. A signal at true
steady state (small but non-zero variation) could trip it. This measures
the false-freeze rate as a healthy signal's live noise shrinks, swept
over the three freeze constants, and checks whether a per-signal
calibration helps.

Artifacts: ``i7_deadband.csv`` and ``i7_freeze_quiescent.csv``.

Run::

    uv run python examples/11_i7_ablations.py

CPU-only; ~3 min. Synthetic carriers for full control over correlation,
fault magnitude and steady-state noise.
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

from functions.anomaly import ConditionalGaussianScorer
from functions.fault_diagnosis import SensorFaultClassifier
from functions.proba import MultivariateGaussian

logger = logging.getLogger(__name__)

# CONSTANTS
RANDOM_STATE = 42
SIGNALS = ["s0", "s1", "s2", "s3", "s4"]
BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"

N_TRAIN = 2000
N_WARMUP = 400
N_FAULT = 400
WINDOW = 25
LONG_WINDOW = 400
TAIL = 150
MEAN_THRESHOLD = 3.0
RHO = 0.6  # fixed moderate correlation for the dead-band study
SEEDS = [0, 1, 2, 3, 4]


def make_stream(rho: float, seed: int, n: int) -> list[dict[str, float]]:
    """Return ``n`` rows of a unit-variance correlation-``rho`` stream."""
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


# R3 -- CO-OCCURRING-FAULT DEAD-BAND
# A constant bias is absorbed by the classifier's own long-window
# baseline (and by the conditional mean), so it is a poor probe for the
# suppression dead-band. A drift RAMP grows without bound and is not
# absorbed: under a raised threshold it is simply detected later, once it
# has grown past ``scale * mean_threshold``. The fault magnitude at
# detection therefore reads the dead-band ceiling directly.
def deadband_delay(
    rows: list[dict[str, float]],
    scorer: ConditionalGaussianScorer,
    slope: float,
    scale: float,
) -> tuple[int, float]:
    """Detection delay and fault magnitude (in sigma) at detection.

    A drift ramp of ``slope`` conditional-sigma per step is injected into
    ``target`` while the change-point flag is held True (the fault
    co-occurs with a regime change). Returns the step from onset to the
    first non-``normal`` label and the ramp magnitude (in sigma) reached
    at that step, or ``(NEVER, nan)`` if never detected in the window.
    """
    never = N_FAULT + 1
    target = "s0"
    clf = SensorFaultClassifier(
        window=WINDOW,
        long_window=LONG_WINDOW,
        mean_threshold=MEAN_THRESHOLD,
        suppress_threshold_scale=scale,
    )
    base = rows[N_TRAIN : N_TRAIN + N_WARMUP + N_FAULT]
    sigma = float(scorer.residuals_one(base[N_WARMUP])[target][1])
    for i, raw in enumerate(base):
        x = dict(raw)
        regime = i >= N_WARMUP
        if regime:
            t = i - N_WARMUP
            x[target] += slope * sigma * t
        out = clf.process_one(
            x, scorer.residuals_one(x), drift_detected=regime
        )
        if regime and out[target] != "normal":
            t = i - N_WARMUP
            return t, slope * t
    return never, float("nan")


def ablate_deadband() -> pd.DataFrame:
    """Map drift detection delay / magnitude-at-detection vs scale.

    A larger suppression scale forces the ramp to grow further before it
    clears the raised threshold -- the dead-band cost, in sigma.
    """
    slope = 0.2  # conditional-sigma per step
    scales = [1.0, 5.0, math.inf]
    fitted = [
        (r, fit_scorer(r))
        for r in (
            make_stream(RHO, s, N_TRAIN + N_WARMUP + N_FAULT) for s in SEEDS
        )
    ]
    records: list[dict[str, object]] = []
    for scale in scales:
        delays = []
        mags = []
        for r, sc in fitted:
            d, m = deadband_delay(r, sc, slope, scale)
            delays.append(d)
            mags.append(m)
        da = np.array(delays, dtype=float)
        ma = np.array(mags, dtype=float)
        detected = da <= N_FAULT
        records.append(
            {
                "suppress_scale": "inf" if scale == math.inf else scale,
                "drift_slope_sigma": slope,
                "detect_rate": round(float(detected.mean()), 3),
                "delay_median": (
                    float(np.median(da[detected]))
                    if detected.any()
                    else float("nan")
                ),
                "magnitude_sigma_at_detection": (
                    round(float(np.median(ma[detected])), 2)
                    if detected.any()
                    else float("nan")
                ),
                "n_seeds": len(fitted),
            }
        )
        logger.info(
            "scale=%s: detect=%.2f delay_med=%s mag_sigma=%s",
            records[-1]["suppress_scale"],
            records[-1]["detect_rate"],
            records[-1]["delay_median"],
            records[-1]["magnitude_sigma_at_detection"],
        )
    return pd.DataFrame(records)


# R5 -- FALSE FREEZING ON A QUIESCENT (NOT STUCK) SIGNAL
def false_freeze_rate(
    rows: list[dict[str, float]],
    scorer: ConditionalGaussianScorer,
    quiet_std: float,
    freeze_eps: float,
    freeze_var_ratio: float,
    freeze_abs_scale: float,
) -> float:
    """Fraction of a genuinely quiet (not stuck) window labelled frozen.

    The target signal is replaced by its mean plus live Gaussian noise of
    std ``quiet_std`` (a true steady state, never constant), so any
    ``freezing`` label is a false positive.
    """
    target = "s0"
    clf = SensorFaultClassifier(
        window=WINDOW,
        long_window=LONG_WINDOW,
        freeze_eps=freeze_eps,
        freeze_var_ratio=freeze_var_ratio,
        freeze_abs_scale=freeze_abs_scale,
    )
    base = rows[N_TRAIN : N_TRAIN + N_WARMUP + N_FAULT]
    level = base[N_WARMUP][target]
    rng = np.random.default_rng(RANDOM_STATE)
    fp = 0
    total = 0
    for i, raw in enumerate(base):
        x = dict(raw)
        quiet = i >= N_WARMUP
        if quiet:
            x[target] = level + quiet_std * float(rng.standard_normal())
        out = clf.process_one(
            x, scorer.residuals_one(x), scorer.drift_detected
        )
        if quiet:
            total += 1
            fp += int(out[target] == "freezing")
    return fp / total if total else float("nan")


def ablate_freeze_quiescent() -> pd.DataFrame:
    """Sweep steady-state noise vs the three freeze constants."""
    fitted = [
        (r, fit_scorer(r))
        for r in (
            make_stream(RHO, s, N_TRAIN + N_WARMUP + N_FAULT) for s in SEEDS
        )
    ]
    # quiet_std as a fraction of the signal's unit marginal std.
    quiet_stds = [0.5, 0.1, 0.02, 0.005]
    # (freeze_eps, freeze_var_ratio, freeze_abs_scale): the default plus
    # a stricter and a looser setting.
    configs = [
        ("default", 1e-3, 1e-2, 20.0),
        ("strict", 1e-4, 5e-3, 5.0),
        ("loose", 5e-3, 5e-2, 50.0),
    ]
    records: list[dict[str, object]] = []
    for name, eps, vr, absc in configs:
        for q in quiet_stds:
            rates = [
                false_freeze_rate(r, sc, q, eps, vr, absc) for r, sc in fitted
            ]
            arr = np.array(rates)
            records.append(
                {
                    "config": name,
                    "freeze_eps": eps,
                    "freeze_var_ratio": vr,
                    "freeze_abs_scale": absc,
                    "quiet_std": q,
                    "false_freeze_mean": round(float(arr.mean()), 4),
                    "false_freeze_std": round(float(arr.std()), 4),
                    "n_seeds": len(fitted),
                }
            )
            logger.info(
                "%s quiet_std=%.3f: false_freeze=%.4f",
                name,
                q,
                records[-1]["false_freeze_mean"],
            )
    return pd.DataFrame(records)


# MAIN
def main() -> None:
    """Run both ablations and write the CSV artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== R3: co-occurring-fault dead-band ===")
    deadband = ablate_deadband()
    logger.info("\n%s", deadband.to_string(index=False))

    logger.info("\n=== R5: false freezing on a quiescent signal ===")
    quiescent = ablate_freeze_quiescent()
    logger.info("\n%s", quiescent.to_string(index=False))

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    deadband.to_csv(BENCH_DIR / "i7_deadband.csv", index=False)
    quiescent.to_csv(BENCH_DIR / "i7_freeze_quiescent.csv", index=False)
    logger.info("\nArtifacts written under %s", BENCH_DIR)


if __name__ == "__main__":
    main()
