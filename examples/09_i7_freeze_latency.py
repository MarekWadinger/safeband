"""Freeze detection latency: I7 classifier vs the published CDF detector.

Adversarial review (defense opponent) showed the freezing "blind-spot"
claim is overstated: a frozen sensor is only structurally invisible to
the published ``ConditionalGaussianScorer`` in the degenerate
``cond_std -> 0`` case. In a correlated system the peers keep moving, so
the conditional mean ``E[x_i | x_rest]`` slews away from the stuck value
and the published per-signal CDF test (anomaly when the conditional CDF
score is below ``alpha`` or above ``1 - alpha``, i.e. |residual| > ~3
conditional sigma) flags the freeze on its own.

This script measures, per channel and across the coupling range, whether
the I7 freeze test detects a stuck sensor EARLIER than the published CDF
detector, or merely redundantly. Both detectors read the SAME read-only
scorer state (fit once on the clean prefix, not updated during the fault
window), so the comparison is fair. The honest hypothesis: the freeze
test is redundant on strongly-coupled channels (the conditional mean
slews, so the CDF catches it fast) but is the ONLY detector on
weakly-coupled / near-isolated channels (the conditional mean stays near
the frozen value, so the CDF never fires) — which is exactly the regime
a univariate stuck-at test exists for.

Artifact: ``examples/benchmarks/i7_freeze_latency.csv`` -- per channel,
the freeze-test and CDF detection rates and median delays, the fraction
of trials where the freeze test is strictly earlier, and the fraction
where the CDF detector never fires within the window.

Run::

    uv run python examples/09_i7_freeze_latency.py

CPU-only; ~10 min.
"""

# IMPORTS
import logging
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
# Ordered weakly- to strongly-coupled (ascending conditional sigma, from
# the scaled validation): bed2/bed1 are near-isolated, cso1/bfo2/bso1 are
# strongly predicted by their peers.
CATS_SIGNALS = ["bso1", "cso1", "bfo2", "bed1", "bed2"]
CATS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "multivariate"
    / "cats"
    / "data_1t_agg_last.csv"
)
BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"

N_TRAIN = 3000
N_WARMUP = 400
N_FAULT = 400
TRIAL = N_WARMUP + N_FAULT
WINDOW = 25
LONG_WINDOW = 400
NEVER = N_FAULT + 1  # sentinel delay for "never detected within window"


def load_clean_cats() -> list[dict[str, float]]:
    """Return the CATS nominal block (rows before the first anomaly)."""
    need = N_TRAIN + 60 * TRIAL
    df = pd.read_csv(CATS_PATH, index_col=0, nrows=need + 20000)
    anomalies = df["y"].to_numpy() > 0
    first_anom = int(np.argmax(anomalies)) if anomalies.any() else len(df)
    clean = df.iloc[:first_anom][CATS_SIGNALS].reset_index(drop=True)
    logger.info("carrier: CATS nominal block, %s clean rows", len(clean))
    return [
        {str(k): float(v) for k, v in row.items()}
        for _, row in clean.iterrows()
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


def freeze_trial(
    rows: list[dict[str, float]],
    scorer: ConditionalGaussianScorer,
    target: str,
    offset: int,
) -> tuple[int, int]:
    """Freeze ``target`` at one onset; return (freeze_delay, cdf_delay).

    Each delay is the step from onset to first detection by, respectively,
    the I7 freeze test and the published per-signal CDF test, or ``NEVER``
    if that detector does not fire within the fault window.
    """
    clf = SensorFaultClassifier(window=WINDOW, long_window=LONG_WINDOW)
    start = N_TRAIN + offset
    base = rows[start : start + TRIAL]
    frozen_at = base[N_WARMUP][target]
    freeze_delay = NEVER
    cdf_delay = NEVER
    for i, raw in enumerate(base):
        x = dict(raw)
        if i >= N_WARMUP:
            x[target] = frozen_at
        scores = scorer.scores_one(x)
        labels = clf.process_one(
            x, scorer.residuals_one(x), scorer.drift_detected
        )
        if i >= N_WARMUP:
            t = i - N_WARMUP
            if freeze_delay == NEVER and labels[target] == "freezing":
                freeze_delay = t
            flagged = (
                scores[target] < scorer.alpha
                or scores[target] > 1 - scorer.alpha
            )
            if cdf_delay == NEVER and flagged:
                cdf_delay = t
    return freeze_delay, cdf_delay


def conditional_sigma(
    rows: list[dict[str, float]],
    scorer: ConditionalGaussianScorer,
) -> dict[str, float]:
    """Mean per-signal conditional sigma over a slice of clean rows."""
    sums = dict.fromkeys(CATS_SIGNALS, 0.0)
    sample = rows[N_TRAIN : N_TRAIN + 200]
    for x in sample:
        res = scorer.residuals_one(x)
        for s in CATS_SIGNALS:
            sums[s] += float(res[s][1])
    return {s: sums[s] / len(sample) for s in CATS_SIGNALS}


def main() -> None:
    """Run the freeze-latency comparison and write the CSV artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not CATS_PATH.exists():
        logger.error("CATS carrier absent at %s -- aborting", CATS_PATH)
        sys.exit(1)
    rows = load_clean_cats()
    scorer = fit_scorer(rows)
    sigmas = conditional_sigma(rows, scorer)
    spare = len(rows) - N_TRAIN - TRIAL
    offsets = list(range(0, spare, TRIAL))
    logger.info("%s onsets per channel", len(offsets))

    records: list[dict[str, object]] = []
    for target in sorted(CATS_SIGNALS, key=lambda s: sigmas[s]):
        fz = []
        cd = []
        for off in offsets:
            f, c = freeze_trial(rows, scorer, target, off)
            fz.append(f)
            cd.append(c)
        fz_a = np.array(fz)
        cd_a = np.array(cd)
        n = len(offsets)
        records.append(
            {
                "signal": target,
                "cond_sigma": round(sigmas[target], 4),
                "freeze_detect_rate": round(float((fz_a < NEVER).mean()), 3),
                "cdf_detect_rate": round(float((cd_a < NEVER).mean()), 3),
                "freeze_delay_median": (
                    float(np.median(fz_a[fz_a < NEVER]))
                    if (fz_a < NEVER).any()
                    else float("nan")
                ),
                "cdf_delay_median": (
                    float(np.median(cd_a[cd_a < NEVER]))
                    if (cd_a < NEVER).any()
                    else float("nan")
                ),
                "frac_freeze_earlier": round(float((fz_a < cd_a).mean()), 3),
                "frac_cdf_never": round(float((cd_a == NEVER).mean()), 3),
                "n": n,
            }
        )
        logger.info(
            "%s (sigma=%.2f): freeze_rate=%.2f cdf_rate=%.2f "
            "freeze_earlier=%.2f cdf_never=%.2f",
            target,
            sigmas[target],
            records[-1]["freeze_detect_rate"],
            records[-1]["cdf_detect_rate"],
            records[-1]["frac_freeze_earlier"],
            records[-1]["frac_cdf_never"],
        )

    table = pd.DataFrame(records)
    logger.info(
        "\n=== Freeze latency: I7 vs published CDF ===\n%s",
        table.to_string(index=False),
    )
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(BENCH_DIR / "i7_freeze_latency.csv", index=False)
    logger.info(
        "\nArtifact written to %s", BENCH_DIR / "i7_freeze_latency.csv"
    )


if __name__ == "__main__":
    main()
