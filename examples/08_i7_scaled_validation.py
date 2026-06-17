"""Scaled, CI-backed validation of the sensor-fault taxonomy (IDEAS I7).

This extends ``07_fault_diagnosis_validation.py`` from a pilot (one trial
per signal, N=5 per type) to a publication-grade evaluation: each
(fault, signal) pair is injected over many non-overlapping onset windows
sliding across the CATS nominal block, yielding N~=80 trials per fault
type with Wilson 95% confidence intervals on recall.

It also characterises the headline limitation honestly. The pilot found
bias recall ~0.4 and the discussion attributed it to the conditional
model *absorbing* a constant offset on high-variance correlated channels
(the offset is partially explained away as the rolling Gaussian
re-estimates the cross-signal relation), so bias is isolable only on
low-conditional-sigma channels. This script makes that claim falsifiable
by reporting **bias recall per channel against the channel's conditional
sigma**: if the absorption hypothesis holds, recall must fall as
conditional sigma rises, monotonically -- a defensible characterised
limitation rather than an unexplained weak number.

Artifacts (written under ``examples/benchmarks/``):

* ``i7_scaled_per_type.csv`` -- per-type recall, Wilson 95% CI, N.
* ``i7_scaled_bias_by_channel.csv`` -- per-channel bias recall vs the
  channel's conditional sigma (the absorption characterisation).
* ``i7_scaled_normal_precision.csv`` -- healthy-peer (cross-talk)
  mislabel breakdown bounding normal precision.

Run::

    uv run python examples/08_i7_scaled_validation.py

CPU-only; ~30 min (320 trials x 800 steps). Intended to be launched in
the background.
"""

# IMPORTS
import logging
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from river.utils import Rolling

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

from functions.anomaly import ConditionalGaussianScorer
from functions.fault_diagnosis import FaultLabel, SensorFaultClassifier
from functions.proba import MultivariateGaussian

logger = logging.getLogger(__name__)

# CONSTANTS
RANDOM_STATE = 42
FAULTS: list[FaultLabel] = ["bias", "drift", "accuracy_loss", "freezing"]
# Same correlated, dup-free CATS channels as the pilot. The first three
# are high-variance (raw std ~10-16); the last two are low-variance
# (~0.7, 0.3) -- the contrast that exposes bias absorption.
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
TAIL = 150
# Severities in the reliably-detectable regime (from the pilot sweep), so
# the per-channel recall reflects the conditional-model interaction, not
# a too-weak injection.
B_BIAS = 4.0
R_DRIFT = 0.05
K_ACC = 4.0
Z = 1.96  # 95% normal quantile for the Wilson interval


# CARRIER STREAM
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


# INJECTION
def inject(
    x: dict[str, float],
    target: str,
    fault: str,
    t: int,
    sigma: float,
    frozen_at: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Return a copy of ``x`` with ``fault`` injected into ``target``."""
    x = dict(x)
    if fault == "bias":
        x[target] += B_BIAS * sigma
    elif fault == "drift":
        x[target] += R_DRIFT * sigma * t
    elif fault == "accuracy_loss":
        x[target] += K_ACC * sigma * float(rng.standard_normal())
    elif fault == "freezing":
        x[target] = frozen_at
    return x


def trial(
    rows: list[dict[str, float]],
    scorer: ConditionalGaussianScorer,
    target: str,
    fault: str,
    offset: int,
) -> tuple[FaultLabel, dict[str, FaultLabel]]:
    """Run one injection trial at a sliding onset; return steady labels.

    Returns the steady-state label of the injected signal and, for each
    healthy peer, its steady-state label (for the cross-talk count).
    """
    clf = SensorFaultClassifier(
        window=WINDOW,
        long_window=LONG_WINDOW,
        mean_threshold=3.0,
        trend_threshold=1.0,
        var_ratio=4.0,
    )
    rng = np.random.default_rng(RANDOM_STATE + offset)
    start = N_TRAIN + offset
    base = rows[start : start + TRIAL]
    sigma = float(scorer.residuals_one(base[N_WARMUP])[target][1])
    frozen_at = base[N_WARMUP][target]
    tgt: list[FaultLabel] = []
    peers: dict[str, list[FaultLabel]] = {
        s: [] for s in CATS_SIGNALS if s != target
    }
    for i, raw in enumerate(base):
        if i >= N_WARMUP:
            x = inject(raw, target, fault, i - N_WARMUP, sigma, frozen_at, rng)
        else:
            x = dict(raw)
        labels = clf.process_one(
            x, scorer.residuals_one(x), scorer.drift_detected
        )
        if i >= N_WARMUP:
            tgt.append(labels[target])
            for s, series in peers.items():
                series.append(labels[s])
    steady_peers = {s: _steady(series) for s, series in peers.items()}
    return _steady(tgt), steady_peers


def _steady(labels: list[FaultLabel]) -> FaultLabel:
    """Majority label over the steady-state tail of the fault window."""
    return Counter(labels[-TAIL:]).most_common(1)[0][0]


def _wilson(hits: int, n: int) -> tuple[float, float, float]:
    """Return (point, lo, hi) recall with a Wilson 95% interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = hits / n
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = (Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


# EVALUATION
def evaluate(
    rows: list[dict[str, float]],
    scorer: ConditionalGaussianScorer,
    sigmas: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all scaled trials; return per-type, bias-by-channel, peer tables.

    Offsets slide non-overlapping windows across the spare clean rows so
    every trial sees an independent stretch of the carrier.
    """
    spare = len(rows) - N_TRAIN - TRIAL
    offsets = list(range(0, spare, TRIAL))
    logger.info(
        "%s onset windows x %s signals per fault",
        len(offsets),
        len(CATS_SIGNALS),
    )

    # hits[fault] = correct steady labels; chan[fault][sig] = per-signal.
    hits: dict[str, int] = dict.fromkeys(FAULTS, 0)
    totals: dict[str, int] = dict.fromkeys(FAULTS, 0)
    chan: dict[str, dict[str, list[int]]] = {
        f: {s: [] for s in CATS_SIGNALS} for f in FAULTS
    }
    peer_counts: Counter[FaultLabel] = Counter()
    peer_total = 0
    for fault in FAULTS:
        for target in CATS_SIGNALS:
            for off in offsets:
                steady, peers = trial(rows, scorer, target, fault, off)
                ok = int(steady == fault)
                hits[fault] += ok
                totals[fault] += 1
                chan[fault][target].append(ok)
                for plabel in peers.values():
                    peer_counts[plabel] += 1
                    peer_total += 1
        logger.info(
            "%s done: recall=%.3f (N=%s)",
            fault,
            hits[fault] / totals[fault],
            totals[fault],
        )

    per_type = pd.DataFrame(
        [
            {
                "type": f,
                "recall": round(_wilson(hits[f], totals[f])[0], 3),
                "ci_lo": round(_wilson(hits[f], totals[f])[1], 3),
                "ci_hi": round(_wilson(hits[f], totals[f])[2], 3),
                "n": totals[f],
            }
            for f in FAULTS
        ]
    )

    bias_by_channel = pd.DataFrame(
        [
            {
                "signal": s,
                "cond_sigma": round(sigmas[s], 4),
                "bias_recall": round(
                    _wilson(sum(chan["bias"][s]), len(chan["bias"][s]))[0], 3
                ),
                "ci_lo": round(
                    _wilson(sum(chan["bias"][s]), len(chan["bias"][s]))[1], 3
                ),
                "ci_hi": round(
                    _wilson(sum(chan["bias"][s]), len(chan["bias"][s]))[2], 3
                ),
                "n": len(chan["bias"][s]),
            }
            for s in sorted(CATS_SIGNALS, key=lambda s: sigmas[s])
        ]
    )

    peer = pd.DataFrame(
        [
            {
                "peer_label": lbl,
                "count": peer_counts[lbl],
                "frac": round(peer_counts[lbl] / peer_total, 4),
            }
            for lbl in sorted(peer_counts, key=lambda k: -peer_counts[k])
        ]
    )
    return per_type, bias_by_channel, peer


# MAIN
def main() -> None:
    """Run the scaled I7 validation and write the CSV artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not CATS_PATH.exists():
        logger.error("CATS carrier absent at %s -- aborting", CATS_PATH)
        sys.exit(1)
    rows = load_clean_cats()
    scorer = fit_scorer(rows)
    sigmas = conditional_sigma(rows, scorer)
    logger.info(
        "conditional sigma per signal: %s",
        {k: round(v, 3) for k, v in sigmas.items()},
    )

    per_type, bias_by_channel, peer = evaluate(rows, scorer, sigmas)
    logger.info(
        "\n=== Per-type recall (Wilson 95%% CI) ===\n%s",
        per_type.to_string(index=False),
    )
    logger.info(
        "\n=== Bias recall vs conditional sigma ===\n%s",
        bias_by_channel.to_string(index=False),
    )
    logger.info(
        "\n=== Healthy-peer label distribution (cross-talk) ===\n%s",
        peer.to_string(index=False),
    )

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    per_type.to_csv(BENCH_DIR / "i7_scaled_per_type.csv", index=False)
    bias_by_channel.to_csv(
        BENCH_DIR / "i7_scaled_bias_by_channel.csv", index=False
    )
    peer.to_csv(BENCH_DIR / "i7_scaled_normal_precision.csv", index=False)
    logger.info("\nArtifacts written under %s", BENCH_DIR)


if __name__ == "__main__":
    main()
