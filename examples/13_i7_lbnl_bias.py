"""I7 bias validation on a real labelled sensor-bias dataset (LBNL FDD).

Adversarial review asked for at least one fault TYPE validated on
genuine labelled-fault data rather than only controlled injection. The
LBNL building-FDD MZVAV-1 set (Granderson & Lin 2019; PNNL large-office
AHU model, 1-min cadence) is exactly an outdoor-air-temperature sensor
bias dataset: a constant offset of +/-1/2/4 C is imposed on the
outdoor-air-temperature reading over labelled date windows, with a
per-timestamp ``Fault Detection Ground Truth`` column. Because the file
contains only outdoor-air-temp bias scenarios, ground truth = 1 means
the outdoor-air-temperature sensor is biased.

This is a demanding real-bias case: the imposed offset (1.8-7.2 F) is
small relative to the outdoor signal's natural weather swing (marginal
std ~20 F), so it tests bias isolation at realistic, unfavourable
magnitudes on a highly non-stationary signal -- the regime the
controlled CATS study and the bias-coupling characterization predict is
hard.

The detector runs in its online adaptive mode over the four coupled AHU
air temperatures (supply, outdoor, mixed, return); the fault column is
held out as the label. We report, over a mean-threshold sweep, the
bias-typing recall on the outdoor-air-temperature signal during faulted
periods and the healthy false-positive rate on the unfaulted periods.

Artifact: ``examples/benchmarks/i7_lbnl_bias.csv``.

Run::

    uv run python examples/13_i7_lbnl_bias.py

Requires ``.temp/data/realfaults/lbnl/MZVAV-1.csv`` (LBNL OEDI submission
910). CPU-only; ~2 min.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import pandas as pd
from river.utils import Rolling

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

from functions.anomaly import ConditionalGaussianScorer
from functions.fault_diagnosis import SensorFaultClassifier
from functions.proba import MultivariateGaussian

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
SIGNALS = [
    "AHU: Supply Air Temperature",
    "AHU: Outdoor Air Temperature",
    "AHU: Mixed Air Temperature",
    "AHU: Return Air Temperature",
]
TARGET = "AHU: Outdoor Air Temperature"  # the biased sensor
GT = "Fault Detection Ground Truth"
DATA = (
    Path(__file__).resolve().parent.parent
    / ".temp"
    / "data"
    / "realfaults"
    / "lbnl"
    / "MZVAV-1.csv"
)
BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"

N_FIT = 1440  # one day of 1-min samples to warm up the online model
MEAN_THRESHOLDS = [2.0, 3.0, 4.0, 5.0]

Step = tuple[dict[str, float], dict[str, tuple[float, float]], bool, int]


def load() -> pd.DataFrame:
    """Return MZVAV-1 sorted chronologically with the needed columns."""
    df = pd.read_csv(DATA)
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df.dropna(subset=[*SIGNALS, GT])


def scorer_pass(df: pd.DataFrame) -> list[Step]:
    """One online adaptive scorer pass; cache per-step inputs."""
    sig = {s: df[s].to_numpy(dtype=float) for s in SIGNALS}
    gt = df[GT].to_numpy(dtype=int)
    scorer = ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=RANDOM_STATE), N_FIT),
        grace_period=200,
        protect_anomaly_detector=True,
    )
    cache: list[Step] = []
    for i in range(len(gt)):
        x = {s: float(sig[s][i]) for s in SIGNALS}
        res = scorer.residuals_one(x)
        cache.append((x, res, scorer.drift_detected, int(gt[i])))
        scorer.learn_one(x)
    return cache


def classify_pass(
    cache: list[Step], mean_threshold: float
) -> dict[str, float]:
    """Run the classifier over a cached pass at one threshold."""
    clf = SensorFaultClassifier(
        window=25,
        long_window=400,
        freeze_eps=1e-4,
        freeze_var_ratio=5e-3,
        freeze_abs_scale=5.0,
        mean_threshold=mean_threshold,
    )
    h_fp = h_n = f_bias = f_any = f_n = 0
    for i, (x, res, drift, gt) in enumerate(cache):
        labels = clf.process_one(x, res, drift)
        if i < N_FIT:
            continue
        if gt == 0:
            h_n += 1
            h_fp += int(any(labels[s] != "normal" for s in SIGNALS))
        else:
            f_n += 1
            f_any += int(labels[TARGET] != "normal")
            f_bias += int(labels[TARGET] == "bias")
    return {
        "mean_threshold": mean_threshold,
        "healthy_fp_rate": round(h_fp / h_n, 4) if h_n else float("nan"),
        "bias_detect_rate": round(f_any / f_n, 4) if f_n else float("nan"),
        "bias_typed_rate": round(f_bias / f_n, 4) if f_n else float("nan"),
        "healthy_n": h_n,
        "fault_n": f_n,
    }


def cond_sigma(cache: list[Step]) -> float:
    """Mean conditional sigma of the target over all healthy steps.

    Skips the grace-period steps whose residual scale is not yet defined
    (non-finite), so the estimate reflects the settled model.
    """
    vals = [
        res[TARGET][1]
        for _, res, _, gt in cache
        if gt == 0 and math.isfinite(res[TARGET][1]) and res[TARGET][1] > 0
    ]
    return sum(vals) / len(vals) if vals else math.nan


def main() -> None:
    """Run the LBNL outdoor-air-temp bias validation; write the artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not DATA.exists():
        logger.error("LBNL MZVAV-1 absent at %s -- aborting", DATA)
        sys.exit(1)
    df = load()
    logger.info(
        "MZVAV-1: %d rows, %d faulted, %d healthy",
        len(df),
        int((df[GT] == 1).sum()),
        int((df[GT] == 0).sum()),
    )
    cache = scorer_pass(df)
    cs = cond_sigma(cache)
    # Imposed offsets are +/-1/2/4 C = 1.8/3.6/7.2 F; express in
    # conditional-sigma units, the scale the mean-threshold test sees.
    logger.info(
        "outdoor-air-temp conditional sigma (healthy, F): %.3f", cs
    )
    logger.info(
        "imposed bias in conditional sigma: 1C=%.2f, 2C=%.2f, 4C=%.2f",
        1.8 / cs,
        3.6 / cs,
        7.2 / cs,
    )
    rows = [classify_pass(cache, mt) for mt in MEAN_THRESHOLDS]
    table = pd.DataFrame(rows)
    logger.info(
        "\n=== LBNL outdoor-air-temp bias (real labelled fault) ===\n%s",
        table.to_string(index=False),
    )
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(BENCH_DIR / "i7_lbnl_bias.csv", index=False)
    logger.info("\nArtifact written to %s", BENCH_DIR / "i7_lbnl_bias.csv")


if __name__ == "__main__":
    main()
