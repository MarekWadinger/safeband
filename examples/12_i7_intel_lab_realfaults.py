"""I7 validation on REAL sensor faults: Intel Berkeley Lab.

Adversarial review's first demand was validation on genuine faults, not
only injected ones. The Intel Berkeley Lab deployment (54 motes,
2004; temperature / humidity / light / voltage at ~31 s cadence) is a
real, noisy, correlated multivariate stream with a well-known *real*
fault mode: as a mote's battery depletes (voltage falls below ~2.3 V)
its temperature and humidity readings degrade catastrophically --
temperatures of 100-122 C and negative humidity. Battery voltage is a
physically-grounded fault proxy we never feed to the detector, so it
serves as an external ground-truth label.

This script runs the detector in its intended ONLINE adaptive mode over
the environmental signals temp / humidity / light (voltage held out as
the label only): at each step the read residuals feed the classifier,
then the scorer learns the sample with ``protect_anomaly_detector=True``
so it tracks legitimate non-stationary drift (Intel-Lab is strongly
diurnal) while gating out the fault. A static model fit on a short
prefix instead false-alarms on healthy drift (~0.97 healthy FP), which
is itself the finding that adaptation is essential on real
non-stationary data. It reports, per mote and pooled, the FALSE-POSITIVE
rate on the healthy span (the opponent's healthy-data check) and the
REAL-fault detection rate and assigned fault type on the
battery-depleted span (voltage < 2.3 V). The strict freeze configuration
(``freeze_eps=1e-4``) -- recommended by the quiescent-signal ablation --
is used throughout.

A ``mean_threshold`` sweep traces the operating curve: raising it from 3
to 8 sigma cuts the healthy FP rate from ~0.22 to ~0.07 while real-fault
detection barely moves (~0.93 -> ~0.90), because the real faults are
tens of sigma while the healthy diurnal ramps the two-signal conditional
model cannot fully cancel are only a few sigma.

Artifacts: ``i7_intel_lab_operating_points.csv`` (FP / detect rate per
mean_threshold) and ``i7_intel_lab_fault_types.csv`` (assigned type
distribution on the real fault span).

Run::

    uv run python examples/12_i7_intel_lab_realfaults.py

Requires ``.temp/data/realfaults/intel_lab.txt.gz`` (downloaded from
https://db.csail.mit.edu/labdata/data.txt.gz). CPU-only; ~3 min.
"""

# IMPORTS
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from river.utils import Rolling

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

from functions.anomaly import ConditionalGaussianScorer
from functions.fault_diagnosis import FaultLabel, SensorFaultClassifier
from functions.proba import MultivariateGaussian

logger = logging.getLogger(__name__)

# CONSTANTS
RANDOM_STATE = 42
# temp + humidity: two smooth, physically anti-correlated environmental
# signals that both degrade in the battery-depletion fault. Light is
# excluded -- its violent diurnal swing (zero at night) breaks the
# Gaussian conditional model and trips the freeze test on healthy night
# data, and it is not part of the fault signature.
SIGNALS = ["temp", "humidity"]
DATA = (
    Path(__file__).resolve().parent.parent
    / ".temp"
    / "data"
    / "realfaults"
    / "intel_lab.txt.gz"
)
BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"

HEALTHY_V = 2.4  # voltage at/above which the mote is considered healthy
FAULT_V = 2.3  # voltage below which battery-depletion faults appear
# Plausible physical ranges; rows outside are dropped as parse garbage,
# NOT as faults (the fault shows as extreme but finite readings).
TEMP_RANGE = (-40.0, 130.0)
HUM_RANGE = (-10000.0, 100.0)
MIN_HEALTHY = 800  # min healthy rows to fit a mote
MIN_FAULT = 100  # min fault rows to score a mote
N_FIT = 600  # healthy prefix used to fit the scorer


def load_motes() -> dict[int, pd.DataFrame]:
    """Return per-mote chronological frames with enough healthy+fault data."""
    # The file always has these 8 physical columns regardless of which
    # signals are analysed; SIGNALS selects a subset downstream.
    cols = [
        "date",
        "time",
        "epoch",
        "moteid",
        "temp",
        "humidity",
        "light",
        "voltage",
    ]
    df = pd.read_csv(
        DATA,
        sep=r"\s+",
        names=cols,
        na_values=[""],
        on_bad_lines="skip",
        engine="c",
    )
    df = df.dropna(subset=["moteid", "voltage", *SIGNALS])
    df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["ts"])
    # Drop unphysical parse garbage but keep the finite extreme faults.
    df = df[df["temp"].between(*TEMP_RANGE)]
    df = df[df["humidity"].between(*HUM_RANGE)]
    out: dict[int, pd.DataFrame] = {}
    for mote_val in df["moteid"].unique():
        grp = df[df["moteid"] == mote_val]
        gs = grp.sort_values("ts")
        healthy = (gs["voltage"] >= HEALTHY_V).sum()
        fault = (gs["voltage"] < FAULT_V).sum()
        if healthy >= MIN_HEALTHY + N_FIT and fault >= MIN_FAULT:
            out[int(mote_val)] = gs.reset_index(drop=True)
    logger.info("usable motes: %s", sorted(out))
    return out


Step = tuple[dict[str, float], dict[str, tuple[float, float]], bool, float]


def scorer_pass(g: pd.DataFrame) -> list[Step]:
    """One ONLINE adaptive scorer pass; cache per-step inputs.

    The scorer runs in its intended online mode -- at each step it yields
    residuals (predict), then learns the sample with
    ``protect_anomaly_detector=True`` so it tracks legitimate
    non-stationary drift (Intel-Lab is diurnal) while gating out the
    fault. A static model fit on a short prefix instead false-alarms on
    healthy drift (~0.97 healthy FP), which is why adaptation is
    essential on real non-stationary data. Returns, per step, the raw
    observation, the conditional residuals, the change-point flag and the
    voltage, so the lightweight classifier can be re-run at several
    thresholds without repeating this expensive pass.
    """
    sig = {s: g[s].to_numpy(dtype=float) for s in SIGNALS}
    volt = g["voltage"].to_numpy(dtype=float)
    scorer = ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=RANDOM_STATE), N_FIT),
        grace_period=200,
        protect_anomaly_detector=True,
    )
    cache: list[Step] = []
    for i in range(len(volt)):
        x = {s: float(sig[s][i]) for s in SIGNALS}
        res = scorer.residuals_one(x)
        cache.append((x, res, scorer.drift_detected, float(volt[i])))
        scorer.learn_one(x)
    return cache


def classify_pass(
    cache: list[Step],
    mean_threshold: float,
) -> tuple[int, int, int, int, Counter[FaultLabel], Counter[FaultLabel]]:
    """Run the classifier over a cached scorer pass at one threshold.

    Returns (healthy_fp, healthy_n, fault_hit, fault_n, fault_types,
    healthy_fp_types). A healthy false positive is any non-``normal``
    label on a healthy row (voltage >= ``HEALTHY_V``); a real-fault hit
    is any non-``normal`` temp/humidity label on the battery-depleted
    span (voltage < ``FAULT_V``).
    """
    # Strict freeze config (recommended by the quiescent-signal ablation).
    clf = SensorFaultClassifier(
        window=25,
        long_window=400,
        freeze_eps=1e-4,
        freeze_var_ratio=5e-3,
        freeze_abs_scale=5.0,
        mean_threshold=mean_threshold,
    )
    healthy_fp = healthy_n = fault_hit = fault_n = 0
    types: Counter[FaultLabel] = Counter()
    hp_types: Counter[FaultLabel] = Counter()
    for i, (x, res, drift, v) in enumerate(cache):
        labels = clf.process_one(x, res, drift)
        if i < N_FIT:
            continue  # skip the online warm-up
        if v >= HEALTHY_V:
            healthy_n += 1
            fp_labels = [labels[s] for s in SIGNALS if labels[s] != "normal"]
            healthy_fp += int(bool(fp_labels))
            hp_types.update(fp_labels)
        elif v < FAULT_V:
            fault_n += 1
            faulted = [
                labels[s]
                for s in ("temp", "humidity")
                if labels[s] != "normal"
            ]
            fault_hit += int(bool(faulted))
            types.update(faulted)
    return healthy_fp, healthy_n, fault_hit, fault_n, types, hp_types


# mean_threshold operating points. The real faults are tens of sigma, so
# raising the threshold separates them from the few-sigma healthy diurnal
# ramps that leak into drift/bias labels when only two signals are
# available to cancel the common-mode drift.
MEAN_THRESHOLDS = [3.0, 4.0, 5.0, 6.0, 8.0]
REPORT_AT = 6.0  # threshold for the per-type breakdown artifacts


def main() -> None:
    """Run the Intel-Lab real-fault validation and write artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not DATA.exists():
        logger.error("Intel-Lab data absent at %s -- aborting", DATA)
        sys.exit(1)
    motes = load_motes()
    if not motes:
        logger.error("no usable motes found")
        sys.exit(1)

    # Pooled counters per threshold, plus per-type breakdowns at REPORT_AT.
    pool = {
        mt: {"hfp": 0, "hn": 0, "fh": 0, "fn": 0} for mt in MEAN_THRESHOLDS
    }
    pooled_types: Counter[FaultLabel] = Counter()
    pooled_hp: Counter[FaultLabel] = Counter()
    for mote, g in motes.items():
        cache = scorer_pass(g)  # one expensive online pass per mote
        for mt in MEAN_THRESHOLDS:
            hfp, hn, fh, fn, types, hp_types = classify_pass(cache, mt)
            pool[mt]["hfp"] += hfp
            pool[mt]["hn"] += hn
            pool[mt]["fh"] += fh
            pool[mt]["fn"] += fn
            if mt == REPORT_AT:
                pooled_types.update(types)
                pooled_hp.update(hp_types)
        logger.info("mote %s scored (n=%d)", mote, len(cache))

    op_rows = [
        {
            "mean_threshold": mt,
            "healthy_fp_rate": round(p["hfp"] / p["hn"], 4)
            if p["hn"]
            else None,
            "fault_detect_rate": round(p["fh"] / p["fn"], 4)
            if p["fn"]
            else None,
            "healthy_n": p["hn"],
            "fault_n": p["fn"],
        }
        for mt, p in pool.items()
    ]
    op = pd.DataFrame(op_rows)
    types_df = pd.DataFrame(
        [
            {
                "fault_type": k,
                "count": v,
                "frac": round(v / (sum(pooled_types.values()) or 1), 4),
            }
            for k, v in pooled_types.most_common()
        ]
    )
    logger.info(
        "\n=== Intel-Lab operating points (mean_threshold sweep) ===\n%s",
        op.to_string(index=False),
    )
    logger.info(
        "\n=== Assigned fault type at mean_threshold=%s ===\n%s",
        REPORT_AT,
        types_df.to_string(index=False),
    )
    hp_total = sum(pooled_hp.values()) or 1
    logger.info(
        "\n=== Healthy FP composition at mean_threshold=%s ===", REPORT_AT
    )
    for k, v in pooled_hp.most_common():
        logger.info("  %-14s %8d  %.3f", k, v, v / hp_total)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    op.to_csv(BENCH_DIR / "i7_intel_lab_operating_points.csv", index=False)
    types_df.to_csv(BENCH_DIR / "i7_intel_lab_fault_types.csv", index=False)
    logger.info("\nArtifacts written under %s", BENCH_DIR)


if __name__ == "__main__":
    main()
