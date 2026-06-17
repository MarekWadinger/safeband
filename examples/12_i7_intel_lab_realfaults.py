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

Artifacts: ``i7_intel_lab_summary.csv`` (per-mote + pooled rates) and
``i7_intel_lab_fault_types.csv`` (assigned type distribution on the real
fault span).

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


def score_mote(
    g: pd.DataFrame,
) -> tuple[int, int, int, int, Counter[FaultLabel]]:
    """Return (healthy_fp, healthy_n, fault_hit, fault_n, type_counts).

    The scorer runs in its intended ONLINE adaptive mode: at each step it
    yields residuals (predict), the classifier labels, then the scorer
    learns from the sample -- with ``protect_anomaly_detector=True`` it
    adapts to legitimate non-stationary drift (Intel-Lab is diurnal) but
    gates out anomalies, so it does not absorb the fault. A read-only
    model fit on a short prefix instead false-alarms on healthy drift
    (verified: ~0.97 healthy FP), which is why adaptation is essential
    on real non-stationary data.

    A healthy false positive is any non-``normal`` label on a healthy
    row; a real-fault hit is any non-``normal`` label on a temp/humidity
    signal during the battery-depleted span; ``type_counts`` tallies the
    assigned fault types on the faulted temp/humidity readings.
    """
    sig = {s: g[s].to_numpy(dtype=float) for s in SIGNALS}
    volt = g["voltage"].to_numpy(dtype=float)
    scorer = ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=RANDOM_STATE), N_FIT),
        grace_period=200,
        protect_anomaly_detector=True,
    )
    # Strict freeze config (recommended by the quiescent-signal ablation).
    clf = SensorFaultClassifier(
        window=25,
        long_window=400,
        freeze_eps=1e-4,
        freeze_var_ratio=5e-3,
        freeze_abs_scale=5.0,
    )
    healthy_fp = healthy_n = fault_hit = fault_n = 0
    types: Counter[FaultLabel] = Counter()
    for i in range(len(volt)):
        x = {s: float(sig[s][i]) for s in SIGNALS}
        # predict (residuals) -> classify -> adapt (gated learn).
        labels = clf.process_one(
            x, scorer.residuals_one(x), scorer.drift_detected
        )
        scorer.learn_one(x)
        if i < N_FIT:
            continue  # skip the online warm-up
        v = float(volt[i])
        if v >= HEALTHY_V:
            healthy_n += 1
            healthy_fp += int(any(labels[s] != "normal" for s in SIGNALS))
        elif v < FAULT_V:
            fault_n += 1
            faulted = [
                labels[s]
                for s in ("temp", "humidity")
                if labels[s] != "normal"
            ]
            fault_hit += int(bool(faulted))
            types.update(faulted)
    return healthy_fp, healthy_n, fault_hit, fault_n, types


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

    rows: list[dict[str, object]] = []
    pooled: Counter[FaultLabel] = Counter()
    tot_hfp = tot_hn = tot_fh = tot_fn = 0
    for mote, g in motes.items():
        hfp, hn, fh, fn, types = score_mote(g)
        tot_hfp += hfp
        tot_hn += hn
        tot_fh += fh
        tot_fn += fn
        pooled.update(types)
        rows.append(
            {
                "mote": mote,
                "healthy_n": hn,
                "healthy_fp_rate": round(hfp / hn, 4) if hn else float("nan"),
                "fault_n": fn,
                "fault_detect_rate": (
                    round(fh / fn, 4) if fn else float("nan")
                ),
            }
        )
        logger.info(
            "mote %s: healthy_fp=%.4f (n=%d)  fault_detect=%.4f (n=%d)",
            mote,
            rows[-1]["healthy_fp_rate"],
            hn,
            rows[-1]["fault_detect_rate"],
            fn,
        )

    rows.append(
        {
            "mote": "POOLED",
            "healthy_n": tot_hn,
            "healthy_fp_rate": round(tot_hfp / tot_hn, 4) if tot_hn else None,
            "fault_n": tot_fn,
            "fault_detect_rate": round(tot_fh / tot_fn, 4) if tot_fn else None,
        }
    )
    summary = pd.DataFrame(rows)
    types_df = pd.DataFrame(
        [
            {
                "fault_type": k,
                "count": v,
                "frac": round(v / sum(pooled.values()), 4),
            }
            for k, v in pooled.most_common()
        ]
    )
    logger.info(
        "\n=== Intel-Lab real-fault validation ===\n%s",
        summary.to_string(index=False),
    )
    logger.info(
        "\n=== Assigned type on real (battery-depletion) faults ===\n%s",
        types_df.to_string(index=False),
    )

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(BENCH_DIR / "i7_intel_lab_summary.csv", index=False)
    types_df.to_csv(BENCH_DIR / "i7_intel_lab_fault_types.csv", index=False)
    logger.info("\nArtifacts written under %s", BENCH_DIR)


if __name__ == "__main__":
    main()
