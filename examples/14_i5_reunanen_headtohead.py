"""Detection head-to-head: base detector vs Reunanen (2020) on SKAB.

The fault-typing layer of this paper sits on top of the base
self-supervised conditional-Gaussian detector. To establish that the
detector it extends is competitive, we run it head-to-head against the
streaming autoencoder detector of Reunanen et al. (2020) -- a published
one-pass, no-window outlier detector -- on the SKAB multivariate
benchmark (Skoltech Anomaly Benchmark; 8 sensors, binary anomaly label).

Both detectors run online, one observation at a time, predicting before
learning. We score every point and report the threshold-free ROC AUC
against the anomaly label, plus the best achievable F1 over a threshold
sweep. ROC AUC is computed by the rank identity (Mann-Whitney U), so no
extra dependency is needed.

Artifact: ``examples/benchmarks/i5_reunanen_skab.csv``.

Run::

    uv run python examples/14_i5_reunanen_headtohead.py

CPU-only; ~2 min.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from river.utils import Rolling

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

from safeband.anomaly import ConditionalGaussianScorer
from safeband.proba import MultivariateGaussian
from safeband.reunanen import ReunanenScorer

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
SKAB = (
    Path(__file__).resolve().parent
    / "data"
    / "multivariate"
    / "alldata_skab.csv"
)
BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"
WINDOW = 1000  # rolling window for the base conditional model
GRACE = 200


def load_skab() -> tuple[list[dict[str, float]], np.ndarray]:
    """Return SKAB rows as feature dicts and the binary anomaly label."""
    df = pd.read_csv(SKAB)
    label_col = "anomaly"
    feats = [
        c
        for c in df.columns
        if c not in ("datetime", "anomaly", "changepoint")
    ]
    df = df.dropna(subset=[*feats, label_col])
    rows = [{c: float(r[c]) for c in feats} for _, r in df.iterrows()]
    return rows, df[label_col].to_numpy(dtype=int)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank identity (Mann-Whitney U)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / counts)[inv]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[labels == 1].sum()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Best F1 over a sweep of score thresholds."""
    qs = np.quantile(scores, np.linspace(0.5, 0.999, 60))
    best = 0.0
    for thr in np.unique(qs):
        pred = scores >= thr
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        denom = 2 * tp + fp + fn
        if denom:
            best = max(best, 2 * tp / denom)
    return best


def score_base(rows: list[dict[str, float]]) -> np.ndarray:
    """Online anomaly scores from the base conditional-Gaussian detector.

    The conditional CDF score is mapped to a distance-from-centre anomaly
    score ``2|s - 0.5|`` in ``[0, 1]`` (higher is more anomalous).
    """
    scorer = ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=RANDOM_STATE), WINDOW),
        grace_period=GRACE,
        protect_anomaly_detector=False,
    )
    out = np.empty(len(rows))
    for i, x in enumerate(rows):
        out[i] = abs(scorer.score_one(x) - 0.5) * 2.0
        scorer.learn_one(x)
    return out


def score_reunanen(rows: list[dict[str, float]]) -> np.ndarray:
    """Online anomaly scores from the Reunanen streaming autoencoder."""
    scorer = ReunanenScorer()
    out = np.empty(len(rows))
    for i, x in enumerate(rows):
        out[i] = scorer.score_one(x)
        scorer.learn_one(x)
    return out


def main() -> None:
    """Run the SKAB detection head-to-head; write the artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not SKAB.exists():
        logger.error("SKAB data absent at %s -- aborting", SKAB)
        sys.exit(1)
    rows, labels = load_skab()
    logger.info(
        "SKAB: %d points, %d anomalies (%.1f%%)",
        len(labels),
        int(labels.sum()),
        100 * labels.mean(),
    )
    records = []
    for name, fn in (
        ("base (conditional Gaussian)", score_base),
        ("Reunanen 2020 (autoencoder)", score_reunanen),
    ):
        scores = fn(rows)
        auc = roc_auc(scores, labels)
        f1 = best_f1(scores, labels)
        records.append(
            {
                "detector": name,
                "roc_auc": round(auc, 4),
                "best_f1": round(f1, 4),
            }
        )
        logger.info("%-30s ROC-AUC=%.4f  best-F1=%.4f", name, auc, f1)

    table = pd.DataFrame(records)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(BENCH_DIR / "i5_reunanen_skab.csv", index=False)
    logger.info("\nArtifact written to %s", BENCH_DIR / "i5_reunanen_skab.csv")


if __name__ == "__main__":
    main()
