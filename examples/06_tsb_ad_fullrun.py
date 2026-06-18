"""Resumable TSB-AD full-split benchmark for streaming AID detectors.

Evaluates the Adaptive Interpretable Detector (AID) family under the
strict one-pass *predict-then-learn* regime against the TSB-AD
(NeurIPS 2024 Datasets & Benchmarks) univariate and multivariate
evaluation splits, alongside a tuned ``ReunanenScorer`` baseline and the
z-score / random reference floors.

Headline metric is VUS-PR (the metric the TSB-AD paper identifies as
most reliable). Hyperparameters for AID and Reunanen are tuned on the
official TUNING split via Bayesian optimisation with an identical
budget; all methods are then evaluated on the EVA split. Per-series
continuous scores are cached as ``.npz`` so the run is fully resumable.

Usage::

    # Smoke test (3 U + 3 M series, end-to-end, minutes):
    uv run python examples/06_tsb_ad_fullrun.py --split both --subset 3 \
        --out examples/benchmarks/tsb_ad_fullrun_smoke.csv

    # Full sweep (350 U + 180 M series, HOURS -- run detached):
    uv run python examples/06_tsb_ad_fullrun.py --split both

Install the metrics-only TSB-AD stack first (no torch needed)::

    uv pip install --no-deps TSB-AD==1.5
    uv pip install statsmodels

Notes:
    This script is the authoritative source of the full-split numbers.
    The companion notebook ``06_tsb_ad_benchmark.ipynb`` is a pilot on a
    small representative subset.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
import urllib.request
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# AID lives in the repo's functions package; ensure it is importable when
# the script is launched from the examples/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
from river.proba import Gaussian
from river.utils import Rolling

from functions.anomaly import (
    ConditionalGaussianScorer,
    GaussianScorer,
)
from functions.proba import MultivariateGaussian
from functions.reunanen import ReunanenScorer

# Silence sklearn's noisy per-window precision/recall warnings at MODULE
# scope so the filter is also installed in every joblib (loky) worker
# process -- workers re-import this module but never run main(), so a
# filter set inside main() would not reach them and the warnings would
# flood from the parallel tuning. This affects only logging, not numbers.
warnings.filterwarnings("ignore")

try:
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
except ImportError as exc:  # pragma: no cover - environment guard
    _MSG = (
        "TSB-AD metrics not installed. Run:\n"
        "  uv pip install --no-deps TSB-AD==1.5\n"
        "  uv pip install statsmodels"
    )
    raise RuntimeError(_MSG) from exc

logger = logging.getLogger("tsb_ad_fullrun")


class _StreamScorer(Protocol):
    """The streaming detector interface used by the scoring loops."""

    def score_one(self, x: float | dict[str, float]) -> float: ...
    def predict_one(self, x: float | dict[str, float]) -> int: ...
    def learn_one(self, x: float | dict[str, float]) -> object: ...


RANDOM_STATE = 42
# Parallel workers for the tuning objective's per-series scoring. -1 uses
# all cores; loky pins each worker's inner BLAS threads to 1, keeping the
# result deterministic and order-identical to the serial computation.
N_JOBS = -1

# Repo-relative paths (resolved against this file, not the CWD).
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / ".temp" / "data"
FILE_LIST_DIR = DATA_DIR / "File_List"
SCORE_CACHE = ROOT / ".temp" / "tsb_ad_scores"
TUNE_CACHE = ROOT / ".temp" / "tsb_ad_tuning"
DEFAULT_OUT = ROOT / "examples" / "benchmarks" / "tsb_ad_fullrun.csv"

RAW_BASE = "https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/main"
_LIST = f"{RAW_BASE}/Datasets/File_List"
_EVAL = f"{RAW_BASE}/benchmark_exp/benchmark_eval_results"
FILE_LIST_URLS = {
    "TSB-AD-U-Eva.csv": f"{_LIST}/TSB-AD-U-Eva.csv",
    "TSB-AD-U-Tuning.csv": f"{_LIST}/TSB-AD-U-Tuning.csv",
    "TSB-AD-M-Eva.csv": f"{_LIST}/TSB-AD-M-Eva.csv",
    "TSB-AD-M-Tuning.csv": f"{_LIST}/TSB-AD-M-Tuning.csv",
    "uni_mergedTable_VUS-PR.csv": f"{_EVAL}/uni_mergedTable_VUS-PR.csv",
    "multi_mergedTable_VUS-PR.csv": f"{_EVAL}/multi_mergedTable_VUS-PR.csv",
}

# Headline metrics kept from the TSB-AD metric suite.
METRIC_KEYS = ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC", "Standard-F1")

# Metadata columns in the published merged tables that are not algorithms.
_META_COLS = {
    "ts_len",
    "anomaly_len",
    "num_anomaly",
    "avg_anomaly_len",
    "anomaly_ratio",
    "point_anomaly",
    "seq_anomaly",
}


def fetch_file_list(name: str) -> Path:
    """Return a cached File_List CSV, downloading it from GitHub if absent."""
    dest = FILE_LIST_DIR / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching %s", name)
        urllib.request.urlretrieve(FILE_LIST_URLS[name], dest)  # noqa: S310
    return dest


def read_split(
    split: Literal["U", "M"],
    kind: Literal["Eva", "Tuning"],
) -> list[str]:
    """Read the official series file names for a split and stage."""
    path = fetch_file_list(f"TSB-AD-{split}-{kind}.csv")
    return pd.read_csv(path)["file_name"].tolist()


def load_series(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a TSB-AD CSV into (data, label, feature_columns)."""
    frame = pd.read_csv(path).dropna()
    columns = list(frame.columns[:-1])
    data = frame.iloc[:, :-1].to_numpy(dtype=float)
    label = frame["Label"].astype(int).to_numpy()
    return data, label, columns


def train_prefix(name: str) -> int:
    """Parse the ``tr_XXXX`` training-prefix length from a file name."""
    try:
        return int(name.split(".", maxsplit=1)[0].split("_")[-3])
    except (ValueError, IndexError):
        return 500


# --------------------------------------------------------------------------
# Streaming detectors and reference baselines
# --------------------------------------------------------------------------
def stream_scores(
    model: _StreamScorer,
    data: np.ndarray,
    columns: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one predict-then-learn pass; return score, online preds, runtime.

    Mirrors the honest streaming protocol: at time ``t`` the detector has
    seen only ``[0, t]``. The raw CDF score is mapped to a two-sided
    anomaly magnitude ``2 * |cdf - 0.5|`` (higher = more anomalous).
    """
    raw: list[float] = []
    preds: list[int] = []
    start = time.perf_counter()
    for row in data:
        x = (
            dict(zip(columns, row, strict=True))
            if columns is not None
            else float(row[0])
        )
        # river's Gaussian can produce a complex sigma when its running
        # variance dips slightly negative numerically; neutralise such a
        # point (and any non-finite score) instead of failing the whole
        # series, which would otherwise spuriously score it zero.
        try:
            score_val = float(model.score_one(x))
            if not np.isfinite(score_val):
                score_val = 0.5
        except (TypeError, ValueError):
            score_val = 0.5
        try:
            pred_val = int(model.predict_one(x))
        except (TypeError, ValueError):
            pred_val = 0
        raw.append(score_val)
        preds.append(pred_val)
        with contextlib.suppress(TypeError, ValueError):
            model.learn_one(x)
    runtime = time.perf_counter() - start
    score = 2 * np.abs(np.asarray(raw, dtype=float) - 0.5)
    return score, np.asarray(preds, dtype=bool), runtime


def stream_reunanen(
    model: ReunanenScorer,
    data: np.ndarray,
    columns: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One predict-then-learn pass for the Reunanen autoencoder detector.

    ``ReunanenScorer`` already returns a non-negative reconstruction cost
    (higher = more anomalous), so no two-sided transform is applied.
    """
    names = columns if columns is not None else ["x0"]
    raw: list[float] = []
    preds: list[int] = []
    start = time.perf_counter()
    for row in data:
        x = {name: float(row[j]) for j, name in enumerate(names)}
        # river's Gaussian can produce a complex sigma when its running
        # variance dips slightly negative numerically; neutralise such a
        # point (and any non-finite score) instead of failing the whole
        # series, which would otherwise spuriously score it zero.
        try:
            score_val = float(model.score_one(x))
            if not np.isfinite(score_val):
                score_val = 0.5
        except (TypeError, ValueError):
            score_val = 0.5
        try:
            pred_val = int(model.predict_one(x))
        except (TypeError, ValueError):
            pred_val = 0
        raw.append(score_val)
        preds.append(pred_val)
        with contextlib.suppress(TypeError, ValueError):
            model.learn_one(x)
    runtime = time.perf_counter() - start
    return np.asarray(raw, dtype=float), np.asarray(preds, dtype=bool), runtime


def zscore_baseline(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-pass online z-score on the first channel (a streaming floor)."""
    x = data[:, 0]
    n = np.arange(1, len(x) + 1)
    mean = np.cumsum(x) / n
    var = np.cumsum(x**2) / n - mean**2
    std = np.sqrt(np.maximum(var, 0.0))
    prev_mean = np.concatenate([[x[0]], mean[:-1]])
    prev_std = np.concatenate([[1.0], std[:-1]])
    score = np.abs(x - prev_mean) / (prev_std + 1e-12)
    pred = score > 3.0
    return score, pred


def random_baseline(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random scores (the absolute floor)."""
    rng = np.random.default_rng(RANDOM_STATE)
    score = rng.random(n)
    return score, score > 0.5


def evaluate_scores(
    score: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    sliding_window: int,
) -> dict[str, float]:
    """Min-max normalise the score and run the TSB-AD metric suite."""
    span = score.max() - score.min()
    score = (score - score.min()) / (span + 1e-12)
    version = "opt_mem" if len(label) > 100_000 else "opt"
    res = get_metrics(
        score,
        label,
        slidingWindow=sliding_window,
        pred=pred,
        version=version,
    )
    return {k: float(res[k]) for k in METRIC_KEYS}


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------
def make_aid_univariate(
    tr: int,
    window_mult: float,
    threshold: float,
) -> GaussianScorer:
    """AID univariate: rolling Gaussian, window = window_mult * tr."""
    window = max(50, int(window_mult * tr))
    return GaussianScorer(
        Rolling(Gaussian(), window_size=window),
        grace_period=min(tr, window),
        threshold=threshold,
    )


def make_aid_multivariate(
    tr: int,
    window_mult: float,
    threshold: float,
) -> ConditionalGaussianScorer:
    """AID multivariate: rolling conditional Gaussian."""
    window = max(50, int(window_mult * tr))
    return ConditionalGaussianScorer(
        Rolling(MultivariateGaussian(seed=RANDOM_STATE), window_size=window),
        grace_period=min(tr, window),
        threshold=threshold,
    )


def make_reunanen(n_hidden: float, lr: float, k: float) -> ReunanenScorer:
    """Reunanen autoencoder detector with tunable capacity / threshold."""
    return ReunanenScorer(
        n_hidden=max(2, round(n_hidden)),
        lr=lr,
        k=k,
        M=2000,
        seed=RANDOM_STATE,
    )


# --------------------------------------------------------------------------
# Hyperparameter tuning on the TUNING split (maximise mean VUS-PR)
# --------------------------------------------------------------------------
def _score_tuning_series(
    make_model: Callable[..., object],
    stream_fn: Callable,
    needs_tr: bool,
    use_columns: bool,
    params: dict[str, float],
    item: tuple[np.ndarray, np.ndarray, list[str], int, int],
) -> float:
    """VUS-PR of one tuning series for a candidate HP set (0.0 on error).

    Each series is scored with a fresh, deterministically seeded model and
    is independent of the others, so this is safe to run in a worker
    process; the caller aggregates in series order.
    """
    data, label, columns, sliding_window, tr = item
    try:
        model = make_model(tr, **params) if needs_tr else make_model(**params)
        cols = columns if use_columns else None
        score, pred, _ = stream_fn(model, data, cols)
        res = evaluate_scores(score, label, pred, sliding_window)
        return float(res["VUS-PR"])
    except Exception:
        logger.exception("tuning eval failed on a series; scoring 0")
        return 0.0


def _tuning_objective(
    make_model: Callable[..., object],
    stream_fn: Callable,
    series: Sequence[tuple[np.ndarray, np.ndarray, list[str], int, int]],
    needs_tr: bool,
    use_columns: bool,
    **params: float,
) -> float:
    """Mean VUS-PR over the tuning series for a candidate HP set.

    The per-series scorings are independent and run in parallel across
    cores (``N_JOBS``). joblib's loky backend pins each worker's inner
    math threads to one, and results are collected in series order, so
    the mean is identical to the serial computation -- and thus to the
    hyperparameters the score caches were built with. ``use_columns``
    mirrors the evaluation regime: univariate AID consumes bare floats
    (``columns=None``); multivariate AID and Reunanen build a feature
    dict from the columns.
    """
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_score_tuning_series)(
            make_model, stream_fn, needs_tr, use_columns, dict(params), item
        )
        for item in series
    )
    return float(np.mean(results)) if results else 0.0


def tune(
    name: str,
    make_model: Callable[..., object],
    stream_fn: Callable,
    pbounds: dict[str, tuple[float, float]],
    tuning_series: Sequence,
    needs_tr: bool,
    use_columns: bool,
    init_points: int,
    n_iter: int,
) -> dict[str, float]:
    """Bayesian-optimise hyperparameters; return the best parameter dict."""
    logger.info(
        "Tuning %s on %d series (%d+%d evals)",
        name,
        len(tuning_series),
        init_points,
        n_iter,
    )
    objective = partial(
        _tuning_objective,
        make_model,
        stream_fn,
        tuning_series,
        needs_tr,
        use_columns,
    )
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        verbose=1,
        random_state=RANDOM_STATE,
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best = optimizer.max
    assert best is not None
    logger.info(
        "%s best VUS-PR=%.4f params=%s",
        name,
        best["target"],
        best["params"],
    )
    return dict(best["params"])


def prepare_tuning_series(
    split: Literal["U", "M"],
    limit: int | None,
    max_len: int,
) -> list[tuple[np.ndarray, np.ndarray, list[str], int, int]]:
    """Load the tuning split into ready-to-score tuples (cached in memory).

    Tuning runs many full predict-then-learn passes per series; for the
    multivariate split the per-row conditional-MVN CDF makes long, wide
    series prohibitively slow. We therefore cap each tuning series to a
    window of at most ``max_len`` rows for hyperparameter search only -- the
    EVA evaluation always uses the full, untruncated series. The window is
    chosen to contain the first anomaly so TSB-AD's range-AUC has a
    non-empty positive sequence (it indexes the first anomaly run directly).
    """
    files = read_split(split, "Tuning")
    if limit is not None:
        files = files[:limit]
    out = []
    sub = "TSB-AD-U" if split == "U" else "TSB-AD-M"
    for name in files:
        path = DATA_DIR / sub / name
        if not path.exists():
            logger.warning("tuning series missing, skipping: %s", name)
            continue
        try:
            data, label, columns = load_series(path)
            if len(label) > max_len:
                # Slice a bounded window of <= max_len rows that contains
                # the first anomaly (offset by a quarter-window of context).
                anomalies = np.flatnonzero(label)
                start = 0
                if anomalies.size and anomalies[0] >= max_len:
                    start = max(0, int(anomalies[0]) - max_len // 4)
                end = min(len(label), start + max_len)
                data, label = data[start:end], label[start:end]
            if not label.any():
                logger.warning(
                    "tuning series has no anomaly, skipping: %s", name
                )
                continue
            sliding_window = int(find_length_rank(data[:, :1], rank=1))
            out.append(
                (data, label, columns, sliding_window, train_prefix(name)),
            )
        except Exception:
            logger.exception("failed to load tuning series %s", name)
    return out


# --------------------------------------------------------------------------
# Per-series benchmark with score caching
# --------------------------------------------------------------------------
def cached_scores(
    method: str,
    split: str,
    series: str,
    compute: Callable[[], tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return cached (score, pred, runtime) or compute and cache them."""
    SCORE_CACHE.mkdir(parents=True, exist_ok=True)
    stem = f"{split}__{method}__{series}".replace("/", "_")
    npz = SCORE_CACHE / f"{stem}.npz"
    if npz.exists():
        with np.load(npz) as d:
            return d["score"], d["pred"], float(d["runtime"])
    score, pred, runtime = compute()
    np.savez(npz, score=score, pred=pred, runtime=np.asarray(runtime))
    return score, pred, runtime


def bench_file(
    path: Path,
    split: Literal["U", "M"],
    aid_factory: Callable[[int], _StreamScorer],
    reunanen_params: dict[str, float],
) -> list[dict[str, object]]:
    """Benchmark AID, Reunanen, z-score and random on one series."""
    data, label, columns = load_series(path)
    sliding_window = int(find_length_rank(data[:, :1], rank=1))
    tr = train_prefix(path.name)
    multivariate = split == "M"
    cols = columns if multivariate else None
    nan = float("nan")

    aid_score, aid_pred, aid_rt = cached_scores(
        "AID",
        split,
        path.name,
        lambda: stream_scores(aid_factory(tr), data, cols),
    )
    re_score, re_pred, re_rt = cached_scores(
        "Reunanen",
        split,
        path.name,
        lambda: stream_reunanen(make_reunanen(**reunanen_params), data, cols),
    )
    z_score, z_pred, _ = cached_scores(
        "Z-score",
        split,
        path.name,
        lambda: (*zscore_baseline(data), nan),
    )
    r_score, r_pred, _ = cached_scores(
        "Random",
        split,
        path.name,
        lambda: (*random_baseline(len(label)), nan),
    )

    candidates = (
        ("AID", aid_score, aid_pred, aid_rt),
        ("Reunanen", re_score, re_pred, re_rt),
        ("Z-score", z_score, z_pred, nan),
        ("Random", r_score, r_pred, nan),
    )
    rows: list[dict[str, object]] = []
    for method, s, p, rt in candidates:
        try:
            metrics = evaluate_scores(s, label, p, sliding_window)
        except Exception:
            # TSB-AD's range-AUC can fail on degenerate series; record NaN
            # for this method rather than dropping the whole series.
            logger.exception("metrics failed for %s on %s", method, path.name)
            metrics = dict.fromkeys(METRIC_KEYS, nan)
        rows.append(
            {
                "split": split,
                "series": path.name,
                "method": method,
                "n": len(label),
                "slidingWindow": sliding_window,
                "runtime_s": round(rt, 2) if not np.isnan(rt) else nan,
                **metrics,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Post-hoc analysis: bootstrap CIs and win/loss/tie counts
# --------------------------------------------------------------------------
def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def win_loss_tie(
    a: pd.Series,
    b: pd.Series,
    tol: float = 1e-6,
) -> tuple[int, int, int]:
    """Count series where a beats / loses to / ties b on a shared index."""
    common = a.index.intersection(b.index)
    diff = a.loc[common] - b.loc[common]
    wins = int((diff > tol).sum())
    losses = int((diff < -tol).sum())
    ties = int((diff.abs() <= tol).sum())
    return wins, losses, ties


def analyse(results: pd.DataFrame, splits: Sequence[str]) -> None:
    """Log bootstrap CIs and win/loss/tie tables (vs z-score, published)."""
    logger.info(
        "\n%s\nBOOTSTRAP 95%% CIs ON MEAN VUS-PR\n%s",
        "=" * 70,
        "=" * 70,
    )
    for split in splits:
        sub = results[results["split"] == split]
        for method in ("AID", "Reunanen", "Z-score", "Random"):
            vals = sub[sub["method"] == method]["VUS-PR"].to_numpy()
            mean, lo, hi = bootstrap_ci(vals)
            logger.info(
                "%s %-10s n=%3d  VUS-PR mean=%.4f  95%% CI [%.4f, %.4f]",
                split,
                method,
                len(vals),
                mean,
                lo,
                hi,
            )

    logger.info(
        "\n%s\nWIN/LOSS/TIE vs Z-SCORE (per series, VUS-PR)\n%s",
        "=" * 70,
        "=" * 70,
    )
    for split in splits:
        sub = results[results["split"] == split]
        z = sub[sub["method"] == "Z-score"].set_index("series")["VUS-PR"]
        for method in ("AID", "Reunanen"):
            m = sub[sub["method"] == method].set_index("series")["VUS-PR"]
            w, lose, tie = win_loss_tie(m, z)
            logger.info(
                "%s %-10s vs Z-score: W/L/T = %d/%d/%d",
                split,
                method,
                w,
                lose,
                tie,
            )

    _analyse_vs_published(results, splits)


def _analyse_vs_published(
    results: pd.DataFrame, splits: Sequence[str]
) -> None:
    """Compare AID's per-series VUS-PR to the published TSB-AD leaderboard."""
    logger.info(
        "\n%s\nAID vs PUBLISHED LEADERBOARD (per series, VUS-PR)\n%s",
        "=" * 70,
        "=" * 70,
    )
    table_name = {
        "U": "uni_mergedTable_VUS-PR.csv",
        "M": "multi_mergedTable_VUS-PR.csv",
    }
    for split in splits:
        try:
            pub = pd.read_csv(
                fetch_file_list(table_name[split]),
            ).set_index("file")
        except Exception:
            logger.exception("could not load published table for %s", split)
            continue
        algo_cols = [
            c
            for c in pub.columns
            if pub[c].dtype.kind == "f" and c not in _META_COLS
        ]
        sub = results[
            (results["split"] == split) & (results["method"] == "AID")
        ]
        aid = sub.set_index("series")["VUS-PR"]
        common = aid.index.intersection(pub.index)
        if len(common) == 0:
            logger.info("%s: no overlap with published table", split)
            continue
        aid_c = aid.loc[common]
        ranks = np.array(
            [
                1
                + int(
                    (
                        pub.loc[s, algo_cols].to_numpy(dtype=float) > aid_c[s]
                    ).sum(),
                )
                for s in common
            ]
        )
        n_algos = len(algo_cols) + 1
        logger.info(
            "%s AID vs %d published algos on %d series: "
            "mean rank %.1f/%d, top-1 on %d, top-5 on %d",
            split,
            len(algo_cols),
            len(common),
            ranks.mean(),
            n_algos,
            int((ranks == 1).sum()),
            int((ranks <= 5).sum()),
        )
        best_pub = pub.loc[common, algo_cols].max(axis=1)
        w, lose, tie = win_loss_tie(aid_c, best_pub)
        logger.info(
            "%s AID vs BEST-published: W/L/T = %d/%d/%d",
            split,
            w,
            lose,
            tie,
        )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
# Tuning search spaces (compact: window multiplier, threshold, AE capacity).
AID_PBOUNDS = {"window_mult": (1.0, 6.0), "threshold": (0.90, 0.99994)}
REUNANEN_PBOUNDS = {"n_hidden": (2, 10), "lr": (0.02, 0.3), "k": (2.0, 4.0)}


def cached_tune(
    key: str, run: Callable[[], dict[str, float]]
) -> dict[str, float]:
    """Return tuned params from the JSON cache, or tune once and cache.

    Bayesian optimisation is the single most expensive non-resumable step;
    caching its result to disk means a restart never re-tunes.
    """
    TUNE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = TUNE_CACHE / f"{key}.json"
    if cache.exists():
        params = json.loads(cache.read_text())
        logger.info("loaded cached tuning %s: %s", key, params)
        return params
    params = run()
    cache.write_text(json.dumps(params))
    return params


def already_done(out: Path, split: str) -> set[str]:
    """Return the series already written to ``out`` for ``split``."""
    if not out.exists():
        return set()
    try:
        prev = pd.read_csv(out, usecols=["split", "series"])
    except Exception:
        logger.exception("could not read resume CSV %s; starting fresh", out)
        return set()
    return set(prev.loc[prev["split"] == split, "series"].unique())


def append_rows(out: Path, rows: list[dict[str, object]]) -> None:
    """Append one series' rows to ``out`` and flush -- a crash-safe point.

    Writing after every series means a failure at hour 5 loses at most the
    series in flight, never the accumulated results.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, mode="a", header=not out.exists(), index=False)


def run_split(
    split: Literal["U", "M"],
    subset: int | None,
    init_points: int,
    n_iter: int,
    max_tuning_len: int,
    out: Path,
) -> list[dict[str, object]]:
    """Tune on the tuning split, then benchmark the eva split.

    Fully resumable: tuned hyperparameters are cached to disk and the
    per-series result rows are appended to ``out`` as they are produced,
    so a crash at any point loses at most the series in flight.
    """
    sub = "TSB-AD-U" if split == "U" else "TSB-AD-M"
    is_uni = split == "U"
    aid_make = make_aid_univariate if is_uni else make_aid_multivariate

    # --- TUNE (cached; tuning series prepared lazily only on a cache miss).
    _tuning: list | None = None

    def series() -> Sequence:
        nonlocal _tuning
        if _tuning is None:
            tune_limit = max(2, subset) if subset is not None else None
            _tuning = prepare_tuning_series(split, tune_limit, max_tuning_len)
        return _tuning

    # Cache tuned params only for a full run; a --subset smoke uses a
    # reduced tuning budget and must not poison the real cache.
    def tuned(
        key: str, runner: Callable[[], dict[str, float]]
    ) -> dict[str, float]:
        return runner() if subset is not None else cached_tune(key, runner)

    aid_params = tuned(
        f"AID-{split}",
        lambda: tune(
            f"AID-{split}",
            aid_make,
            stream_scores,
            AID_PBOUNDS,
            series(),
            needs_tr=True,
            use_columns=not is_uni,
            init_points=init_points,
            n_iter=n_iter,
        ),
    )
    reunanen_params = tuned(
        f"Reunanen-{split}",
        lambda: tune(
            f"Reunanen-{split}",
            make_reunanen,
            stream_reunanen,
            REUNANEN_PBOUNDS,
            series(),
            needs_tr=False,
            use_columns=True,
            init_points=init_points,
            n_iter=n_iter,
        ),
    )

    def aid_factory(tr: int) -> _StreamScorer:
        return aid_make(tr, **aid_params)

    # --- EVALUATE (on the eva split; resume past already-written series).
    files = read_split(split, "Eva")
    if subset is not None:
        files = files[:subset]
    done = already_done(out, split)
    logger.info(
        "Evaluating %s-Eva: %d series, %d already done, %d to do",
        split,
        len(files),
        len(done),
        len(files) - len(done),
    )

    records: list[dict[str, object]] = []
    for i, name in enumerate(files, 1):
        if name in done:
            continue
        path = DATA_DIR / sub / name
        tag = f"[{i}/{len(files)}]"
        if not path.exists():
            logger.warning("%s missing, skipping: %s", tag, name)
            continue
        t0 = time.perf_counter()
        try:
            rows = bench_file(path, split, aid_factory, reunanen_params)
            append_rows(out, rows)  # crash-safe checkpoint per series
            records += rows
            aid_vus = next(r["VUS-PR"] for r in rows if r["method"] == "AID")
            logger.info(
                "%s %s n=%d AID VUS-PR=%.3f (%.1fs)",
                tag,
                name,
                rows[0]["n"],
                aid_vus,
                time.perf_counter() - t0,
            )
        except Exception:
            logger.exception("%s FAILED, skipping: %s", tag, name)
    return records


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the requested splits, write CSV, print analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["U", "M", "both"], default="both")
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Run only the first N series of each split (smoke test).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--init-points",
        type=int,
        default=8,
        help="Bayesian-opt random init points (per method/split).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=24,
        help="Bayesian-opt guided iterations (per method/split).",
    )
    parser.add_argument(
        "--max-tuning-len",
        type=int,
        default=20_000,
        help=(
            "Truncate each TUNING series to this many rows for HP search "
            "(EVA evaluation always uses the full series). Keeps the "
            "multivariate conditional-MVN tuning passes tractable."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    splits: list[Literal["U", "M"]] = (
        ["U", "M"] if args.split == "both" else [args.split]
    )

    # Under --subset, shrink the tuning budget and series length so smoke
    # tests finish in minutes (the multivariate per-row MVN CDF is costly:
    # one conditional CDF per feature per row, so wide series dominate).
    if args.subset is not None and args.subset <= 5:
        args.init_points = min(args.init_points, 2)
        args.n_iter = min(args.n_iter, 1)
        args.max_tuning_len = min(args.max_tuning_len, 1500)

    for split in splits:
        run_split(
            split,
            args.subset,
            args.init_points,
            args.n_iter,
            args.max_tuning_len,
            args.out,
        )

    # Results were appended incrementally during the run; read them back
    # (this also folds in any rows from earlier, resumed sessions).
    if not args.out.exists():
        logger.error("No results produced.")
        return 1
    results = pd.read_csv(args.out)
    logger.info("Total %d rows in %s", len(results), args.out)

    analyse(results, splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
