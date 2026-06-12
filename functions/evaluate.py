"""Progressive evaluation and metric utilities for river anomaly models."""

import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from river.compose import Pipeline
from river.metrics.base import Metric, MultiClassMetric

from functions.anomaly import GaussianScorer
from functions.compose import build_model, convert_to_nested_dict

logger = logging.getLogger(__name__)


def progressive_val_predict(
    # model is duck-typed across river scorers/pipelines/forecasters;
    # no protocol covers all call sites without breaking ty
    model,  # noqa: ANN001
    dataset: pd.DataFrame,
    metrics: Sequence[Metric] | None = None,
    print_every: int = 0,
    print_final: bool = True,
    compute_limits: bool = False,
    detect_signal: bool = False,
    detect_change: bool = False,
    sampling_model: GaussianScorer | None = None,
    compute_latency: bool = False,
    **kwargs: int,
) -> tuple[list, dict[str, list]]:
    """Run prequential evaluation and return predictions and metadata."""
    # CREATE REFERENCE TO LAST STEP OF PIPELINE (TRACK STATE OF MDOEL)
    model_ = model[-1] if isinstance(model, Pipeline) else model
    y_pred = []
    meta: dict[str, list] = {}
    if compute_limits:
        meta["Limit High"], meta["Limit Low"] = [], []
    if detect_signal:
        meta["Signal Anomaly"] = []
    if detect_change:
        meta["Changepoint"] = []
    if sampling_model is not None:
        meta["Sampling Anomaly"] = []
    if compute_latency:
        meta["Latency"] = []
    t_prev = pd.Timestamp.utcnow()

    if hasattr(model_, "forecast"):
        period = kwargs.get("period", 5)

    start = time.time()
    for i, (t, x) in enumerate(dataset.iterrows()):
        if compute_latency:
            start_i = time.time()
        # PREPOCESSING
        t_loc = t.tz_localize(None) if isinstance(t, pd.Timestamp) else t
        x_: dict[str, float] = x.to_dict()
        y = x_.pop("anomaly", "") if "anomaly" in x_ else None
        # PREDICT
        if (
            metrics is not None
            and all(isinstance(metric, MultiClassMetric) for metric in metrics)
            and hasattr(model_, "get_root_cause")
        ):
            is_anomaly = model_.get_root_cause()
            y_pred.append(is_anomaly)
        elif hasattr(model_, "forecast"):
            ys = model_.forecast(period)
            if i < period:
                y_pred.insert(i, y)
            y_pred.append(ys[-1])
            is_anomaly = ys[0]
        else:
            is_anomaly = model.predict_one(x_)
            y_pred.append(is_anomaly)

        # EVALUATE
        if metrics is not None:
            if y is not None:
                if isinstance(metrics, Metric):
                    metrics = [metrics]
                for metric in metrics:
                    metric.update(cast("bool", y), is_anomaly)
                    if (print_every > 0) and (i % print_every == 0):
                        logger.info("%s", metric)
            else:
                msg = "Dataset must contain column 'anomaly' to use metrics."
                raise ValueError(
                    msg,
                )

        # DYNAMIC OPERATING LIMITS
        if compute_limits and hasattr(model_, "limit_one"):
            thresh_high, thresh_low = model_.limit_one(x_)
            meta["Limit High"].append(thresh_high)
            meta["Limit Low"].append(thresh_low)

            # ISOLATE ROT CAUSES
            if detect_signal:
                x_in = {
                    k: v
                    for k, v in x_.items()
                    if k in model_.feature_names_in_
                }
                meta["Signal Anomaly"].append(
                    {
                        k: not (thresh_low[k] < v < thresh_high[k])
                        for k, v in x_in.items()
                    },
                )

        # DETECT NON-UNIFORM SAMPLING
        if sampling_model is not None and isinstance(t, pd.Timestamp):
            if i > 0:
                t_ = (t.tz_localize(None) - t_prev).seconds
                sample_a = sampling_model.predict_one(t_)
                meta["Sampling Anomaly"].append(sample_a)

                w = 1 - sampling_model.score_one(t_) if sample_a else 1
                sampling_model.learn_one(t_, w=w)
            else:
                meta["Sampling Anomaly"].append(0)
            t_prev = t.tz_localize(None)

        # DETECT CHANGE POINTS
        if detect_change:
            meta["Changepoint"].append(model_.drift_detected)

        # UPDATE MODEL
        if hasattr(model, "gaussian") and inspect.signature(
            model.gaussian.update,
        ).parameters.get("t"):
            model.learn_one(x_, t=t_loc)
        elif hasattr(model, "_supervised") and model._supervised:
            model_up = model.learn_one(x_, y)
            model = model_up if model_up is not None else model
        else:
            model_up = model.learn_one(x_)
            model = model_up if model_up is not None else model

        if compute_latency:
            meta["Latency"].append((time.time() - start_i) * 1000)

    # POSTPROCESSING FOR SYNCHRONEOUS SAMPLING EVALUATION
    if sampling_model is not None:
        for i in range(len(meta["Sampling Anomaly"])):
            if meta["Sampling Anomaly"][i] == 1:
                meta["Sampling Anomaly"][i - 1] = 1

    if hasattr(model_, "forecast"):
        y_pred = y_pred[:-period]

    end = time.time()

    if print_final:
        logger.info(
            "Avg. latency per sample: %sms",
            (end - start) * 1000 / len(dataset),
        )
        if metrics is not None:
            for metric in metrics:
                logger.info("%s", metric)

    return y_pred, meta


def print_stats(df: pd.DataFrame, y_pred: list) -> None:
    """Log predicted vs actual anomaly sample counts and event proportions."""
    df_y_pred = pd.Series(y_pred, index=df.anomaly.index)
    res = pd.concat([df.anomaly, df_y_pred], axis=1)
    real = res[res["anomaly"] == 1]
    sum_ = sum(real.apply(lambda x: x["anomaly"] == x[0], axis=1))
    len_real = len(real) if len(real) != 0 else float("nan")
    logger.info(
        "%s %s | %s | %s\n%s %s | %s | %s",
        "Pred anomalous samples | events | proportion:".ljust(55),
        str(sum(df_y_pred)).ljust(8),
        str(sum(df_y_pred.diff().dropna() == 1)).ljust(5),
        f"{sum(df_y_pred) / len(df_y_pred):.02%}",
        "Found samples | events | proportion:".ljust(55),
        str(sum_).ljust(8),
        " ".ljust(5),
        f"{sum_ / len_real:.02%}",
    )


def cluster_map(y_true: Iterable, y_pred: Iterable) -> list:
    """Remap cluster labels in y_pred to true labels by maximum overlap."""
    # Create a dictionary to store the counts of overlaps
    overlap_counts = defaultdict(lambda: defaultdict(int))

    # Iterate over y_true and y_pred to count overlaps
    for true_val, pred_val in zip(y_true, y_pred, strict=False):
        overlap_counts[pred_val][true_val] += 1

    # Map values in y_pred to values in y_true based on maximum overlap count
    return [
        max(
            overlap_counts[pred_val],
            key=lambda k, pv=pred_val: overlap_counts[pv][k],
        )
        for pred_val in y_pred
    ]


def drop_no_support_labels[M: MultiClassMetric](metric: M) -> M:
    """Remove zero-support labels from the confusion matrix in-place."""
    for c in metric.cm.classes:
        if metric.cm.support(c) == 0.0:
            if c in metric.cm.data:
                metric.cm.data.pop(c)
            for label in metric.cm.data:
                if c in metric.cm.data[label]:
                    metric.cm.data[label].pop(c)
            metric.cm.sum_row.pop(c)
            metric.cm.sum_col.pop(c)
    return metric


def save_evaluate_metrics(
    metrics: list,
    path: str,
    task: Literal["classification", "clustering"],
    map_cluster_to_rc: bool,
    drop_no_support: bool,
) -> None:
    """Compute and save per-column metric results to a CSV in path."""
    col_names = [metric.__class__.__name__ for metric in metrics]
    report_in_metrics = "ClassificationReport" in col_names
    if report_in_metrics:
        report_idx = col_names.index("ClassificationReport")
        del col_names[report_idx]
        col_names += [
            "MacroPrecision",
            "MacroRecall",
            "MacroF1",
            "WeightedPrecision",
            "WeightedRecall",
            "WeightedF1",
            "FAR",
        ]

    df_ys = pd.read_csv(f"{path}/ys.csv")
    df_ys = df_ys.fillna("")
    df_metrics = pd.DataFrame(index=col_names)
    for col in df_ys.columns[1:]:
        metrics_ = [metric.clone() for metric in metrics]
        if map_cluster_to_rc and df_ys[col].dtypes == "int64":
            df_ys[col] = cluster_map(df_ys.anomaly, df_ys[col])
        for y_true, y_pred in zip(df_ys.anomaly, df_ys[col], strict=False):
            for metric in metrics_:
                metric.update(y_true, y_pred)
        if drop_no_support:
            metrics_ = [drop_no_support_labels(metric) for metric in metrics_]

        if report_in_metrics:
            cr = metrics_.pop(report_idx)
            cm = cr.cm
            result = [metric.get() for metric in metrics_] + [
                cr._macro_precision.get(),
                cr._macro_recall.get(),
                cr._macro_f1.get(),
                cr._weighted_precision.get(),
                cr._weighted_recall.get(),
                cr._weighted_f1.get(),
                cm.total_false_positives
                / (cm.total_false_positives + cm.total_true_negatives),
            ]
            with (Path(path) / f"{col.split('__', 1)[0]}.txt").open("w") as f:
                f.write(str(cr))
        else:
            result = [metric.get() for metric in metrics_]

        df_metrics[col] = result

    df_metrics.to_csv(f"{path}/metrics_{task}.csv")


def batch_save_evaluate_metrics(
    metrics: list,
    path: str,
    task: Literal["classification", "clustering"] = "classification",
    map_cluster_to_rc: bool = False,
    drop_no_support: bool = False,
) -> None:
    """Call save_evaluate_metrics for every subdirectory containing ys.csv."""
    for folder in Path(path).iterdir():
        # check if listed object is a folder and does not start with a period
        if folder.is_dir() and not folder.name.startswith("."):
            # loop through the files in the folder
            for file in folder.iterdir():
                if file.name == "ys.csv":
                    save_evaluate_metrics(
                        metrics,
                        str(folder),
                        task,
                        map_cluster_to_rc,
                        drop_no_support,
                    )


def build_fit_evaluate(
    steps: list,
    df: pd.DataFrame,
    metric: Metric,
    map_cluster_to_rc: bool = False,  # 2023-10-30 - ADD: DBStream comparison
    drop_no_support: bool = False,  # 2023-10-30 - ADD: DBStream comparison
    **params: float,
) -> float:
    """Build, fit, and evaluate a model; return the scalar metric value."""
    params = convert_to_nested_dict(params)
    model = build_model(steps, params)
    metric = metric.__class__()  # Make sure metric is fresh
    try:
        y_pred, _ = progressive_val_predict(
            model,
            df,
            [],
            print_every=0,
            print_final=False,
        )
        if map_cluster_to_rc:
            y_pred = cluster_map(df.anomaly, y_pred)
        for yt, yp in zip(df.anomaly, y_pred, strict=False):
            metric.update(yt, yp)
        if drop_no_support:
            metric = drop_no_support_labels(
                cast("MultiClassMetric", metric),
            )
        return metric.get() if metric.bigger_is_better else -metric.get()
    except Exception:
        logger.exception("build_fit_evaluate failed")
        return 0 if metric.bigger_is_better else -float("inf")
