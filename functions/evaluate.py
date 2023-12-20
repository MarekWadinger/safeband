import inspect
import os
import time
from collections import defaultdict
from typing import Literal, Union

import pandas as pd
from river.compose import Pipeline
from river.metrics.base import BinaryMetric, Metric, MultiClassMetric

from functions.compose import build_model, convert_to_nested_dict  # noqa: E402


def progressive_val_predict(  # noqa: C901
    model,
    dataset: pd.DataFrame,
    metrics: Union[list[Union[BinaryMetric, MultiClassMetric]], None] = None,
    print_every: int = 0,
    print_final: bool = True,
    compute_limits: bool = False,
    detect_signal: bool = False,
    detect_change: bool = False,
    sampling_model=None,
    compute_latency: bool = False,
    **kwargs,
):
    # CREATE REFERENCE TO LAST STEP OF PIPELINE (TRACK STATE OF MDOEL)
    if isinstance(model, Pipeline):
        model_ = model[-1]
    else:
        model_ = model
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
        if isinstance(t, pd.Timestamp):
            t = t.tz_localize(None)
        x_: dict[str, float] = x.to_dict()
        if "anomaly" in x_:
            y = x_.pop("anomaly", "")
        else:
            y = None
        # PREDICT
        if (
            metrics is not None
            and all(
                [isinstance(metric, MultiClassMetric) for metric in metrics]
            )
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
                if not isinstance(metrics, list):
                    metrics = [metrics]
                for metric in metrics:
                    metric = metric.update(y, is_anomaly)
                    if (print_every > 0) and (i % print_every == 0):
                        print(metric)
            else:
                raise ValueError(
                    "Dataset must contain column 'anomaly' to " "use metrics."
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
                    if k in model_._feature_names_in
                }
                meta["Signal Anomaly"].append(
                    {
                        k: not ((thresh_low[k] < v) and (v < thresh_high[k]))
                        for k, v in x_in.items()
                    }
                )

        # DETECT NON-UNIFORM SAMPLING
        if sampling_model is not None and isinstance(t, pd.Timestamp):
            if i > 0:
                t_ = (t - t_prev).seconds  # noqa: F821
                sample_a = sampling_model.predict_one(t_)
                meta["Sampling Anomaly"].append(sample_a)

                w = 1 - sampling_model.score_one(t_) if sample_a else 1
                sampling_model.learn_one(t_, w=w)
            else:
                meta["Sampling Anomaly"].append(0)
            t_prev = t  # noqa: F841

        # DETECT CHANGE POINTS
        if detect_change:
            meta["Changepoint"].append(model_._drift_detected())

        # UPDATE MODEL
        if hasattr(model, "gaussian") and inspect.signature(
            model.gaussian.update
        ).parameters.get("t"):
            model = model.learn_one(x_, **{"t": t})
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
        print(f"Avg. latency per sample: {(end - start)*1000/len(dataset)}ms")
        if metrics is not None:
            for metric in metrics:
                print(metric)

    return y_pred, meta


def print_stats(df, y_pred):
    df_y_pred = pd.Series(y_pred, index=df.anomaly.index)
    res = pd.concat([df.anomaly, df_y_pred], axis=1)
    real = res[res["anomaly"] == 1]
    sum_ = sum(real.apply(lambda x: x["anomaly"] == x[0], axis=1))
    len_real = len(real) if not len(real) == 0 else float("nan")
    print(
        f"{'Pred anomalous samples | events | proportion:':<55} "
        f"{sum(df_y_pred):<8} | {sum(df_y_pred.diff().dropna() == 1):<5} | "
        f"{sum(df_y_pred)/len(df_y_pred):.02%}\n"
        f"{'Found samples | events | proportion:':<55} "
        f"{sum_:<8} | {' ':<5} | {sum_/len_real:.02%}"
    )


def cluster_map(y_true, y_pred):
    # Create a dictionary to store the counts of overlaps
    overlap_counts = defaultdict(lambda: defaultdict(int))

    # Iterate over y_true and y_pred to count overlaps
    for true_val, pred_val in zip(y_true, y_pred):
        overlap_counts[pred_val][true_val] += 1

    # Map values in y_pred to values in y_true based on maximum overlap count
    return [
        max(overlap_counts[pred_val], key=overlap_counts[pred_val].get)
        for pred_val in y_pred
    ]


def drop_no_support_labels(metric):
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
):
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
        for y_true, y_pred in zip(df_ys.anomaly, df_ys[col]):
            for metric in metrics_:
                metric = metric.update(y_true, y_pred)
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
            with open(f"{path}/{col.split('__', 1)[0]}.txt", "w") as f:
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
):
    for folder in os.listdir(path):
        # check if listed object is a folder and does not start with a period
        if os.path.isdir(os.path.join(path, folder)) and not folder.startswith(
            "."
        ):
            # loop through the files in the folder
            for file in os.listdir(os.path.join(path, folder)):
                if file == "ys.csv":
                    save_evaluate_metrics(
                        metrics,
                        os.path.join(path, folder),
                        task,
                        map_cluster_to_rc,
                        drop_no_support,
                    )


def build_fit_evaluate(
    steps,
    df,
    metric: Metric,
    map_cluster_to_rc: bool = False,  # 2023-10-30 - ADD: DBStream comparison
    drop_no_support: bool = False,  # 2023-10-30 - ADD: DBStream comparison
    **params,
):
    params = convert_to_nested_dict(params)
    model = build_model(steps, params)
    metric = metric.__class__()  # Make sure metric is fresh
    try:
        y_pred, _ = progressive_val_predict(
            model, df, [], print_every=0, print_final=False
        )
        if map_cluster_to_rc:
            y_pred = cluster_map(df.anomaly, y_pred)
        for yt, yp in zip(df.anomaly, y_pred):
            metric.update(yt, yp)
        if drop_no_support:
            metric = drop_no_support_labels(metric)
        return metric.get() if metric.bigger_is_better else -metric.get()
    except Exception as e:
        print(e)
        return 0 if metric.bigger_is_better else -float("inf")
