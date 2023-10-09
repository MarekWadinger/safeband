import inspect
import time
from typing import Union

import pandas as pd
from river.metrics.base import BinaryMetric


def progressive_val_predict(  # noqa: C901
        model,
        dataset: pd.DataFrame,
        metrics: Union[list[BinaryMetric], None],
        print_every: int = 0,
        print_final: bool = True,
        compute_limits: bool = False,
        detect_signal: bool = False,
        detect_change: bool = False,
        sampling_model=None,
        **kwargs):
    y_pred = []
    meta = {}
    if compute_limits:
        meta['level_high'], meta['level_low'] = [], []
    if detect_signal:
        meta['signal'] = []
    if detect_change:
        meta['change_point'] = []
    if sampling_model is not None:
        meta['sampling'] = []
    start = time.time()
    for i, (t, x) in enumerate(dataset.iterrows()):
        # PREPOCESSING
        if isinstance(t, pd.Timestamp):
            t = t.tz_localize(None)
        x = x.to_dict()
        if "anomaly" in x:
            y = x.pop("anomaly")
        else:
            y = None

        # PREDICT
        is_anomaly = model.predict_one(x)
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
                raise ValueError("Dataset must contain column 'anomaly' to "
                                 "use metrics.")

        # DYNAMIC OPERATING LIMITS
        if compute_limits and hasattr(model, 'limit_one'):
            thresh_high, thresh_low = model.limit_one(x)
            meta['level_high'].append(thresh_high)
            meta['level_low'].append(thresh_low)

            # ISOLATE ROT CAUSES
            if detect_signal:
                meta['signal'].append(
                    {k: not ((thresh_low[k] < v) and (v < thresh_high[k]))
                     for i, (k, v) in enumerate(x.items())})

        # DETECT NON-UNIFORM SAMPLING
        if sampling_model is not None and isinstance(t, pd.Timestamp):
            if i > 0:
                t_ = (t-t_prev).seconds  # noqa: F821
                sample_a = sampling_model.predict_one(t_)
                meta['sampling'].append(sample_a)

                w = 1-sampling_model.score_one(t_) if sample_a else 1
                sampling_model.learn_one(t_, w=w)
            else:
                meta['sampling'].append(0)
            t_prev = t  # noqa: F841

        # DETECT CHANGE POINTS
        if detect_change:
            meta['change_point'].append(model._drift_detected())

        # UPDATE MODEL
        if (hasattr(model, 'gaussian') and
            inspect.signature(
                model.gaussian.update).parameters.get("t")):
            model = model.learn_one(x, **{'t': t})
        else:
            model = model.learn_one(x)

    # POSTPROCESSING FOR SYNCHRONEOUS SAMPLING EVALUATION
    if sampling_model is not None:
        for i in range(len(meta['sampling'])):
            if meta['sampling'][i] == 1:
                meta['sampling'][i-1] = 1

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
    real = res[res['anomaly'] == 1]
    sum_ = sum(real.apply(lambda x: x['anomaly'] == x[0], axis=1))
    len_real = len(real) if not len(real) == 0 else float('nan')
    print(f"{'Pred anomalous samples | events | proportion:':<55} "
          f"{sum(df_y_pred):<8} | {sum(df_y_pred.diff().dropna() == 1):<5} | "
          f"{sum(df_y_pred)/len(df_y_pred):.02%}\n"
          f"{'Found samples | events | proportion:':<55} "
          f"{sum_:<8} | {' ':<5} | {sum_/len_real:.02%}")
