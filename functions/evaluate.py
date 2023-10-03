import inspect
import time
from typing import Union

import pandas as pd
from river.metrics.base import BinaryMetric


def progressive_val_predict(  # noqa: C901
        model,
        dataset,
        metrics: Union[list[BinaryMetric], None],
        print_every: int = 0,
        protect_anomaly_detector: bool = False,
        print_final: bool = True,
        compute_limits: bool = False,
        **kwargs):
    system_anomaly = []
    change_point = []
    ths, tls = [], []

    start = time.time()
    for i, (t, x) in enumerate(dataset.iterrows()):
        t = t.tz_localize(None)
        x = x.to_dict()
        if "anomaly" in x:
            y = x.pop("anomaly")
        else:
            y = None

    # Check anomaly in system
        is_anomaly = model.predict_one(x)
        system_anomaly.append(is_anomaly)

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

        if compute_limits and hasattr(model, 'limit_one'):
            thresh_high, thresh_low = model.limit_one(x)
            ths.append(thresh_high)
            tls.append(thresh_low)

        if (i != 0) and kwargs.get("t_a"):
            is_change = (sum(system_anomaly[-kwargs["t_a"]:-1]) /
                         len(system_anomaly[-kwargs["t_a"]:-1]) >
                         model.threshold)
        else:
            is_change = 0
        change_point.append(is_change)

        if protect_anomaly_detector:
            if not is_anomaly or is_change:
                if (hasattr(model, 'gaussian') and
                    inspect.signature(
                        model.gaussian.update).parameters.get("t")):
                    model = model.learn_one(x, **{'t': t})
                else:
                    model = model.learn_one(x)
        else:
            if (hasattr(model, 'gaussian') and
                inspect.signature(
                    model.gaussian.update).parameters.get("t")):
                model = model.learn_one(x, **{'t': t})
            else:
                model = model.learn_one(x)

    end = time.time()
    if print_final:
        print(f"Avg. latency per sample: {(end - start)*1000/len(dataset)}ms")
        if metrics is not None:
            for metric in metrics:
                print(metric)

    if compute_limits and hasattr(model, 'limit_one'):
        return system_anomaly, change_point, ths, tls
    else:
        return system_anomaly, change_point, None, None


def print_stats(df, y_pred, change_point):
    df_y_pred = pd.Series(y_pred, index=df.anomaly.index)
    res = pd.concat([df.anomaly, df_y_pred], axis=1)
    real = res[res['anomaly'] == 1]
    sum_ = sum(real.apply(lambda x: x['anomaly'] == x[0], axis=1))
    len_real = len(real) if not len(real) == 0 else float('nan')
    print(f"{'Pred anomalous samples | events | proportion:':<55} "
          f"{sum(df_y_pred):<8} | {sum(df_y_pred.diff().dropna() == 1):<5} | "
          f"{sum(df_y_pred)/len(df_y_pred):.02%}\n"
          f"{'Pred changepoints:':<55} {sum(change_point)}\n"
          f"{'Found samples | events | proportion:':<55} "
          f"{sum_:<8} | {' ':<5} | {sum_/len_real:.02%}")
