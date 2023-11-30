"""Scalabiity of Detection Tasks and Detection + Limits Task"""
# Import

import datetime as dt
import sys

from pathlib import Path

import pandas as pd

from river import utils

sys.path.insert(1, str(Path().resolve().parent))
from functions.anomaly import ConditionalGaussianScorer  # noqa: E402
from functions.proba import MultivariateGaussian  # noqa: E402
from functions.evaluate import progressive_val_predict  # noqa: E402

output_path = ".results/scalability/"
if not Path(output_path).exists():
    Path(output_path).mkdir(parents=True)

# Load

df = pd.read_csv(
    "/data/kokam/2023-11-24_kokam_norm.csv",
    index_col=0,
)
df.index = pd.to_datetime(df.index, utc=True)


# CONSTANTS
days = 28
WINDOW = dt.timedelta(hours=24 * days)
minutes = int(WINDOW.total_seconds() / 60)
GRACE_PERIOD = dt.timedelta(minutes=48 * 60 / 2)  # 48 * 60
THRESHOLD = 0.99994

for compute_limits in [True, False]:
    latencies_df = pd.DataFrame([])
    latencies_desc_df = pd.DataFrame([])
    try:
        for n_cols in [1, 10, 20, 30, 40, 50, 60]:
            df_ = df.iloc[:, 0:n_cols].copy()
            model = ConditionalGaussianScorer(
                utils.TimeRolling(MultivariateGaussian(), period=WINDOW),
                grace_period=GRACE_PERIOD,
                t_a=int(minutes),
                threshold=THRESHOLD,
            )

            system_anomaly, meta = progressive_val_predict(
                model=model,
                dataset=df_,
                detect_signal=False,
                detect_change=True,
                compute_limits=compute_limits,
                compute_latency=True,
            )

            df_out = pd.DataFrame(
                {"System Anomaly": system_anomaly, **meta}, index=df_.index
            )
            latencies_df[n_cols] = df_out.Latency
            latencies_desc_df[n_cols] = df_out.Latency.describe()
            print(f"Done with {n_cols} columns")
    except Exception as e:
        print(e)
    finally:
        file_name = "latencies_detection" + (
            "_limits" if compute_limits else ""
        )
        latencies_df.to_csv(f"{output_path}{file_name}.csv")
        latencies_desc_df.to_csv(f"{output_path}{file_name}_desc.csv")
