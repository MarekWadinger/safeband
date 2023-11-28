# IMPORTS
import ast
import os
import random
import sys
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_opt import (
    BayesianOptimization,
    SequentialDomainReductionTransformer,
)
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from river import anomaly, cluster, utils
from river.metrics import AdjustedMutualInfo
from river.metrics.base import MultiClassMetric
from sklearn.decomposition import PCA

sys.path.insert(1, str(Path().resolve().parent))
from functions.anomaly import ConditionalGaussianScorer  # noqa: E402
from functions.compose import build_model, convert_to_nested_dict  # noqa: E402
from functions.evaluate import progressive_val_predict  # noqa: E402
from functions.proba import MultivariateGaussian  # noqa: E402

# CONSTANTS
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# DATA
df = pd.read_csv("data/multivariate/cats/data_1t_agg_last.csv", index_col=0)
df.index = pd.to_datetime(df.index, utc=True)

df_y = df[["y", "category"]]
df = df.drop(columns=["y", "category"])

df_meta = pd.read_csv("data/multivariate/cats/metadata.csv")
df_meta.start_time = pd.to_datetime(df_meta.start_time, utc=True)
df_meta.end_time = pd.to_datetime(df_meta.end_time, utc=True)

df_y["rc"] = None
df_y["affected"] = None
for i in range(len(df_meta)):
    start = df_meta.start_time[i]
    end = df_meta.end_time[i]
    df_y.loc[start:end, "rc"] = df_meta.root_cause[i]
    df_y.loc[start:end, "affected"] = ast.literal_eval(df_meta.affected[i])[0]

# df["is_anomaly"] = df_y.category.replace({None: ""})
df["is_anomaly"] = df_y.rc.replace({None: ""})


# FUNCTIONS
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


def tune_train_model(steps, df, val_kwargs: dict = {}, **params):
    params = convert_to_nested_dict(params)
    model = build_model(steps, params)
    metric: MultiClassMetric = AdjustedMutualInfo()
    try:
        val_kwargs.update(params.get("Val", {}))
        y_pred, _ = progressive_val_predict(
            model, df, [], print_every=0, print_final=False, **val_kwargs
        )
        y_pred = cluster_map(df_ys.anomaly, y_pred)
        for yt, yp in zip(df.anomaly, y_pred):
            metric.update(yt, yp)
        return metric.get()
    except Exception as e:
        print(e)
        return 0


def get_random_samples(df: pd.DataFrame, num_samples=10000):
    if len(df) <= num_samples:
        return df
    else:
        return df.sample(n=num_samples, random_state=RANDOM_STATE)


def plot_detection(df: pd.DataFrame, y_pred):
    df["pred"] = y_pred
    if "anomaly" in df.columns:
        df = get_random_samples(df)
        if len(df.columns) >= 4:
            # Separate the feature columns from the target column ("anomaly")
            X = df.drop(columns=["anomaly", "pred"])
            y = df["anomaly"]
            y_pred = df["pred"]

            # Apply PCA to reduce the feature columns to 2 components
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            # Create a new DataFrame with the reduced components and "anomaly" column
            df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
            df_pca["anomaly"] = y.values
            df_pca["pred"] = y_pred.values
        else:
            print(True)
            df_pca = pd.DataFrame(df.reset_index().copy())
            df_pca.columns = ["PC1", "PC2", "anomaly", "pred"]

        # Plot the 2D scatter plot
        plt.scatter(
            df_pca[df_pca["anomaly"] == 0]["PC1"],
            df_pca[df_pca["anomaly"] == 0]["PC2"],
        )
        plt.scatter(
            df_pca[df_pca["anomaly"] == 1]["PC1"],
            df_pca[df_pca["anomaly"] == 1]["PC2"],
            facecolors="none",
            edgecolors="r",
            linewidths=0.5,
        )
        plt.scatter(
            df_pca[df_pca["pred"] == 1]["PC1"],
            df_pca[df_pca["pred"] == 1]["PC2"],
            marker="x",
            linewidths=1,
        )  # type: ignore
        plt.xticks(())
        plt.yticks(())


def save_results_y(df_ys, path):
    dir_path = path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df_ys.to_csv(f"{dir_path}/ys.csv", index=False)


def save_results_metrics(metrics_res, path):
    dir_path = path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    metrics_res.to_csv(f"{dir_path}/metrics.csv")


# MODS
class QuantileFilter(anomaly.QuantileFilter):
    def __init__(
        self, anomaly_detector, q: float, protect_anomaly_detector=True
    ):
        super().__init__(
            anomaly_detector=anomaly_detector,
            protect_anomaly_detector=protect_anomaly_detector,
            q=q,
        )

    def predict_one(self, *args):
        score = self.score_one(*args)
        return self.classify(score)


# SETTINGS

# DETECTION ALGORITHMS
detection_algorithms = [
    (
        "Conditional Gaussian Scorer",
        [
            [
                partial(ConditionalGaussianScorer, grace_period=16667),
                [utils.Rolling, MultivariateGaussian],
            ]
        ],
        {
            "ConditionalGaussianScorer__threshold": (0.95, 0.99994),
            "Rolling__window_size__round": (150, 30000),
            "ConditionalGaussianScorer__t_a__int": (50, 10000),
        },
    ),
    (
        "DBStream",
        [cluster.DBSTREAM],
        {
            "DBSTREAM__clustering_threshold": (0.01, 100),
            "DBSTREAM__fading_factor": (0.0001, 1.0),
            "DBSTREAM__cleanup_interval__int": (1, 1000),
            "DBSTREAM__intersection_factor": (0.03, 3.0),
            "DBSTREAM__minimum_weight": (0.1, 10),
        },
    ),
]

# DATASETS
datasets = [
    {
        "name": "CATS",
        "data": df,
        "anomaly_col": "is_anomaly",
        "drop": None,
    },
]

# RUN
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dataset in datasets:
            # PREPROCESS DATA
            df = dataset["data"]
            df.index = pd.to_timedelta(
                range(0, len(df)), "T"
            ) + pd.Timestamp.utcnow().replace(microsecond=0)
            if isinstance(dataset["anomaly_col"], str):
                df = df.rename(columns={dataset["anomaly_col"]: "anomaly"})
            elif isinstance(dataset["anomaly_col"], pd.Series):
                df_y = dataset["anomaly_col"]
                df["anomaly"] = df_y.rename("anomaly").values
            if dataset["drop"] is not None:
                df = df.drop(columns=dataset["drop"])
            print(f"\n=== {dataset['name']} === [{len(df)}]".ljust(80, "="))

            df_ys = df[["anomaly"]].copy()
            # RUN EACH MODEL AGAINST DATASET
            for alg in detection_algorithms:
                print(f"\n===== {alg[0]}".ljust(80, "="))
                # INITIALIZE OPTIMIZER
                pbounds = alg[2]
                mod_fun = partial(tune_train_model, alg[1], df, {})

                # INITIALIZE METRICS
                metrics_list = []

                # TUNE HYPERPARAMETERS
                optimizer = BayesianOptimization(
                    f=mod_fun,
                    pbounds=pbounds,
                    verbose=2,
                    random_state=RANDOM_STATE,
                    allow_duplicate_points=True,
                    bounds_transformer=SequentialDomainReductionTransformer(),
                )
                logger = JSONLogger(
                    path=f"./.results/{dataset['name']}-{alg[0]}.log"
                )
                optimizer.subscribe(Events.OPTIMIZATION_END, logger)
                optimizer.maximize()  # init_points=1, n_iter=5)
                params = convert_to_nested_dict(optimizer.max["params"])
                print(params)
                model = build_model(alg[1], params)
                if hasattr(model, "seed"):
                    model.seed = RANDOM_STATE  # type: ignore
                if hasattr(model, "random_state"):
                    model.random_state = RANDOM_STATE  # type: ignore
                # USE TUNED MODEL
                # PROGRESSIVE PREDICT
                y_pred, _ = progressive_val_predict(
                    model, df, metrics=[]
                )

                # SAVE PREDICITONS
                df_ys[f"{alg[0]}__{params}"] = y_pred

            # LOAD RESULTS
            #  Save
            dir_path = f".results/{dataset['name']}"
            save_results_y(df_ys, f".results/{dataset['name']}")
