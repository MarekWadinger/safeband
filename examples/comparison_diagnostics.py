# IMPORTS
import ast
import os
import pickle
import random
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_opt import (
    BayesianOptimization,
    SequentialDomainReductionTransformer,
)
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from river import cluster, metrics, utils
from river.metrics import MacroF1

sys.path.insert(1, str(Path().resolve().parent))
from functions.anomaly import ConditionalGaussianScorer  # noqa: E402
from functions.compose import build_model, convert_to_nested_dict  # noqa: E402
from functions.evaluate import (  # noqa: E402
    batch_save_evaluate_metrics,
    build_fit_evaluate,
    progressive_val_predict,
)
from functions.proba import MultivariateGaussian  # noqa: E402

# CONSTANTS
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# FUNCTIONS
def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{alg[0]}.pkl", "wb") as f:
        pickle.dump(model, f)


def save_results_y(df_ys, path):
    os.makedirs(path, exist_ok=True)
    df_ys.to_csv(f"{path}/ys.csv", index=False)


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
                mod_fun = partial(
                    build_fit_evaluate,
                    alg[1],
                    df,
                    MacroF1(),
                    map_cluster_to_rc=True,
                    drop_no_support=True,
                )

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
                y_pred, _ = progressive_val_predict(model, df, metrics=[])

                # SAVE PREDICITONS
                df_ys[f"{alg[0]}__{params}"] = y_pred

                dir_path = f".results/{dataset['name']}"
                # SAVE MODEL
                save_model(model, dir_path)

            # LOAD RESULTS
            #  Save
            save_results_y(df_ys, f".results/{dataset['name']}")

            metrics_clustering = [
                metrics.Completeness(),
                metrics.AdjustedMutualInfo(),
                metrics.AdjustedRand(),
                metrics.FowlkesMallows(),
                metrics.VBeta(),
                metrics.Rand(),
                metrics.MutualInfo(),
            ]

            metrics_classification = [
                metrics.Precision(),
                metrics.Recall(),
                metrics.F1(),
                metrics.ClassificationReport(),
            ]

            path = ".results/MF1_opt_rc"

            batch_save_evaluate_metrics(
                metrics_clustering, path, task="clustering"
            )

            batch_save_evaluate_metrics(
                metrics_classification,
                path,
                task="classification",
                map_cluster_to_rc=True,
                drop_no_support=True,
            )
