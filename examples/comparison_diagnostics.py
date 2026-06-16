"""Bayesian hyperparameter optimisation comparing anomaly detectors."""

# IMPORTS
import ast
import collections
import logging
import pickle
import random
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from river import utils

sys.path.insert(1, str(Path.cwd().parent))
from functions.anomaly import ConditionalGaussianScorer
from functions.fault_diagnosis import SensorFaultClassifier
from functions.proba import MultivariateGaussian

logger = logging.getLogger(__name__)

# CONSTANTS
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)


# FUNCTIONS
def save_model(model: object, path: str) -> None:
    """Serialise model to a pickle file inside the given directory."""
    Path(path).mkdir(parents=True, exist_ok=True)
    with Path(f"{path}/{alg[0]}.pkl").open("wb") as f:
        pickle.dump(model, f)


def save_results_y(df_ys: pd.DataFrame, path: str) -> None:
    """Save prediction DataFrame to a CSV file inside the given directory."""
    Path(path).mkdir(parents=True, exist_ok=True)
    df_ys.to_csv(f"{path}/ys.csv", index=False)


# FAULT-TYPE DIAGNOSIS DEMO (IDEAS I7)
def demo_fault_diagnosis(
    n_train: int = 300,
    n_gap: int = 80,
    n_fault: int = 120,
    seed: int = 42,
) -> None:
    """Classify the four sensor-fault types on a synthetic stream.

    Trains a ``ConditionalGaussianScorer`` on a correlated four-signal
    stream, then injects each fault of the taxonomy — bias, drift,
    accuracy loss, freezing — into one signal in turn and reports the
    labels assigned by ``SensorFaultClassifier``. Kept small so it
    finishes in seconds::

        python comparison_diagnostics.py --demo-faults
    """
    rng_demo = np.random.default_rng(seed)
    signals = [f"s{i}" for i in range(4)]

    def sample() -> dict[str, float]:
        latent = rng_demo.standard_normal()
        return {
            s: 0.6 * latent + 0.8 * rng_demo.standard_normal() for s in signals
        }

    scorer = ConditionalGaussianScorer(
        utils.Rolling(MultivariateGaussian(seed=seed), n_train),
        grace_period=50,
        protect_anomaly_detector=False,
    )
    for _ in range(n_train):
        scorer.learn_one(sample())

    clf = SensorFaultClassifier(window=20, long_window=80)
    injections = [
        ("bias", "s0"),
        ("drift", "s1"),
        ("accuracy_loss", "s2"),
        ("freezing", "s3"),
    ]
    logger.info("fault           target  labels on target (last %s)", 50)
    for fault, target in injections:
        # Healthy gap so the streaming statistics settle again.
        for _ in range(n_gap):
            x = sample()
            clf.process_one(x, scorer.residuals_one(x))
        frozen_at = float(scorer.gaussian.mu[target])
        tail: collections.Counter[str] = collections.Counter()
        cross: collections.Counter[str] = collections.Counter()
        for t in range(n_fault):
            x = sample()
            if fault == "bias":
                x[target] += 4.0
            elif fault == "drift":
                x[target] += 0.08 * t
            elif fault == "accuracy_loss":
                x[target] += 4.0 * rng_demo.standard_normal()
            else:  # freezing — stuck at a perfectly plausible value
                x[target] = frozen_at
            labels = clf.process_one(
                x,
                scorer.residuals_one(x),
                scorer.drift_detected,
            )
            if t >= n_fault - 50:
                tail[labels[target]] += 1
            cross.update(
                label
                for s, label in labels.items()
                if s != target and label != "normal"
            )
        logger.info(
            "%-15s %-7s %s | other signals: %s",
            fault,
            target,
            dict(tail),
            dict(cross) or "all normal",
        )


if __name__ == "__main__" and "--demo-faults" in sys.argv:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    demo_fault_diagnosis()
    sys.exit(0)


# Optimisation-study-only imports deliberately kept below the demo
# guard so ``--demo-faults`` stays fast and independent of them.
from bayes_opt import (  # noqa: E402
    BayesianOptimization,
    SequentialDomainReductionTransformer,
)

# Event/JSONLogger API removed in bayes_opt 3.x; study ran on 2.x.
from bayes_opt.event import Events  # noqa: E402
from bayes_opt.logger import JSONLogger  # noqa: E402
from river import cluster, metrics  # noqa: E402
from river.metrics import MacroF1  # noqa: E402

from functions.compose import (  # noqa: E402
    build_model,
    convert_to_nested_dict,
)
from functions.evaluate import (  # noqa: E402
    batch_save_evaluate_metrics,
    build_fit_evaluate,
    progressive_val_predict,
)

# DETECTION ALGORITHMS
detection_algorithms = [
    (
        "Conditional Gaussian Scorer",
        [
            [
                partial(ConditionalGaussianScorer, grace_period=16667),
                [utils.Rolling, MultivariateGaussian],
            ],
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
assert isinstance(df, pd.DataFrame)
df.index = pd.to_datetime(df.index, utc=True)

df_y = df[["y", "category"]]
df = df.drop(columns=["y", "category"])

df_meta = pd.read_csv("data/multivariate/cats/metadata.csv")
assert isinstance(df_meta, pd.DataFrame)
df_meta.start_time = pd.to_datetime(df_meta.start_time, utc=True)
df_meta.end_time = pd.to_datetime(df_meta.end_time, utc=True)

df_y["rc"] = None
df_y["affected"] = None
for i in range(len(df_meta)):
    start = df_meta.start_time[i]
    end = df_meta.end_time[i]
    df_y.loc[start:end, "rc"] = df_meta.root_cause[i]
    df_y.loc[start:end, "affected"] = ast.literal_eval(df_meta.affected[i])[0]

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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dataset in datasets:
            # PREPROCESS DATA
            df = dataset["data"]
            assert isinstance(df, pd.DataFrame)
            df.index = pd.to_timedelta(
                list(range(len(df))),
                "min",
            ) + pd.Timestamp.utcnow().replace(microsecond=0)
            if isinstance(dataset["anomaly_col"], str):
                df = df.rename(columns={dataset["anomaly_col"]: "anomaly"})
            elif isinstance(dataset["anomaly_col"], pd.Series):
                df_y = dataset["anomaly_col"]
                df["anomaly"] = df_y.rename("anomaly").to_numpy()
            if dataset["drop"] is not None:
                df = df.drop(columns=dataset["drop"])
            ds_name = str(dataset["name"])
            logger.info(
                "\n=== %s === [%s]%s",
                ds_name,
                len(df),
                "=" * (80 - len(ds_name) - len(str(len(df))) - 12),
            )

            df_ys = df[["anomaly"]].copy()
            # RUN EACH MODEL AGAINST DATASET
            for alg in detection_algorithms:
                logger.info(
                    "\n===== %s%s",
                    alg[0],
                    "=" * (80 - 6 - len(alg[0])),
                )
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
                json_logger = JSONLogger(
                    path=f"./.results/{dataset['name']}-{alg[0]}.log",
                )
                # subscribe() removed in bayes_opt 3.x; study ran on 2.x.
                optimizer.subscribe(
                    Events.OPTIMIZATION_END,
                    json_logger,
                )
                optimizer.maximize()  # init_points=1, n_iter=5)
                assert optimizer.max is not None
                params = convert_to_nested_dict(optimizer.max["params"])
                logger.info("%s", params)
                model = build_model(alg[1], params)
                if hasattr(model, "seed"):
                    model.seed = RANDOM_STATE
                if hasattr(model, "random_state"):
                    model.random_state = RANDOM_STATE
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
                metrics_clustering,
                path,
                task="clustering",
            )

            batch_save_evaluate_metrics(
                metrics_classification,
                path,
                task="classification",
                map_cluster_to_rc=True,
                drop_no_support=True,
            )
