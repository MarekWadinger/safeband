"""Joblib-based model save and load helpers for versioned recovery files."""

import datetime as dt
import logging
from pathlib import Path

import joblib

from functions.anomaly import GaussianScorer
from functions.utils import common_prefix

logger = logging.getLogger(__name__)


def load_model(path: str, topics: list[str]) -> GaussianScorer | None:
    """Load a model from a given path.

    Args:
        path: The path to the model.
        topics: The topics of the model.

    """
    if path:
        model_name = f"model_{common_prefix(topics).replace('/', '_')}_*.pkl"
        model_files = sorted(Path(path).glob(model_name), reverse=True)
        if model_files:
            for latest_model in model_files:
                recovery_data = joblib.load(latest_model)
                if recovery_data["topics"] == topics:
                    logger.info("Latest model found: %s", latest_model)
                    return recovery_data["model"]
            logger.info(
                "No matching model files found in the recovery folder.",
            )
        else:
            logger.info("No model files found in the recovery folder.")
    return None


def save_model(path: str, topics: list[str], model: object) -> None:
    """Save a model to a given path.

    Args:
        path: The path to the model.
        topics: The topics of the model.
        model: The model to save.

    """
    if path:
        model_prefix = f"model_{common_prefix(topics).replace('/', '_')}"
        now = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True)
        recovery_path = p / f"{model_prefix}_{now}.pkl"
        with recovery_path.open("wb") as f:
            joblib.dump({"model": model, "topics": topics}, f)
            logger.info("Model saved to %s", recovery_path)
