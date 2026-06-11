import datetime as dt
import glob
import logging
import os

import joblib

from functions.utils import common_prefix

logger = logging.getLogger(__name__)


def load_model(path: str, topics: list[str]):
    """Load a model from a given path.

    Args:
        path: The path to the model.
        topics: The topics of the model.

    """
    if path:
        model_name = f"model_{common_prefix(topics).replace('/', '_')}_*.pkl"
        model_files = glob.glob(os.path.join(path, model_name))
        if model_files:
            model_files.sort(reverse=True)
            for latest_model in model_files:
                recovery_data = joblib.load(latest_model)
                if recovery_data["topics"] == topics:
                    logger.info("Latest model found: %s", latest_model)
                    return recovery_data["model"]
            logger.info(
                "No matching model files found in the recovery folder."
            )
        else:
            logger.info("No model files found in the recovery folder.")
    return None


def save_model(path: str, topics: list[str], model) -> None:
    """Save a model to a given path.

    Args:
        path: The path to the model.
        topics: The topics of the model.
        model: The model to save.

    """
    if path:
        model_prefix = f"model_{common_prefix(topics).replace('/', '_')}"
        now = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(path):
            os.makedirs(path)
        recovery_path = f"{path}/{model_prefix}_{now}.pkl"
        with open(recovery_path, "wb") as f:
            joblib.dump({"model": model, "topics": topics}, f)
            logger.info("Model saved to %s", recovery_path)
