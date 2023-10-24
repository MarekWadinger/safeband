import datetime as dt
import glob
import os

import joblib


def load_model(path: str, topics: list[str]):
    if path:
        model_files = glob.glob(
            os.path.join(
                path, f"model_{len(topics)}_*.pkl")
            )
        if model_files:
            model_files.sort(reverse=True)
            for latest_model in model_files:
                recovery_data = joblib.load(latest_model)
                if recovery_data["topics"] == topics:
                    model = recovery_data["model"]
                    print("Latest model found:", latest_model)
                    return model
            print("No matching model files found in the recovery folder.")
        else:
            print("No model files found in the recovery folder.")
    return None


def save_model(path: str, topics: list[str], model):
    if path:
        now = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(path):
            os.makedirs(path)
        recovery_path = f"{path}/model_{len(topics)}_{now}.pkl"
        with open(recovery_path, 'wb') as f:
            joblib.dump({"model": model, "topics": topics}, f)
            print(f"Model saved to {recovery_path}")
