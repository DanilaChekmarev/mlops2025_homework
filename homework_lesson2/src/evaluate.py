import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def get_metrics():
    params = load_params()

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")

    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {"accuracy": float(accuracy), "rows": len(df)}


def evaluate_model():
    metrics = get_metrics()
    Path("metrics").mkdir(exist_ok=True)

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    evaluate_model()
