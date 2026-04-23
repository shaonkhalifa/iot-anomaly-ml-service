"""
Isolation Forest anomaly detection model.

Isolation Forest works by randomly partitioning data into trees.
Anomalies are isolated in fewer splits (shorter path length), so
they receive higher anomaly scores.

Best for: High-dimensional, imbalanced datasets; scalable to large data.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "isolation_forest.pkl"
)


def train(X: np.ndarray, contamination: float = 0.05, n_estimators: int = 100) -> IsolationForest:
    """
    Train an Isolation Forest model.

    Parameters
    ----------
    X             : scaled feature array (n_samples, n_features)
    contamination : expected proportion of anomalies (default 5%)
    n_estimators  : number of trees (default 100)

    Returns
    -------
    Fitted IsolationForest model
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    _save(model)
    return model


def predict(X: np.ndarray) -> dict:
    """
    Run inference.

    Returns
    -------
    dict with:
        labels  : np.ndarray  (-1 = anomaly, 1 = normal)
        scores  : np.ndarray  (lower = more anomalous)
        readable: list[str]   ("Anomaly" / "Normal")
    """
    model = _load()
    labels = model.predict(X)            # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)  # distance from boundary
    readable = ["Anomaly" if l == -1 else "Normal" for l in labels]
    return {"labels": labels, "scores": scores, "readable": readable}


def _save(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  [OK] Isolation Forest saved: {MODEL_PATH}")


def _load() -> IsolationForest:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)
