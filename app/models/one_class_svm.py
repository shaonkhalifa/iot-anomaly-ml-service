"""
One-Class SVM anomaly detection model.

Trains a tight boundary around normal data points.
Observations falling outside that boundary are flagged as anomalies.

Best for: Clean normal-only training data; well-defined boundaries.
Note    : Slower than Isolation Forest on large datasets.
"""

import os
import joblib
import numpy as np
from sklearn.svm import OneClassSVM

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "one_class_svm.pkl"
)


def train(X: np.ndarray, nu: float = 0.05, kernel: str = "rbf") -> OneClassSVM:
    """
    Train a One-Class SVM.

    Parameters
    ----------
    X      : scaled feature array
    nu     : upper bound on fraction of outliers (≈ contamination)
    kernel : 'rbf' works best for most sensor data

    Returns
    -------
    Fitted OneClassSVM model
    """
    model = OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma="scale",
    )
    model.fit(X)
    _save(model)
    return model


def predict(X: np.ndarray) -> dict:
    """
    Run inference.

    Returns
    -------
    dict with labels (-1/1), scores (decision_function), readable strings
    """
    model = _load()
    labels = model.predict(X)
    scores = model.decision_function(X)
    readable = ["Anomaly" if l == -1 else "Normal" for l in labels]
    return {"labels": labels, "scores": scores, "readable": readable}


def _save(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  [OK] One-Class SVM saved: {MODEL_PATH}")


def _load() -> OneClassSVM:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)
