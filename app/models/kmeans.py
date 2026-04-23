"""
K-Means anomaly detection model.

K-Means clusters the data. Points that are far from their nearest
cluster centroid (above a distance threshold) are flagged as anomalies.

Best for: Comparison baseline; interpretable clusters.
Threshold: 95th percentile of cluster distances on training data.
"""

import os
import joblib
import numpy as np
from sklearn.cluster import KMeans

MODEL_PATH     = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "kmeans.pkl"
)
THRESHOLD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "kmeans_threshold.pkl"
)


def train(X: np.ndarray, n_clusters: int = 3, percentile: float = 95) -> tuple:
    """
    Train K-Means and compute anomaly distance threshold.

    Parameters
    ----------
    X           : scaled feature array
    n_clusters  : number of clusters (3 works well for most IoT data)
    percentile  : distance percentile above which a point is anomalous

    Returns
    -------
    (model, threshold)
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    model.fit(X)

    distances = _distances_to_centroid(model, X)
    threshold = np.percentile(distances, percentile)

    _save(model, threshold)
    return model, threshold


def predict(X: np.ndarray) -> dict:
    """
    Run inference using saved model and threshold.

    Returns
    -------
    dict with labels (-1/1), scores (distances), readable strings
    """
    model, threshold = _load()
    distances = _distances_to_centroid(model, X)

    labels   = np.where(distances > threshold, -1, 1)
    scores   = distances  # higher distance = more anomalous
    readable = ["Anomaly" if l == -1 else "Normal" for l in labels]
    return {"labels": labels, "scores": scores, "readable": readable}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _distances_to_centroid(model: KMeans, X: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to its nearest cluster centroid."""
    cluster_centers = model.cluster_centers_
    assigned = model.predict(X)
    distances = np.array(
        [np.linalg.norm(X[i] - cluster_centers[assigned[i]]) for i in range(len(X))]
    )
    return distances


def _save(model, threshold):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)
    print(f"  [OK] K-Means saved: {MODEL_PATH}")
    print(f"  [OK] K-Means threshold: {threshold:.4f}")


def _load() -> tuple:
    for path in (MODEL_PATH, THRESHOLD_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Run train.py first.")
    return joblib.load(MODEL_PATH), joblib.load(THRESHOLD_PATH)
