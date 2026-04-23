"""
Evaluation and visualization for unsupervised anomaly detection.

Since data is unlabeled, evaluation is done via:
  1. Anomaly score distributions  (histogram per model)
  2. Time-series plots             (data with anomalies highlighted in red)
  3. Model comparison bar chart    (anomaly counts + score stats)
  4. Scatter plots                 (feature pairs with anomaly overlay)
  5. Cross-model consensus report  (points flagged by all 3 models)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server/Colab
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
NORMAL_COLOR  = "#3B82F6"   # blue
ANOMALY_COLOR = "#EF4444"   # red
BG_COLOR      = "#0F172A"   # dark navy
GRID_COLOR    = "#1E293B"
TEXT_COLOR    = "#F1F5F9"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.alpha":        0.4,
    "font.family":       "sans-serif",
})


# ── 1. Time-series anomaly plot ──────────────────────────────────────────────
def plot_timeseries_anomalies(df: pd.DataFrame, predictions: np.ndarray,
                               model_name: str, feature: str = "temperature"):
    """
    Line chart of a sensor feature over time with anomalies highlighted in red.

    Parameters
    ----------
    df          : original DataFrame (must have 'timestamp' or numeric index)
    predictions : array of -1 (anomaly) or 1 (normal)
    model_name  : used in title and filename
    feature     : column to plot
    """
    if feature not in df.columns:
        print(f"[warn] '{feature}' not in DataFrame, skipping time-series plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    use_time = "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    x = df["timestamp"] if use_time else df.index

    normal_mask  = predictions == 1
    anomaly_mask = predictions == -1

    # Plot normal points as thin line + dots
    ax.plot(x[normal_mask], df.loc[normal_mask, feature],
            "o", color=NORMAL_COLOR, markersize=3, alpha=0.6, label="Normal")

    # Plot anomalies as large red dots
    ax.scatter(x[anomaly_mask], df.loc[anomaly_mask, feature],
               color=ANOMALY_COLOR, s=60, zorder=5, label="Anomaly")

    if use_time:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    ax.set_title(f"{model_name} — {feature.capitalize()} Anomalies", fontsize=14, pad=12)
    ax.set_xlabel("Time" if use_time else "Index")
    ax.set_ylabel(feature.capitalize())
    ax.legend(loc="upper right")
    ax.grid(True)

    path = os.path.join(RESULTS_DIR, f"{model_name}_{feature}_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓ Time-series plot saved → {path}")


# ── 2. Anomaly score distribution ──────────────────────────────────────────
def plot_score_distribution(scores: np.ndarray, model_name: str):
    """Histogram of anomaly scores to understand the score boundary."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores, bins=60, color=NORMAL_COLOR, edgecolor=BG_COLOR, alpha=0.85)
    ax.axvline(0, color=ANOMALY_COLOR, linestyle="--", linewidth=1.5, label="Decision boundary (0)")
    ax.set_title(f"{model_name} — Score Distribution", fontsize=13)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y")

    path = os.path.join(RESULTS_DIR, f"{model_name}_score_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓ Score distribution   → {path}")


# ── 3. Scatter plot (feature pair) ──────────────────────────────────────────
def plot_scatter(df: pd.DataFrame, predictions: np.ndarray, model_name: str,
                 x_col: str = "temperature", y_col: str = "humidity"):
    """2-D scatter of two features, normal vs anomaly coloured."""
    if x_col not in df.columns or y_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    normal_mask  = predictions == 1
    anomaly_mask = predictions == -1

    ax.scatter(df.loc[normal_mask,  x_col], df.loc[normal_mask,  y_col],
               color=NORMAL_COLOR,  s=15, alpha=0.5, label=f"Normal  ({normal_mask.sum()})")
    ax.scatter(df.loc[anomaly_mask, x_col], df.loc[anomaly_mask, y_col],
               color=ANOMALY_COLOR, s=50, alpha=0.9, label=f"Anomaly ({anomaly_mask.sum()})")

    ax.set_title(f"{model_name} — {x_col.capitalize()} vs {y_col.capitalize()}", fontsize=13)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    ax.legend()
    ax.grid(True)

    path = os.path.join(RESULTS_DIR, f"{model_name}_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓ Scatter plot         → {path}")


# ── 4. Model comparison bar chart ───────────────────────────────────────────
def plot_model_comparison(results: dict):
    """
    Bar chart comparing anomaly counts across all models.

    Parameters
    ----------
    results : {"Isolation Forest": predictions_array, "One-Class SVM": ..., "K-Means": ...}
    """
    model_names   = list(results.keys())
    anomaly_counts = [int((v == -1).sum()) for v in results.values()]
    normal_counts  = [int((v == 1).sum())  for v in results.values()]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_n = ax.bar(x - width / 2, normal_counts,  width, label="Normal",  color=NORMAL_COLOR,  alpha=0.85)
    bars_a = ax.bar(x + width / 2, anomaly_counts, width, label="Anomaly", color=ANOMALY_COLOR, alpha=0.85)

    ax.bar_label(bars_n, padding=3, color=TEXT_COLOR, fontsize=9)
    ax.bar_label(bars_a, padding=3, color=TEXT_COLOR, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_title("Model Comparison — Anomaly Counts", fontsize=14)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y")

    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓ Comparison chart     → {path}")
    return path


# ── 5. Summary statistics ───────────────────────────────────────────────────
def print_summary(model_name: str, predictions: np.ndarray, scores: np.ndarray):
    total   = len(predictions)
    n_anom  = int((predictions == -1).sum())
    n_norm  = int((predictions == 1).sum())
    pct     = n_anom / total * 100

    print(f"\n  ── {model_name} ──")
    print(f"     Total readings : {total}")
    print(f"     Normal         : {n_norm}  ({100 - pct:.1f}%)")
    print(f"     Anomalies      : {n_anom}  ({pct:.1f}%)")
    print(f"     Score mean     : {scores.mean():.4f}")
    print(f"     Score std      : {scores.std():.4f}")
