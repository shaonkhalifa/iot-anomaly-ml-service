"""
Evaluation and visualization for IoT event-log anomaly detection.

Supports both:
  • Unsupervised evaluation (score distributions, visual inspection)
  • Supervised evaluation  (Precision, Recall, F1, Accuracy against true_label)

Visualizations:
  1. Time-series plots         (data with anomalies highlighted in red)
  2. Anomaly score histograms  (per model)
  3. Scatter plots             (feature pairs with anomaly overlay)
  4. Model comparison bar chart
  5. Classification report     (when true_label is available)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server/Colab
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

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
                               model_name: str, feature: str = "LogFloatValue"):
    """
    Line chart of a signal feature over time with anomalies highlighted in red.

    Parameters
    ----------
    df          : original DataFrame (must have 'LogTime' or numeric index)
    predictions : array of -1 (anomaly) or 1 (normal)
    model_name  : used in title and filename
    feature     : column to plot (default: LogFloatValue)
    """
    if feature not in df.columns:
        print(f"[warn] '{feature}' not in DataFrame, skipping time-series plot.")
        return

    # Subsample for readability if dataset is huge
    max_points = 5000
    if len(df) > max_points:
        idx = np.random.RandomState(42).choice(len(df), max_points, replace=False)
        idx.sort()
        df_plot = df.iloc[idx].reset_index(drop=True)
        pred_plot = predictions[idx]
    else:
        df_plot = df.reset_index(drop=True)
        pred_plot = predictions

    fig, ax = plt.subplots(figsize=(14, 5))

    use_time = "LogTime" in df_plot.columns and pd.api.types.is_datetime64_any_dtype(df_plot["LogTime"])
    x = df_plot["LogTime"] if use_time else df_plot.index

    normal_mask  = pred_plot == 1
    anomaly_mask = pred_plot == -1

    # Plot normal points as thin dots
    ax.plot(x[normal_mask], df_plot.loc[normal_mask, feature],
            "o", color=NORMAL_COLOR, markersize=2, alpha=0.4, label="Normal")

    # Plot anomalies as large red dots
    ax.scatter(x[anomaly_mask], df_plot.loc[anomaly_mask, feature],
               color=ANOMALY_COLOR, s=40, zorder=5, label="Anomaly")

    if use_time:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    ax.set_title(f"{model_name} - {feature} Anomalies", fontsize=14, pad=12)
    ax.set_xlabel("Time" if use_time else "Index")
    ax.set_ylabel(feature)
    ax.legend(loc="upper right")
    ax.grid(True)

    path = os.path.join(RESULTS_DIR, f"{model_name}_{feature}_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] Time-series plot saved -> {path}")


# ── 2. Anomaly score distribution ──────────────────────────────────────────
def plot_score_distribution(scores: np.ndarray, model_name: str):
    """Histogram of anomaly scores to understand the score boundary."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores, bins=60, color=NORMAL_COLOR, edgecolor=BG_COLOR, alpha=0.85)
    ax.axvline(0, color=ANOMALY_COLOR, linestyle="--", linewidth=1.5, label="Decision boundary (0)")
    ax.set_title(f"{model_name} - Score Distribution", fontsize=13)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y")

    path = os.path.join(RESULTS_DIR, f"{model_name}_score_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] Score distribution   -> {path}")


# ── 3. Scatter plot (feature pair) ──────────────────────────────────────────
def plot_scatter(df: pd.DataFrame, predictions: np.ndarray, model_name: str,
                 x_col: str = "LogFloatValue", y_col: str = "time_delay_sec"):
    """2-D scatter of two features, normal vs anomaly coloured."""
    if x_col not in df.columns or y_col not in df.columns:
        return

    # Subsample for large datasets
    max_points = 5000
    if len(df) > max_points:
        idx = np.random.RandomState(42).choice(len(df), max_points, replace=False)
        df_plot = df.iloc[idx].reset_index(drop=True)
        pred_plot = predictions[idx]
    else:
        df_plot = df.reset_index(drop=True)
        pred_plot = predictions

    fig, ax = plt.subplots(figsize=(8, 6))
    normal_mask  = pred_plot == 1
    anomaly_mask = pred_plot == -1

    ax.scatter(df_plot.loc[normal_mask,  x_col], df_plot.loc[normal_mask,  y_col],
               color=NORMAL_COLOR,  s=15, alpha=0.5, label=f"Normal  ({normal_mask.sum()})")
    ax.scatter(df_plot.loc[anomaly_mask, x_col], df_plot.loc[anomaly_mask, y_col],
               color=ANOMALY_COLOR, s=50, alpha=0.9, label=f"Anomaly ({anomaly_mask.sum()})")

    ax.set_title(f"{model_name} - {x_col} vs {y_col}", fontsize=13)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(True)

    path = os.path.join(RESULTS_DIR, f"{model_name}_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] Scatter plot         -> {path}")


# ── 4. Model comparison bar chart ───────────────────────────────────────────
def plot_model_comparison(results: dict):
    """
    Bar chart comparing anomaly counts across all models.

    Parameters
    ----------
    results : {"Isolation Forest": predictions_array, "One-Class SVM": ..., ...}
    """
    model_names    = list(results.keys())
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
    ax.set_title("Model Comparison - Anomaly Counts", fontsize=14)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y")

    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] Comparison chart     -> {path}")
    return path


# ── 5. Summary statistics (unsupervised) ─────────────────────────────────────
def print_summary(model_name: str, predictions: np.ndarray, scores: np.ndarray):
    total   = len(predictions)
    n_anom  = int((predictions == -1).sum())
    n_norm  = int((predictions == 1).sum())
    pct     = n_anom / total * 100

    print(f"\n  -- {model_name} --")
    print(f"     Total readings : {total}")
    print(f"     Normal         : {n_norm}  ({100 - pct:.1f}%)")
    print(f"     Anomalies      : {n_anom}  ({pct:.1f}%)")
    print(f"     Score mean     : {scores.mean():.4f}")
    print(f"     Score std      : {scores.std():.4f}")


# ── 6. Supervised evaluation (when true_label is available) ───────────────────
def evaluate_with_labels(model_name: str, predictions: np.ndarray, y_true: np.ndarray):
    """
    Print Precision, Recall, F1, Accuracy using ground-truth labels.

    Both predictions and y_true must use sklearn convention:
        -1 = anomaly,  1 = normal
    """
    # Convert to binary: anomaly=1, normal=0 for metrics
    y_pred_bin = (predictions == -1).astype(int)
    y_true_bin = (y_true == -1).astype(int)

    acc  = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    print(f"\n     -- {model_name}: Supervised Metrics --")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1 Score  : {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    print(f"     Confusion Matrix:")
    print(f"       TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"       FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")

    # Save metrics to file
    metrics_path = os.path.join(RESULTS_DIR, f"{model_name.lower().replace(' ', '_')}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TN={cm[0][0]}  FP={cm[0][1]}\n")
        f.write(f"  FN={cm[1][0]}  TP={cm[1][1]}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(y_true_bin, y_pred_bin,
                                       target_names=["Normal", "Anomaly"],
                                       zero_division=0))
    print(f"     [OK] Metrics saved -> {metrics_path}")

    # Plot confusion matrix
    _plot_confusion_matrix(cm, model_name)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _plot_confusion_matrix(cm: np.ndarray, model_name: str):
    """Visual confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
                ax=ax, cbar_kws={"label": "Count"},
                annot_kws={"size": 14, "color": TEXT_COLOR})

    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=13, pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    path = os.path.join(RESULTS_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"     [OK] Confusion matrix  -> {path}")
