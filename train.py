"""
End-to-end training & evaluation script for IoT event-log anomaly detection.

Pipeline:
  1. Load & preprocess train_data.csv  (unsupervised — no labels)
  2. Train Isolation Forest, One-Class SVM, K-Means
  3. Evaluate on test_data.csv         (has true_label: 0=normal, 1=anomaly)
  4. Report Precision, Recall, F1, Accuracy per model
  5. Show live demo stats from live_data.csv
  6. Save all .pkl models + plots

Usage:
    cd ml-service
    python train.py

    # Or override paths:
    python train.py --train ../data/train_data.csv --test ../data/test_data.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path so 'app' is importable
sys.path.insert(0, os.path.dirname(__file__))

from app.preprocessing import load_and_preprocess
from app.models import isolation_forest, one_class_svm, kmeans
from app.evaluation import (
    plot_timeseries_anomalies,
    plot_score_distribution,
    plot_scatter,
    plot_model_comparison,
    print_summary,
    evaluate_with_labels,
)

# ── Default data paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if 'COLAB_GPU' in os.environ:
    DATA_DIR = "/content/drive/MyDrive/data"
else:
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")

DEFAULT_TRAIN = os.path.join(DATA_DIR, "train_data.csv")
DEFAULT_TEST  = os.path.join(DATA_DIR, "test_data.csv")
DEFAULT_LIVE  = os.path.join(DATA_DIR, "live_data.csv")


def main(train_path: str, test_path: str, live_path: str):
    print("\n" + "=" * 65)
    print("  IoT Anomaly Detection - Training Pipeline")
    print("  (Hierarchical Event-Log Schema)")
    print("=" * 65)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1:  TRAIN  (unsupervised — true_label is NaN / not used)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[1/5] Loading & preprocessing TRAINING data:")
    print(f"      {train_path}")

    if not os.path.exists(train_path):
        print(f"\n[!] Training data not found at: {train_path}")
        print("    Please place train_data.csv in the data/ folder.")
        sys.exit(1)

    df_train, X_train, scaler, train_report = load_and_preprocess(
        train_path, fit_scaler=True, verbose=True
    )
    print(f"\n      Clean rows : {train_report.get('final_rows')} / {train_report.get('raw_rows')}")
    print(f"      Removed    : {train_report.get('rows_removed', 0)} rows")
    print(f"      Features   : {X_train.shape[1]}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2:  TRAIN MODELS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/5] Training models …")

    print("  Training Isolation Forest …")
    if_model = isolation_forest.train(X_train, contamination=0.05)

    print("  Training One-Class SVM …")
    # Colab OOM Crash Protection: One-Class SVM memory complexity is O(n^2).
    # 50,000 rows is a reasonable upper limit for Colab. It may take a few minutes to train.
    if X_train.shape[0] > 50000:
        print(f"    [!] Dataset large ({X_train.shape[0]} rows).")
        print(f"    [!] Sampling down to 50,000 rows to prevent Google Colab RAM crash.")
        np.random.seed(42)
        idx = np.random.choice(X_train.shape[0], 50000, replace=False)
        X_train_oc = X_train[idx]
    else:
        X_train_oc = X_train
    oc_model = one_class_svm.train(X_train_oc, nu=0.05)

    print("  Training K-Means …")
    km_model, km_threshold = kmeans.train(X_train, n_clusters=3)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3:  EVALUATE ON TEST SET  (has true_label)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[3/5] Evaluating on TEST data:")
    print(f"      {test_path}")

    if os.path.exists(test_path):
        df_test, X_test, _, test_report = load_and_preprocess(
            test_path, fit_scaler=False, verbose=False
        )
        print(f"      Test rows: {len(df_test)}")

        # Extract ground truth labels (0=normal, 1=anomaly)
        y_true = None
        if "true_label" in df_test.columns:
            y_true_raw = df_test["true_label"].copy()
            # Convert: 0 → 1 (normal), 1 → -1 (anomaly) to match sklearn convention
            y_true = np.where(y_true_raw == 1, -1, 1)
            n_anom_true = int((y_true == -1).sum())
            n_norm_true = int((y_true == 1).sum())
            print(f"      Ground truth: {n_norm_true} normal, {n_anom_true} anomaly")

        # Run predictions on test set
        if_test = isolation_forest.predict(X_test)
        oc_test = one_class_svm.predict(X_test)
        km_test = kmeans.predict(X_test)

        all_results = {
            "Isolation Forest": if_test["labels"],
            "One-Class SVM":    oc_test["labels"],
            "K-Means":          km_test["labels"],
        }

        # Print metrics with ground truth
        for name, result in [
            ("Isolation Forest", if_test),
            ("One-Class SVM",    oc_test),
            ("K-Means",          km_test),
        ]:
            model_slug = name.lower().replace(" ", "_").replace("-", "_")
            print_summary(name, result["labels"], result["scores"])

            if y_true is not None:
                evaluate_with_labels(name, result["labels"], y_true)

            # Generate plots
            plot_timeseries_anomalies(
                df_test, result["labels"], model_slug, feature="LogFloatValue"
            )
            plot_score_distribution(result["scores"], model_slug)
            plot_scatter(df_test, result["labels"], model_slug,
                         x_col="LogFloatValue", y_col="time_delay_sec")

        # Cross-model comparison chart
        plot_model_comparison(all_results)

        # Consensus
        consensus = (
            (if_test["labels"] == -1) &
            (oc_test["labels"] == -1) &
            (km_test["labels"] == -1)
        )
        print(f"\n  -- Cross-Model Consensus --")
        print(f"     Points flagged by ALL 3 models: {consensus.sum()}")

    else:
        print(f"  [!] Test data not found - skipping evaluation.")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4:  LIVE DATA DEMO
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[4/5] Live data demo:")
    print(f"      {live_path}")

    if os.path.exists(live_path):
        df_live, X_live, _, live_report = load_and_preprocess(
            live_path, fit_scaler=False, verbose=False
        )
        print(f"      Live rows: {len(df_live)}")

        if_live = isolation_forest.predict(X_live)
        n_anom = int((if_live["labels"] == -1).sum())
        print(f"      Isolation Forest live anomalies: {n_anom} / {len(df_live)} "
              f"({n_anom / len(df_live) * 100:.1f}%)")
    else:
        print(f"  [!] Live data not found - skipping.")

    # ═══════════════════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[5/5] Complete!")
    print(f"\n{'=' * 65}")
    print(f"  Models saved in: {os.path.join(BASE_DIR, 'saved_models')}")
    print(f"  Plots  saved in: {os.path.join(BASE_DIR, 'results')}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IoT anomaly detection models")
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="Path to training CSV")
    parser.add_argument("--test",  default=DEFAULT_TEST,  help="Path to test CSV (with true_label)")
    parser.add_argument("--live",  default=DEFAULT_LIVE,  help="Path to live CSV (no labels)")
    args = parser.parse_args()
    main(args.train, args.test, args.live)
