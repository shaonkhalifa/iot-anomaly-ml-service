"""
End-to-end training script.

Runs the full pipeline:
  1. Generate sample data (if no raw CSV exists)
  2. Data Cleaning (9-stage pipeline — see preprocessing.py)
  3. Train Isolation Forest, One-Class SVM, K-Means
  4. Evaluate each model & generate plots
  5. Save all .pkl models to saved_models/

Usage:
    cd ml-service
    python train.py

    # Or point at your own data:
    python train.py --data data/raw/your_file.csv
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
)


DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "raw", "iot_sensor_data.csv")


def main(data_path: str):
    print("\n" + "=" * 65)
    print("  IoT Anomaly Detection — Training Pipeline")
    print("=" * 65)

    # ── 1. Check / generate data ─────────────────────────────────────────
    if not os.path.exists(data_path):
        print(f"\n[!] Dataset not found at: {data_path}")
        print("    Generating sample data first …")
        from generate_sample_data import main as gen_main
        gen_main()

    print(f"\n[1/4] Data Cleaning & Preprocessing:\n      {data_path}")
    df, X_scaled, scaler, report = load_and_preprocess(data_path, fit_scaler=True, verbose=True)
    print(f"\n      Clean rows : {report.get('final_rows')} / {report.get('raw_rows')}")
    print(f"      Removed    : {report.get('rows_removed', 0)} rows")
    print(f"      Features   : {X_scaled.shape[1]}")

    # ── 2. Train models ──────────────────────────────────────────────────
    print("\n[2/4] Training models …")
    print("  Training Isolation Forest …")
    if_model = isolation_forest.train(X_scaled, contamination=0.05)

    print("  Training One-Class SVM …")
    oc_model = one_class_svm.train(X_scaled, nu=0.05)

    print("  Training K-Means …")
    km_model, km_threshold = kmeans.train(X_scaled, n_clusters=3)

    # ── 3. Predict with each model ───────────────────────────────────────
    print("\n[3/4] Generating predictions …")
    if_result = isolation_forest.predict(X_scaled)
    oc_result = one_class_svm.predict(X_scaled)
    km_result = kmeans.predict(X_scaled)

    # ── 4. Evaluate & visualise ──────────────────────────────────────────
    print("\n[4/4] Evaluating models & saving plots …")

    all_results = {
        "Isolation Forest": if_result["labels"],
        "One-Class SVM":    oc_result["labels"],
        "K-Means":          km_result["labels"],
    }

    for name, result in [
        ("Isolation Forest", if_result),
        ("One-Class SVM",    oc_result),
        ("K-Means",          km_result),
    ]:
        model_slug = name.lower().replace(" ", "_").replace("-", "_")
        print_summary(name, result["labels"], result["scores"])
        plot_timeseries_anomalies(df, result["labels"], model_slug, feature="temperature")
        plot_timeseries_anomalies(df, result["labels"], model_slug, feature="humidity")
        plot_score_distribution(result["scores"], model_slug)
        plot_scatter(df, result["labels"], model_slug)

    # Cross-model comparison chart
    plot_model_comparison(all_results)

    # Cross-model consensus: flagged by ALL 3 models
    consensus = (
        (if_result["labels"] == -1) &
        (oc_result["labels"] == -1) &
        (km_result["labels"] == -1)
    )
    print(f"\n  ── Cross-Model Consensus ──")
    print(f"     Points flagged by ALL 3 models: {consensus.sum()}")

    print("\n" + "=" * 65)
    print("  Training complete!")
    print(f"  Models saved in: {os.path.join(os.path.dirname(__file__), 'saved_models')}")
    print(f"  Plots  saved in: {os.path.join(os.path.dirname(__file__), 'results')}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IoT anomaly detection models")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to raw CSV file")
    args = parser.parse_args()
    main(args.data)
