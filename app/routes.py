"""
Flask REST API for IoT anomaly detection inference.

Endpoints:
  GET  /health            → service health check
  GET  /models            → list available trained models
  POST /predict           → predict from JSON body
  POST /predict/batch     → predict from uploaded CSV file
  POST /predict/compare   → run all 3 models and return comparison

Run locally:
  cd ml-service
  python -m flask --app app.routes run --port 5001 --debug
"""

import os
import io
import sys
import json
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Make sure the ml-service root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.preprocessing import preprocess_records, load_and_preprocess
from app.models import isolation_forest, one_class_svm, kmeans

app = Flask(__name__)
CORS(app)   # Allow Angular dev server cross-origin calls

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
MODEL_FILES = {
    "isolation_forest": "isolation_forest.pkl",
    "one_class_svm":    "one_class_svm.pkl",
    "kmeans":           "kmeans.pkl",
}

# ── Columns to attach from original data to prediction results ────────────────
RETURN_COLS = [
    "LogTime", "ServerTime", "LogIntValue", "LogFloatValue",
    "RmsStationId", "NodeId", "log_hour", "time_delay_sec",
    "LogType", "LogSubType"
]

# ── Dispatcher ───────────────────────────────────────────────────────────────
def _run_model(model_name: str, X: np.ndarray) -> dict:
    """Route to the correct model's predict function."""
    if model_name == "isolation_forest":
        return isolation_forest.predict(X)
    elif model_name == "one_class_svm":
        return one_class_svm.predict(X)
    elif model_name == "kmeans":
        return kmeans.predict(X)
    else:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from: {list(MODEL_FILES)}")


def _make_json_safe(obj):
    """Recursively convert report dict to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _extract_id_from_onehot(df_row, prefix: str) -> int:
    """Extract LogTypeID or LogSubTypeID from one-hot columns like LogType_4."""
    for col in df_row.index:
        if col.startswith(prefix):
            val = df_row[col]
            if val == 1 or val == 1.0 or val is True or str(val).strip().lower() in ['1', 'true']:
                try:
                    return int(col.replace(prefix, ""))
                except ValueError:
                    pass
    return 0


def _build_predictions(result: dict, original_df: pd.DataFrame | None = None) -> list:
    """Convert raw arrays into a list of readable prediction dicts."""
    from datetime import datetime
    now = datetime.now()

    out = []
    for i, (label, score, readable) in enumerate(
        zip(result["labels"], result["scores"], result["readable"])
    ):
        row = {
            "index":      i,
            "prediction": readable,
            "label":      int(label),
            "score":      round(float(score), 6),
        }
        warnings = []

        # Attach original IoT event log values if available
        if original_df is not None and i < len(original_df):
            df_row = original_df.iloc[i]
            for col in RETURN_COLS:
                if col in original_df.columns:
                    val = df_row[col]
                    if pd.isna(val):
                        row[col] = None
                    elif isinstance(val, (pd.Timestamp,)):
                        row[col] = str(val)
                        # Check for future timestamps
                        if col == "LogTime" and val > pd.Timestamp(now):
                            warnings.append(f"Future timestamp detected ({val.strftime('%Y-%m-%d %H:%M')})")
                    # Explicitly cast known integer columns to avoid float serialization (e.g., 10.0 -> 10)
                    elif col in ["log_hour", "RmsStationId", "LogType", "LogSubType"]:
                        try:
                            row[col] = int(val)
                        except:
                            row[col] = None
                    elif isinstance(val, (float, np.floating)):
                        row[col] = round(float(val), 4)
                    else:
                        row[col] = _make_json_safe(val)
                else:
                    # Dynamically reconstruct LogTypeID / LogSubTypeID if missing but one-hot encoded
                    if col == "LogType":
                        row[col] = int(_extract_id_from_onehot(df_row, "LogType_"))
                    elif col == "LogSubType":
                        row[col] = int(_extract_id_from_onehot(df_row, "LogSubType_"))

            # Check IQR flags (only LogFloatValue is meaningful)
            if "flag_iqr_LogFloatValue" in original_df.columns and df_row.get("flag_iqr_LogFloatValue", 0) == 1:
                float_val = df_row.get("LogFloatValue", "?")
                warnings.append(f"LogFloatValue ({float_val}) is outside IQR range for this sensor type")

        row["warnings"] = warnings
        out.append(row)
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — reports which models are trained and ready."""
    model_status = {}
    for name, fname in MODEL_FILES.items():
        model_status[name] = os.path.exists(os.path.join(SAVED_MODELS_DIR, fname))
    return jsonify({
        "status": "ok",
        "service": "IoT Anomaly Detection ML Service",
        "models_ready": model_status,
    })


@app.route("/models", methods=["GET"])
def list_models():
    """Return available models and their readiness."""
    models = []
    for name, fname in MODEL_FILES.items():
        ready = os.path.exists(os.path.join(SAVED_MODELS_DIR, fname))
        models.append({"id": name, "name": name.replace("_", " ").title(), "ready": ready})
    return jsonify({"models": models})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict anomalies from JSON body.

    Request body:
    {
        "model": "isolation_forest",
        "data": [
            {
                "LogIntValue": 0,
                "LogFloatValue": 32.83,
                "LogTime": "2026-01-02 10:40:02",
                "ServerTime": "2026-01-02 10:40:06",
                "LogTypeID": 4,
                "LogSubTypeID": 1
            }
        ]
    }
    """
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"error": "Empty or invalid JSON body"}), 400

        model_name = body.get("model", "isolation_forest")
        records    = body.get("data", [])

        if not records:
            return jsonify({"error": "'data' field is required and must not be empty"}), 400

        X, df = preprocess_records(records)
        result = _run_model(model_name, X)

        n_anomaly = int((result["labels"] == -1).sum())
        n_normal  = int((result["labels"] == 1).sum())

        return jsonify({
            "model":       model_name,
            "predictions": _build_predictions(result, df),
            "summary": {
                "total":          len(records),
                "normal":         n_normal,
                "anomaly":        n_anomaly,
                "anomaly_pct":    round(n_anomaly / len(records) * 100, 2),
            },
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e), "hint": "Run train.py first to train models"}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict from an uploaded CSV file.

    Form fields:
        file  : CSV file (required)
        model : model name (optional, default: isolation_forest)
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Use form-data key 'file'"}), 400

        file       = request.files["file"]
        model_name = request.form.get("model", "isolation_forest")

        if not file.filename.endswith((".csv", ".xlsx", ".xls")):
            return jsonify({"error": "Only CSV and Excel files are supported"}), 400

        # Save temporarily and run cleaning + preprocessing
        ext = os.path.splitext(file.filename)[1]
        tmp_path = os.path.join(SAVED_MODELS_DIR, "..", "data", "processed", f"_uploaded{ext}")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        file.save(tmp_path)

        # verbose=False keeps server logs clean; report is returned to client
        df, X_scaled, _, report = load_and_preprocess(tmp_path, fit_scaler=False, verbose=False)
        result = _run_model(model_name, X_scaled)

        n_anomaly = int((result["labels"] == -1).sum())
        n_normal  = int((result["labels"] == 1).sum())

        # Serialise report (may contain nested dicts; make JSON-safe)
        safe_report = _make_json_safe(report)

        return jsonify({
            "model":          model_name,
            "filename":       file.filename,
            "cleaning_report": safe_report,
            "predictions":    _build_predictions(result, df),
            "summary": {
                "total":       len(df),
                "normal":      n_normal,
                "anomaly":     n_anomaly,
                "anomaly_pct": round(n_anomaly / len(df) * 100, 2),
                "rows_removed_during_cleaning": report.get("rows_removed", 0),
            },
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e), "hint": "Run train.py first"}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch/compare", methods=["POST"])
def predict_batch_compare():
    """
    Run all 3 models on the uploaded CSV/Excel file.
    Form fields:
        file: CSV/Excel file (required)
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Use form-data key 'file'"}), 400

        file = request.files["file"]

        if not file.filename.endswith((".csv", ".xlsx", ".xls")):
            return jsonify({"error": "Only CSV and Excel files are supported"}), 400

        # Save temporarily and run cleaning + preprocessing
        ext = os.path.splitext(file.filename)[1]
        tmp_path = os.path.join(SAVED_MODELS_DIR, "..", "data", "processed", f"_uploaded_compare{ext}")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        file.save(tmp_path)

        df, X_scaled, _, _ = load_and_preprocess(tmp_path, fit_scaler=False, verbose=False)

        comparison = {}
        for model_name in MODEL_FILES:
            result    = _run_model(model_name, X_scaled)
            n_anomaly = int((result["labels"] == -1).sum())
            comparison[model_name] = {
                "anomaly_count":  n_anomaly,
                "normal_count":   int((result["labels"] == 1).sum()),
                "anomaly_pct":    round(n_anomaly / len(df) * 100, 2),
                "score_mean":     round(float(np.mean(result["scores"])), 4),
            }

        return jsonify({"total_records": len(df), "comparison": comparison})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict/compare", methods=["POST"])
def predict_compare():
    """
    Run all 3 models on the same data and return a comparison.

    Accepts the same JSON body as /predict (without the 'model' field).
    """
    try:
        body    = request.get_json(force=True)
        records = body.get("data", [])
        if not records:
            return jsonify({"error": "'data' is required"}), 400

        df = pd.DataFrame(records)
        X  = preprocess_records(records)

        comparison = {}
        for model_name in MODEL_FILES:
            result    = _run_model(model_name, X)
            n_anomaly = int((result["labels"] == -1).sum())
            comparison[model_name] = {
                "anomaly_count":  n_anomaly,
                "normal_count":   int((result["labels"] == 1).sum()),
                "anomaly_pct":    round(n_anomaly / len(records) * 100, 2),
                "score_mean":     round(float(np.mean(result["scores"])), 4),
            }

        return jsonify({"total_records": len(records), "comparison": comparison})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
