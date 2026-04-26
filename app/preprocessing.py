"""
Preprocessing pipeline for hierarchical IoT event log data.

The dataset contains logs from 5 device types (LogType) with multiple sub-event
types (LogSubType). Features are already partly engineered in the CSV files:
  - Signals    : LogIntValue, LogFloatValue
  - Temporal   : log_hour, log_dayofweek, is_night, time_delay_sec
  - One-Hot    : LogType_*, LogSubType_*

Pipeline stages (in order):
  Stage 1 — LOAD          : Read CSV, auto-detect columns
  Stage 2 — CLEAN         : Remove duplicates, coerce types, handle NaN
  Stage 3 — TIMESTAMPS    : Parse LogTime / ServerTime, recompute time_delay_sec if raw
  Stage 4 — MISSING VALUES: Impute signals with group-aware median
  Stage 5 — IQR FLAGS     : Context-aware IQR outlier flagging per LogType group
  Stage 6 — SCALE         : StandardScaler (fit during training, transform at inference)

The cleaning report is returned as a dict so it can be served through the
Flask API for display in the Angular UI.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler


# ── Column sets ──────────────────────────────────────────────────────────────

SIGNAL_COLS   = ["LogIntValue", "LogFloatValue"]
TEMPORAL_COLS = ["log_hour", "log_dayofweek", "is_night", "time_delay_sec"]
METADATA_COLS = ["RmsStationId", "NodeId", "TenantId", "TagId", "MobileNumber",
                 "ServerTime", "LogTime"]
LABEL_COL     = "true_label"

# One-hot prefixes — actual columns are detected dynamically at load time
LOGTYPE_PREFIX    = "LogType_"
LOGSUBTYPE_PREFIX = "LogSubType_"

IQR_MULTIPLIER = 3.0   # IQR fence multiplier for outlier flagging
SCALER_PATH    = os.path.join(os.path.dirname(__file__), "..", "saved_models", "scaler.pkl")


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(file_path: str, fit_scaler: bool = True, verbose: bool = True):
    """
    Full preprocessing + cleaning pipeline for a CSV or Excel file.

    Parameters
    ----------
    file_path  : path to the raw CSV or Excel file
    fit_scaler : True during training (fits & saves scaler), False for inference
    verbose    : print the cleaning report to console

    Returns
    -------
    df_clean   : cleaned DataFrame with all columns preserved
    X_scaled   : numpy array ready for model input (n_rows × n_features)
    scaler     : fitted StandardScaler instance
    report     : dict with cleaning statistics (for API / notebook display)
    """
    report = {}

    # ── Stage 1: Load ────────────────────────────────────────────────────
    if file_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    report["raw_rows"] = len(df)
    report["raw_cols"] = list(df.columns)
    _section("Stage 1: Load", verbose)
    _log(f"Loaded {len(df)} rows × {len(df.columns)} columns", verbose)

    # ── Stage 2: Clean ───────────────────────────────────────────────────
    df, clean_stats = _clean(df, verbose)
    report.update(clean_stats)

    # ── Stage 3: Timestamps ──────────────────────────────────────────────
    df, ts_stats = _process_timestamps(df, verbose)
    report.update(ts_stats)

    # ── Stage 4: Missing values ──────────────────────────────────────────
    df, mv_stats = _handle_missing(df, verbose)
    report.update(mv_stats)

    # ── Stage 5: Context-aware IQR outlier flags ─────────────────────────
    df, iqr_stats = _flag_iqr_outliers(df, verbose)
    report.update(iqr_stats)

    # ── Stage 6: Build feature matrix & scale ────────────────────────────
    model_features = _get_model_features(df)
    X = df[model_features].fillna(0).values.astype(np.float64)
    report["model_feature_count"] = len(model_features)
    report["model_features"]     = model_features

    scaler = None
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        
        # Save feature columns
        features_path = os.path.join(os.path.dirname(SCALER_PATH), "features.json")
        with open(features_path, "w") as f:
            json.dump(model_features, f)
            
        _section("Stage 6: Scale", verbose)
        _log(f"StandardScaler fitted and saved -> {len(model_features)} features", verbose)
    else:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            features_path = os.path.join(os.path.dirname(SCALER_PATH), "features.json")
            
            if os.path.exists(features_path):
                with open(features_path, "r") as f:
                    expected_cols = json.load(f)
                X_aligned = df[model_features].reindex(columns=expected_cols, fill_value=0)
                X_scaled = scaler.transform(X_aligned.values.astype(np.float64))
            elif hasattr(scaler, "feature_names_in_"):
                expected_cols = list(scaler.feature_names_in_)
                X_aligned = df[model_features].reindex(columns=expected_cols, fill_value=0)
                X_scaled = scaler.transform(X_aligned.values.astype(np.float64))
            else:
                X_scaled = scaler.transform(X)
        else:
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        _section("Stage 6: Scale", verbose)
        _log("StandardScaler applied (loaded from disk)", verbose)

    report["final_rows"]   = len(df)
    report["rows_removed"] = report["raw_rows"] - len(df)

    if verbose:
        _print_summary(report)

    return df, X_scaled, scaler, report


def preprocess_records(records: list, verbose: bool = False) -> np.ndarray:
    """
    Lightweight preprocessing for real-time API inference (list of dicts).

    Parameters
    ----------
    records : [{"LogIntValue": 0, "LogFloatValue": 32.8, "LogTime": "...",
                "ServerTime": "...", "LogTypeID": 4, "LogSubTypeID": 1}, ...]

    Returns
    -------
    X_scaled : numpy array ready for model.predict()
    """
    df = pd.DataFrame(records)

    # ── Compute temporal features if raw timestamps are present ──────────
    if "LogTime" in df.columns:
        df["LogTime"] = pd.to_datetime(df["LogTime"], errors="coerce")
        df["log_hour"]      = df["LogTime"].dt.hour.fillna(0).astype(int)
        df["log_dayofweek"] = df["LogTime"].dt.dayofweek.fillna(0).astype(int)
        df["is_night"]      = df["log_hour"].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)

    if "ServerTime" in df.columns and "LogTime" in df.columns:
        df["ServerTime"] = pd.to_datetime(df["ServerTime"], errors="coerce")
        df["time_delay_sec"] = (df["ServerTime"] - df["LogTime"]).dt.total_seconds().fillna(0)

    # ── One-hot encode LogTypeID / LogSubTypeID if present ────────────────
    if "LogTypeID" in df.columns:
        dummies = pd.get_dummies(df["LogTypeID"], prefix="LogType", dtype=int)
        df = pd.concat([df, dummies], axis=1)
    if "LogSubTypeID" in df.columns:
        dummies = pd.get_dummies(df["LogSubTypeID"], prefix="LogSubType", dtype=int)
        df = pd.concat([df, dummies], axis=1)

    # ── Ensure signal columns exist ──────────────────────────────────────
    for col in SIGNAL_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in TEMPORAL_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Build feature matrix and scale
    model_features = _get_model_features(df)
    X = df[model_features].fillna(0).values.astype(np.float64)

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        features_path = os.path.join(os.path.dirname(SCALER_PATH), "features.json")
        
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                expected_cols = json.load(f)
            X_aligned = df[model_features].reindex(columns=expected_cols, fill_value=0)
            return scaler.transform(X_aligned.values.astype(np.float64))
        elif hasattr(scaler, "feature_names_in_"):
            expected_cols = list(scaler.feature_names_in_)
            X_aligned = df[model_features].reindex(columns=expected_cols, fill_value=0)
            return scaler.transform(X_aligned.values.astype(np.float64))
        elif hasattr(scaler, "n_features_in_"):
            expected = scaler.n_features_in_
            if X.shape[1] < expected:
                X = np.hstack([X, np.zeros((X.shape[0], expected - X.shape[1]))])
            elif X.shape[1] > expected:
                X = X[:, :expected]
        return scaler.transform(X)
    else:
        scaler = StandardScaler()
        return scaler.fit_transform(X)


# ════════════════════════════════════════════════════════════════════════════
#  STAGE IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════════════

def _clean(df: pd.DataFrame, verbose: bool) -> tuple:
    """Stage 2 — Basic structural cleaning."""
    _section("Stage 2: Clean", verbose)
    stats = {}

    # Remove fully empty rows
    before = len(df)
    df = df.dropna(how="all")
    dropped_empty = before - len(df)
    stats["dropped_fully_empty_rows"] = dropped_empty
    _log(f"Dropped fully empty rows: {dropped_empty}", verbose)

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped_dupes = before - len(df)
    stats["dropped_duplicate_rows"] = dropped_dupes
    _log(f"Dropped duplicate rows:   {dropped_dupes}", verbose)

    # Coerce signal columns to numeric (non-parseable → NaN for imputation)
    for col in SIGNAL_COLS:
        if col in df.columns:
            before_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_nulls = df[col].isna().sum() - before_nulls
            if new_nulls > 0:
                _log(f"  '{col}': {new_nulls} values coerced to NaN (non-numeric)", verbose)

    # Coerce temporal columns
    for col in TEMPORAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce one-hot columns to int (they come as bool from CSV)
    onehot_cols = [c for c in df.columns
                   if c.startswith(LOGTYPE_PREFIX) or c.startswith(LOGSUBTYPE_PREFIX)]
    for col in onehot_cols:
        df[col] = df[col].astype(int)

    _log(f"One-hot columns detected: {len(onehot_cols)} ({onehot_cols})", verbose)

    return df, stats


def _process_timestamps(df: pd.DataFrame, verbose: bool) -> tuple:
    """Stage 3 — Timestamp parsing and time_delay_sec verification."""
    _section("Stage 3: Timestamps", verbose)
    stats = {}

    has_logtime    = "LogTime" in df.columns
    has_servertime = "ServerTime" in df.columns
    stats["has_logtime"]    = has_logtime
    stats["has_servertime"] = has_servertime

    if has_logtime:
        df["LogTime"] = pd.to_datetime(df["LogTime"], errors="coerce")
        n_invalid = int(df["LogTime"].isna().sum())
        stats["invalid_logtimes"] = n_invalid
        _log(f"LogTime parsed. Invalid (unparseable): {n_invalid}", verbose)

    if has_servertime:
        df["ServerTime"] = pd.to_datetime(df["ServerTime"], errors="coerce")
        n_invalid = int(df["ServerTime"].isna().sum())
        stats["invalid_servertimes"] = n_invalid
        _log(f"ServerTime parsed. Invalid: {n_invalid}", verbose)

    # Recompute time_delay_sec if both timestamps are present and it's missing
    if has_logtime and has_servertime and "time_delay_sec" not in df.columns:
        df["time_delay_sec"] = (df["ServerTime"] - df["LogTime"]).dt.total_seconds().fillna(0)
        _log("Computed time_delay_sec from ServerTime - LogTime", verbose)

    # Recompute temporal features if missing
    if has_logtime:
        if "log_hour" not in df.columns:
            df["log_hour"] = df["LogTime"].dt.hour.fillna(0).astype(int)
        if "log_dayofweek" not in df.columns:
            df["log_dayofweek"] = df["LogTime"].dt.dayofweek.fillna(0).astype(int)
        if "is_night" not in df.columns:
            df["is_night"] = df["log_hour"].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)

    return df, stats


def _handle_missing(df: pd.DataFrame, verbose: bool) -> tuple:
    """Stage 4 — Report missing values then impute signals."""
    _section("Stage 4: Missing Values", verbose)
    stats = {"missing_before": {}, "missing_after": {}}

    for col in SIGNAL_COLS + TEMPORAL_COLS:
        if col not in df.columns:
            continue

        n_missing = int(df[col].isna().sum())
        pct       = n_missing / len(df) * 100 if len(df) > 0 else 0
        stats["missing_before"][col] = {"count": n_missing, "pct": round(pct, 2)}

        if n_missing > 0:
            _log(f"'{col}' missing: {n_missing} ({pct:.1f}%)", verbose)
            # Impute with column median
            col_median = df[col].median()
            df[col] = df[col].fillna(col_median if not pd.isna(col_median) else 0)

        remaining = int(df[col].isna().sum())
        stats["missing_after"][col] = remaining

    # Impute one-hot columns with 0 (not present = False)
    onehot_cols = [c for c in df.columns
                   if c.startswith(LOGTYPE_PREFIX) or c.startswith(LOGSUBTYPE_PREFIX)]
    for col in onehot_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0).astype(int)

    _log(f"Imputation complete. Remaining NaN in signals: "
         f"{sum(stats['missing_after'].values())}", verbose)

    return df, stats


def _flag_iqr_outliers(df: pd.DataFrame, verbose: bool) -> tuple:
    """
    Stage 5 — Context-aware IQR outlier flagging.

    Computes IQR fences PER LogType group for LogFloatValue.
    This ensures that a temperature reading (LogType=4) is compared
    only against other temperature readings, not against energy kWh (LogType=2).
    """
    _section("Stage 5: Statistical Outlier Flags (Context-Aware IQR)", verbose)
    stats = {"iqr_outliers": {}}

    # Determine which LogType group each row belongs to
    logtype_cols = sorted([c for c in df.columns if c.startswith(LOGTYPE_PREFIX)])

    if not logtype_cols:
        _log("No LogType columns found — applying global IQR", verbose)
        for col in SIGNAL_COLS:
            if col not in df.columns:
                continue
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo  = Q1 - IQR_MULTIPLIER * IQR
            hi  = Q3 + IQR_MULTIPLIER * IQR
            mask = (df[col] < lo) | (df[col] > hi)
            df[f"flag_iqr_{col}"] = mask.astype(int)
            n = int(mask.sum())
            stats["iqr_outliers"][col] = {"count": n, "lower": round(lo, 3), "upper": round(hi, 3)}
            _log(f"'{col}' global IQR [{lo:.2f}, {hi:.2f}] -> {n} flagged", verbose)
        return df, stats

    # Group-aware IQR: determine the LogType group for each row
    # Create a group label from one-hot columns
    df["_logtype_group"] = "unknown"
    for col in logtype_cols:
        mask = df[col] == 1
        df.loc[mask, "_logtype_group"] = col

    for signal_col in SIGNAL_COLS:
        if signal_col not in df.columns:
            continue

        df[f"flag_iqr_{signal_col}"] = 0
        group_stats = {}

        for group_name, group_df in df.groupby("_logtype_group"):
            if len(group_df) < 10:
                continue
            Q1  = group_df[signal_col].quantile(0.25)
            Q3  = group_df[signal_col].quantile(0.75)
            IQR = Q3 - Q1
            lo  = Q1 - IQR_MULTIPLIER * IQR
            hi  = Q3 + IQR_MULTIPLIER * IQR

            mask = (df.index.isin(group_df.index)) & ((df[signal_col] < lo) | (df[signal_col] > hi))
            df.loc[mask, f"flag_iqr_{signal_col}"] = 1
            n = int(mask.sum())
            group_stats[group_name] = {"count": n, "lower": round(lo, 3), "upper": round(hi, 3)}

        total_flagged = int(df[f"flag_iqr_{signal_col}"].sum())
        stats["iqr_outliers"][signal_col] = {
            "total_flagged": total_flagged,
            "per_group": group_stats,
        }
        _log(f"'{signal_col}' context-aware IQR -> {total_flagged} total flagged "
             f"across {len(group_stats)} groups", verbose)

    # Remove helper column
    df.drop(columns=["_logtype_group"], inplace=True)

    return df, stats


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _get_model_features(df: pd.DataFrame) -> list:
    """Return the ordered list of feature columns present in df."""
    features = []

    # 1. Signal columns
    for col in SIGNAL_COLS:
        if col in df.columns:
            features.append(col)

    # 2. Temporal columns
    for col in TEMPORAL_COLS:
        if col in df.columns:
            features.append(col)

    # 3. One-hot LogType columns (sorted for consistency)
    logtype_cols = sorted([c for c in df.columns if c.startswith(LOGTYPE_PREFIX)])
    features += logtype_cols

    # 4. One-hot LogSubType columns (sorted for consistency)
    logsubtype_cols = sorted([c for c in df.columns if c.startswith(LOGSUBTYPE_PREFIX)])
    features += logsubtype_cols

    # 5. IQR flag columns (if present)
    flag_cols = sorted([c for c in df.columns if c.startswith("flag_iqr_")])
    features += flag_cols

    return features


def _section(title: str, verbose: bool):
    if verbose:
        print(f"\n  {'-' * 50}")
        print(f"  {title}")
        print(f"  {'-' * 50}")


def _log(msg: str, verbose: bool):
    if verbose:
        print(f"    {msg}")


def _print_summary(report: dict):
    print(f"\n  {'=' * 50}")
    print("  CLEANING SUMMARY")
    print(f"  {'=' * 50}")
    print(f"    Raw rows              : {report.get('raw_rows', '?')}")
    print(f"    Duplicate rows removed: {report.get('dropped_duplicate_rows', 0)}")
    print(f"    Empty rows removed    : {report.get('dropped_fully_empty_rows', 0)}")
    print(f"    Final rows            : {report.get('final_rows', '?')}")
    print(f"    Features for model    : {report.get('model_feature_count', '?')}")
    print(f"    Feature list          : {report.get('model_features', [])}")
    print(f"  {'=' * 50}")
