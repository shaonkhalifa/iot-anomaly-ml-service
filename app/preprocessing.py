"""
Preprocessing pipeline for industrial IoT sensor data.

Pipeline stages (in order):
  Stage 1 — LOAD          : Read CSV, detect column types
  Stage 2 — CLEAN         : Remove duplicates, fix column names
  Stage 3 — VALIDATE      : Domain-range checks (temp/humidity bounds)
  Stage 4 — TIMESTAMPS    : Parse, sort, detect future dates, detect gaps
  Stage 5 — MISSING VALUES: Report → Impute (forward-fill then median)
  Stage 6 — OUTLIER FLAGS : IQR-based statistical flag (pre-model)
  Stage 7 — STUCK SENSOR  : Rolling-window zero-variance detection
  Stage 8 — FEATURE ENG.  : Rate-of-change, rolling stats, cyclical time
  Stage 9 — SCALE         : StandardScaler (fit during training, transform at inference)

The cleaning report is printed to the console and also returned as a dict
so it can be served through the Flask API for display in the Angular UI.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_COLS  = ["temperature", "humidity"]

# Physical domain limits for industrial sensors
DOMAIN_LIMITS = {
    "temperature": (-40.0, 150.0),   # °C  — beyond these = physically impossible
    "humidity":    (0.0,   100.0),   # %   — 0–100 by definition
}

SCALER_PATH    = os.path.join(os.path.dirname(__file__), "..", "saved_models", "scaler.pkl")
STUCK_WINDOW   = 10        # consecutive identical readings → stuck sensor
IQR_MULTIPLIER = 3.0       # how many IQRs beyond Q1/Q3 = statistical outlier
FUTURE_CUTOFF  = pd.Timestamp.utcnow().tz_localize(None)


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path: str, fit_scaler: bool = True, verbose: bool = True):
    """
    Full preprocessing + cleaning pipeline for a CSV file.

    Parameters
    ----------
    csv_path   : path to the raw CSV file
    fit_scaler : True during training (fits & saves scaler), False for inference
    verbose    : print the cleaning report to console

    Returns
    -------
    df_clean   : cleaned DataFrame with all engineered columns + flags
    X_scaled   : numpy array ready for model input  (shape: n_rows × n_features)
    scaler     : fitted StandardScaler instance
    report     : dict with cleaning statistics (for API / notebook display)
    """
    report = {}

    # ── Stage 1: Load ────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    report["raw_rows"]    = len(df)
    report["raw_cols"]    = list(df.columns)
    _section("Stage 1: Load", verbose)
    _log(f"Loaded {len(df)} rows × {len(df.columns)} columns", verbose)

    # ── Stage 2: Clean ───────────────────────────────────────────────────────
    df, clean_stats = _clean(df, verbose)
    report.update(clean_stats)

    # ── Stage 3: Validate domain ranges ─────────────────────────────────────
    df, val_stats = _validate_domain(df, verbose)
    report.update(val_stats)

    # ── Stage 4: Timestamps ──────────────────────────────────────────────────
    df, ts_stats = _process_timestamps(df, verbose)
    report.update(ts_stats)

    # ── Stage 5: Missing values ──────────────────────────────────────────────
    df, mv_stats = _handle_missing(df, verbose)
    report.update(mv_stats)

    # ── Stage 6: Statistical outlier flags (IQR) ─────────────────────────────
    df, iqr_stats = _flag_iqr_outliers(df, verbose)
    report.update(iqr_stats)

    # ── Stage 7: Stuck sensor detection ──────────────────────────────────────
    df, stuck_stats = _detect_stuck_sensors(df, verbose)
    report.update(stuck_stats)

    # ── Stage 8: Feature engineering ─────────────────────────────────────────
    df = _engineer_features(df, verbose)

    # ── Stage 9: Build feature matrix & scale ────────────────────────────────
    model_features = _get_model_features(df)
    X = df[model_features].fillna(0).values
    report["model_feature_count"] = len(model_features)
    report["model_features"]      = model_features

    scaler = None
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        _section("Stage 9: Scale", verbose)
        _log(f"StandardScaler fitted and saved → {len(model_features)} features", verbose)
    else:
        if os.path.exists(SCALER_PATH):
            scaler    = joblib.load(SCALER_PATH)
            X_scaled  = scaler.transform(X)
        else:
            scaler    = StandardScaler()
            X_scaled  = scaler.fit_transform(X)
        _section("Stage 9: Scale", verbose)
        _log("StandardScaler applied (loaded from disk)", verbose)

    report["final_rows"]    = len(df)
    report["rows_removed"]  = report["raw_rows"] - len(df)

    if verbose:
        _print_summary(report)

    return df, X_scaled, scaler, report


def preprocess_records(records: list, verbose: bool = False) -> np.ndarray:
    """
    Lightweight preprocessing for real-time API inference (list of dicts).

    Parameters
    ----------
    records : [{"temperature": 23.5, "humidity": 65.2, ...}, ...]

    Returns
    -------
    X_scaled : numpy array ready for model.predict()
    """
    df = pd.DataFrame(records)

    # Ensure required columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Coerce numeric columns
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median()
                                                                  if df[col].notna().any() else 0)

    # Apply domain clamp (don't remove rows — keep all records for inference)
    for col, (lo, hi) in DOMAIN_LIMITS.items():
        if col in df.columns:
            df[f"flag_domain_{col}"] = ((df[col] < lo) | (df[col] > hi)).astype(int)

    df, _ = _detect_stuck_sensors(df, verbose=False)
    df    = _engineer_features(df, verbose=False)

    model_features = _get_model_features(df)
    X = df[model_features].fillna(0).values

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
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

    # Normalise column names → lowercase, strip spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    _log(f"Column names normalised: {list(df.columns)}", verbose)

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

    # Coerce sensor columns to numeric (non-parseable → NaN for imputation)
    for col in FEATURE_COLS:
        if col in df.columns:
            before_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_nulls = df[col].isna().sum() - before_nulls
            if new_nulls > 0:
                _log(f"  '{col}': {new_nulls} values coerced to NaN (non-numeric)", verbose)

    return df, stats


def _validate_domain(df: pd.DataFrame, verbose: bool) -> tuple:
    """Stage 3 — Physical domain range checks."""
    _section("Stage 3: Validate Domain Ranges", verbose)
    stats = {"domain_violations": {}}

    for col, (lo, hi) in DOMAIN_LIMITS.items():
        if col not in df.columns:
            continue

        mask_low  = df[col] < lo
        mask_high = df[col] > hi
        mask_bad  = mask_low | mask_high

        n_bad = int(mask_bad.sum())
        stats["domain_violations"][col] = {
            "below_min": int(mask_low.sum()),
            "above_max": int(mask_high.sum()),
            "total":     n_bad,
            "valid_min": lo, "valid_max": hi,
        }

        # Add a flag column (keep the rows — the models should learn from extremes!)
        df[f"flag_domain_{col}"] = mask_bad.astype(int)

        _log(
            f"'{col}' range [{lo}, {hi}]: "
            f"{mask_low.sum()} below min, {mask_high.sum()} above max → "
            f"flagged as 'flag_domain_{col}'",
            verbose,
        )

    return df, stats


def _process_timestamps(df: pd.DataFrame, verbose: bool) -> tuple:
    """Stage 4 — Timestamp parsing, sorting, future-date flagging, gap detection."""
    _section("Stage 4: Timestamps", verbose)
    stats = {}

    if "timestamp" not in df.columns:
        _log("No 'timestamp' column found — skipping timestamp processing", verbose)
        stats["has_timestamp"] = False
        return df, stats

    stats["has_timestamp"] = True

    # Parse
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_invalid_ts = int(df["timestamp"].isna().sum())
    stats["invalid_timestamps"] = n_invalid_ts
    _log(f"Invalid (unparseable) timestamps: {n_invalid_ts}", verbose)

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    _log("Sorted by timestamp ascending", verbose)

    # Flag future timestamps
    df["flag_future_ts"] = (df["timestamp"] > FUTURE_CUTOFF).astype(int)
    n_future = int(df["flag_future_ts"].sum())
    stats["future_timestamps"] = n_future
    _log(f"Future timestamps (beyond now): {n_future} → flagged as 'flag_future_ts'", verbose)

    # Detect timestamp gaps (> 3× median interval = significant gap)
    valid_ts = df["timestamp"].dropna().sort_values()
    if len(valid_ts) > 1:
        diffs  = valid_ts.diff().dropna()
        median = diffs.median()
        large  = (diffs > median * 3).sum()
        stats["large_gaps"] = int(large)
        _log(f"Large time gaps (>3x median interval): {large}", verbose)
    else:
        stats["large_gaps"] = 0

    return df, stats


def _handle_missing(df: pd.DataFrame, verbose: bool) -> tuple:
    """Stage 5 — Report missing values then impute."""
    _section("Stage 5: Missing Values", verbose)
    stats = {"missing_before": {}, "missing_after": {}}

    for col in FEATURE_COLS:
        if col not in df.columns:
            continue

        n_missing = int(df[col].isna().sum())
        pct       = n_missing / len(df) * 100 if len(df) > 0 else 0
        stats["missing_before"][col] = {"count": n_missing, "pct": round(pct, 2)}
        _log(f"'{col}' missing before imputation: {n_missing} ({pct:.1f}%)", verbose)

        if n_missing > 0:
            # Strategy 1: forward-fill (preserves time-series continuity)
            df[col] = df[col].ffill()
            # Strategy 2: backward-fill (for leading NaNs)
            df[col] = df[col].bfill()
            # Strategy 3: median fallback (if still NaN)
            col_median = df[col].median()
            df[col] = df[col].fillna(col_median if not pd.isna(col_median) else 0)

        remaining = int(df[col].isna().sum())
        stats["missing_after"][col] = remaining
        _log(f"'{col}' missing after imputation:  {remaining}", verbose)

    return df, stats


def _flag_iqr_outliers(df: pd.DataFrame, verbose: bool) -> tuple:
    """
    Stage 6 — IQR-based statistical outlier flagging.

    Uses 3 × IQR rule (more permissive than 1.5× to suit industrial variation).
    Does NOT remove rows — flags them so the model can consider extreme-but-valid
    readings differently from truly anomalous ones.
    """
    _section("Stage 6: Statistical Outlier Flags (IQR)", verbose)
    stats = {"iqr_outliers": {}}

    for col in FEATURE_COLS:
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
        stats["iqr_outliers"][col] = {"count": n, "lower_fence": round(lo, 3), "upper_fence": round(hi, 3)}

        _log(
            f"'{col}' IQR fences: [{lo:.2f}, {hi:.2f}] → "
            f"{n} statistical outliers flagged as 'flag_iqr_{col}'",
            verbose,
        )

    return df, stats


def _detect_stuck_sensors(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Stage 7 — Stuck sensor detection via rolling standard deviation.

    A sensor is "stuck" when it transmits the same value continuously.
    Rolling std ≈ 0 over STUCK_WINDOW consecutive readings triggers a flag.
    """
    _section("Stage 7: Stuck Sensor Detection", verbose)
    stats = {"stuck_readings": {}}

    df["flag_stuck"] = 0

    for col in FEATURE_COLS:
        if col not in df.columns:
            continue

        # Rolling std over STUCK_WINDOW readings; NaN fills (< window rows) = not stuck
        rolling_std = (
            df[col]
            .rolling(window=STUCK_WINDOW, min_periods=STUCK_WINDOW)
            .std()
            .fillna(1.0)          # first (window-1) rows → default "not stuck"
        )
        stuck_mask = (rolling_std < 1e-6).astype(int)
        df["flag_stuck"] = (df["flag_stuck"] | stuck_mask).astype(int)

        n = int(stuck_mask.sum())
        stats["stuck_readings"][col] = n
        _log(f"'{col}': {n} readings in stuck-sensor windows (window={STUCK_WINDOW})", verbose)

    total_stuck = int(df["flag_stuck"].sum())
    stats["total_stuck_flagged"] = total_stuck
    _log(f"Total rows flagged 'flag_stuck': {total_stuck}", verbose)

    return df, stats


def _engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Stage 8 — Feature engineering for anomaly detection.

    New features:
      {col}_diff       : first-order difference (rate of change)
      {col}_abs_diff   : absolute rate of change
      {col}_rolling_mean : rolling mean (smoothed baseline)
      {col}_rolling_std  : rolling std (local variability)
      {col}_z_score    : z-score from rolling mean/std (local deviation)
      hour_sin/cos     : cyclical hour-of-day encoding
    """
    _section("Stage 8: Feature Engineering", verbose)
    ROLL = 5  # rolling window for local statistics

    for col in FEATURE_COLS:
        if col not in df.columns:
            continue

        df[f"{col}_diff"]         = df[col].diff().fillna(0)
        df[f"{col}_abs_diff"]     = df[f"{col}_diff"].abs()
        df[f"{col}_rolling_mean"] = df[col].rolling(ROLL, min_periods=1).mean()
        df[f"{col}_rolling_std"]  = df[col].rolling(ROLL, min_periods=1).std().fillna(0)

        # Local z-score: how far is this point from its 5-step local mean (in stds)?
        safe_std = df[f"{col}_rolling_std"].replace(0, 1e-9)
        df[f"{col}_z_score"] = (df[col] - df[f"{col}_rolling_mean"]) / safe_std

    feat_list = []
    for col in FEATURE_COLS:
        feat_list += [f"{col}_diff", f"{col}_abs_diff",
                      f"{col}_rolling_mean", f"{col}_rolling_std", f"{col}_z_score"]
    _log(f"Added features: {[f for f in feat_list if f in df.columns]}", verbose)

    # Cyclical time encoding (daily seasonality)
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        hour = df["timestamp"].dt.hour.fillna(0)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        _log("Added cyclical time features: hour_sin, hour_cos", verbose)

    return df


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _get_model_features(df: pd.DataFrame) -> list:
    """Return the ordered list of feature columns present in df."""
    candidates = []
    for col in FEATURE_COLS:
        candidates += [
            col,
            f"{col}_diff",
            f"{col}_abs_diff",
            f"{col}_rolling_mean",
            f"{col}_rolling_std",
            f"{col}_z_score",
        ]
    candidates += [
        "flag_stuck",
        "flag_future_ts",
        "hour_sin",
        "hour_cos",
    ]
    # Add any domain/IQR flag columns automatically
    flag_cols = [c for c in df.columns if c.startswith("flag_domain_") or c.startswith("flag_iqr_")]
    candidates += flag_cols
    return [c for c in candidates if c in df.columns]


def _section(title: str, verbose: bool):
    if verbose:
        print(f"\n  {'─' * 50}")
        print(f"  {title}")
        print(f"  {'─' * 50}")


def _log(msg: str, verbose: bool):
    if verbose:
        print(f"    {msg}")


def _print_summary(report: dict):
    print(f"\n  {'=' * 50}")
    print("  CLEANING SUMMARY")
    print(f"  {'=' * 50}")
    print(f"    Raw rows            : {report.get('raw_rows', '?')}")
    print(f"    Duplicate rows removed : {report.get('dropped_duplicate_rows', 0)}")
    print(f"    Empty rows removed  : {report.get('dropped_fully_empty_rows', 0)}")
    print(f"    Final rows          : {report.get('final_rows', '?')}")
    print(f"    Future timestamps   : {report.get('future_timestamps', 0)}")
    print(f"    Stuck sensor flags  : {report.get('total_stuck_flagged', 0)}")
    print(f"    Features for model  : {report.get('model_feature_count', '?')}")
    print(f"  {'=' * 50}")
