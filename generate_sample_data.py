"""
Generate realistic IoT event log data matching the hierarchical schema.

This script creates CSV datasets that mimic the real IoT data structure with
5 device types (LogType) and multiple sub-event types (LogSubType).

Device types:
  1 - ExternalAlarmLog
  2 - DcEnergyMeterLog
  4 - TemperatureSensorLog
  5 - DigitalInput/StatusLog
  6 - AnalogSensorLog

Generated files:
    data/raw/iot_sensor_data.csv  (for local testing / fallback)

Usage:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


# ── Device type definitions ──────────────────────────────────────────────────
DEVICE_TYPES = {
    1: {"name": "ExternalAlarmLog",    "subtypes": [1, 2, 3, 4],
        "int_range": (0, 5), "float_range": (0.0, 1.0)},
    2: {"name": "DcEnergyMeterLog",    "subtypes": [1, 2, 3, 4, 6, 7, 8, 10],
        "int_range": (0, 100), "float_range": (0.0, 5000.0)},
    4: {"name": "TemperatureSensorLog","subtypes": [1, 3],
        "int_range": (0, 2), "float_range": (15.0, 65.0)},
    5: {"name": "DigitalInput",        "subtypes": [1, 2],
        "int_range": (0, 1), "float_range": (0.0, 100.0)},
    6: {"name": "AnalogSensorLog",     "subtypes": [1, 6, 7],
        "int_range": (0, 10), "float_range": (0.0, 500.0)},
}

# Distribution weights (LogType 2 is most common in real data)
TYPE_WEIGHTS = {1: 0.10, 2: 0.45, 4: 0.20, 5: 0.10, 6: 0.15}


def generate_normal_data(n_samples=5000, start_date="2026-01-01"):
    """Generate normal IoT event log readings across all device types."""
    np.random.seed(42)

    start = pd.Timestamp(start_date)
    rows = []

    for i in range(n_samples):
        # Pick device type based on distribution
        log_type = np.random.choice(
            list(TYPE_WEIGHTS.keys()),
            p=list(TYPE_WEIGHTS.values())
        )
        device = DEVICE_TYPES[log_type]

        # Pick a random sub-type for this device
        sub_type = np.random.choice(device["subtypes"])

        # Generate signal values based on device type
        lo_int, hi_int = device["int_range"]
        lo_flt, hi_flt = device["float_range"]

        log_int_value   = np.random.randint(lo_int, hi_int + 1)
        log_float_value = round(np.random.uniform(lo_flt, hi_flt), 6)

        # Generate timestamps
        log_time = start + timedelta(minutes=np.random.randint(0, 60 * 24 * 30))
        delay_sec = np.random.choice([3, 4, 5, 6, 7, 8], p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        server_time = log_time + timedelta(seconds=delay_sec)

        # Metadata
        rms_station = np.random.randint(100, 8000)
        node_id     = np.random.choice([27, 39, 42, 89, 91])
        tenant_id   = np.random.choice([0, 7, 8, 9, 25])
        tag_id      = 0
        mobile      = np.random.choice([0, 1766000000 + np.random.randint(0, 999999)])

        rows.append({
            "RmsStationId":  rms_station,
            "NodeId":        node_id,
            "TenantId":      tenant_id,
            "TagId":         tag_id,
            "MobileNumber":  mobile,
            "LogIntValue":   log_int_value,
            "LogFloatValue": log_float_value,
            "ServerTime":    server_time.strftime("%Y-%m-%d %H:%M:%S"),
            "LogTime":       log_time.strftime("%Y-%m-%d %H:%M:%S"),
            "log_hour":      log_time.hour,
            "log_dayofweek": log_time.dayofweek,
            "is_night":      1 if (log_time.hour >= 22 or log_time.hour <= 6) else 0,
            "time_delay_sec": float(delay_sec),
            "true_label":    np.nan,  # unsupervised — no label
            # One-hot LogType
            **{f"LogType_{t}": (1 if t == log_type else 0) for t in [1, 2, 4, 5, 6]},
            # One-hot LogSubType
            **{f"LogSubType_{s}": (1 if s == sub_type else 0) for s in [1, 2, 3, 4, 6, 7, 8, 10]},
        })

    return pd.DataFrame(rows)


def inject_anomalies(df, anomaly_fraction=0.05):
    """Inject anomalies into the dataset and label them."""
    df = df.copy()
    n = len(df)
    np.random.seed(123)

    n_anomalies = int(n * anomaly_fraction)
    anomaly_indices = np.random.choice(n, size=n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(["extreme_float", "extreme_int",
                                          "negative_delay", "huge_delay",
                                          "wrong_hour"])

        if anomaly_type == "extreme_float":
            # Extreme LogFloatValue way outside normal range
            df.loc[idx, "LogFloatValue"] = round(np.random.uniform(8000, 50000), 4)

        elif anomaly_type == "extreme_int":
            # Extreme LogIntValue
            df.loc[idx, "LogIntValue"] = np.random.randint(500, 10000)

        elif anomaly_type == "negative_delay":
            # ServerTime before LogTime (clock skew)
            df.loc[idx, "time_delay_sec"] = round(np.random.uniform(-60, -5), 1)

        elif anomaly_type == "huge_delay":
            # Huge network delay
            df.loc[idx, "time_delay_sec"] = round(np.random.uniform(300, 3600), 1)

        elif anomaly_type == "wrong_hour":
            # Night-time activity for a device that shouldn't be active
            df.loc[idx, "log_hour"] = np.random.choice([0, 1, 2, 3, 23])
            df.loc[idx, "is_night"] = 1

        df.loc[idx, "true_label"] = 1.0  # anomaly

    # Mark normal rows
    df.loc[df["true_label"] != 1.0, "true_label"] = 0.0

    return df


def main():
    print("=" * 60)
    print("  IoT Event Log Data Generator")
    print("  Hierarchical Schema (5 Device Types)")
    print("=" * 60)

    # Generate normal data
    print("\n[1/3] Generating normal IoT event logs (5000 readings)...")
    df = generate_normal_data(n_samples=5000)

    # Inject anomalies and create a labeled version for testing
    print("[2/3] Injecting anomalies (5%)...")
    df_labeled = inject_anomalies(df)

    # Save to CSV
    output_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "iot_sensor_data.csv")
    df_labeled.to_csv(output_path, index=False)

    n_anom = int((df_labeled["true_label"] == 1.0).sum())
    n_norm = int((df_labeled["true_label"] == 0.0).sum())

    print(f"[3/3] Saved to: {output_path}")
    print(f"\n  Total rows:        {len(df_labeled)}")
    print(f"  Columns:           {len(df_labeled.columns)}")
    print(f"  Normal readings:   {n_norm}")
    print(f"  Anomaly readings:  {n_anom}")
    print(f"  Device types:      {list(DEVICE_TYPES.keys())}")

    logtype_cols = [c for c in df_labeled.columns if c.startswith("LogType_")]
    for col in logtype_cols:
        count = int(df_labeled[col].sum())
        print(f"    {col}: {count} rows")

    print("=" * 60)


if __name__ == "__main__":
    main()
