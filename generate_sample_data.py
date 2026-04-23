"""
Generate realistic industrial IoT sensor data with injected anomalies.

This script creates a CSV dataset that mimics real industrial sensor readings
with the following anomaly types:
  1. Temperature spikes/drops (sudden extreme values)
  2. Humidity impossible values (>100% or <0%)
  3. Stuck sensors (same value repeated for many consecutive readings)
  4. Future timestamps (dates ahead of the data collection period)
  5. Gradual drift (slow deviation from normal range)

Usage:
    python generate_sample_data.py
    
Output:
    data/raw/iot_sensor_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_normal_data(n_samples=3000, start_date="2026-01-01"):
    """Generate normal industrial sensor readings."""
    np.random.seed(42)
    
    start = pd.Timestamp(start_date)
    # Readings every 5 minutes
    timestamps = [start + timedelta(minutes=5 * i) for i in range(n_samples)]
    
    # Normal temperature: 20-30°C with slight daily cycle
    hours = np.array([t.hour for t in timestamps])
    daily_cycle = 2 * np.sin(2 * np.pi * hours / 24)  # ±2°C daily variation
    temperature = 25 + daily_cycle + np.random.normal(0, 0.5, n_samples)
    
    # Normal humidity: 40-70% with slight inverse correlation to temperature
    humidity = 55 - 0.5 * daily_cycle + np.random.normal(0, 2, n_samples)
    
    # Sensor IDs (3 sensors)
    sensor_ids = np.random.choice(["sensor_01", "sensor_02", "sensor_03"], n_samples)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_id": sensor_ids,
        "temperature": np.round(temperature, 2),
        "humidity": np.round(humidity, 2),
    })
    
    return df


def inject_anomalies(df):
    """Inject various types of anomalies into the dataset."""
    df = df.copy()
    n = len(df)
    np.random.seed(123)
    
    # --- 1. Temperature Spikes (sudden extreme values) ---
    spike_indices = np.random.choice(n, size=30, replace=False)
    for idx in spike_indices:
        if np.random.random() > 0.5:
            df.loc[idx, "temperature"] = np.round(np.random.uniform(80, 120), 2)  # Hot spike
        else:
            df.loc[idx, "temperature"] = np.round(np.random.uniform(-20, -5), 2)   # Cold spike
    
    # --- 2. Humidity Impossible Values ---
    humidity_indices = np.random.choice(
        [i for i in range(n) if i not in spike_indices], size=20, replace=False
    )
    for idx in humidity_indices:
        if np.random.random() > 0.5:
            df.loc[idx, "humidity"] = np.round(np.random.uniform(105, 130), 2)  # >100%
        else:
            df.loc[idx, "humidity"] = np.round(np.random.uniform(-15, -1), 2)   # <0%
    
    # --- 3. Stuck Sensors (same value repeated for 20+ consecutive readings) ---
    # Pick 3 stuck windows
    stuck_regions = [
        (400, 440),   # 40 readings stuck
        (1200, 1235), # 35 readings stuck
        (2200, 2230), # 30 readings stuck
    ]
    for start, end in stuck_regions:
        stuck_temp = df.loc[start, "temperature"]
        stuck_hum = df.loc[start, "humidity"]
        df.loc[start:end, "temperature"] = stuck_temp
        df.loc[start:end, "humidity"] = stuck_hum
    
    # --- 4. Future Timestamps ---
    future_indices = np.random.choice(
        [i for i in range(n) if i not in spike_indices and i not in humidity_indices],
        size=10, replace=False
    )
    for idx in future_indices:
        df.loc[idx, "timestamp"] = df.loc[idx, "timestamp"] + timedelta(days=365)
    
    # --- 5. Gradual Drift (temperature slowly increases over a window) ---
    drift_start, drift_end = 1800, 1860
    drift_values = np.linspace(0, 25, drift_end - drift_start)
    df.loc[drift_start:drift_end - 1, "temperature"] = (
        df.loc[drift_start:drift_end - 1, "temperature"].values + drift_values
    )
    df.loc[drift_start:drift_end - 1, "temperature"] = df.loc[
        drift_start:drift_end - 1, "temperature"
    ].round(2)
    
    return df


def main():
    print("=" * 60)
    print("  IoT Sensor Data Generator")
    print("  Industrial Anomaly Detection Dataset")
    print("=" * 60)
    
    # Generate normal data
    print("\n[1/3] Generating normal sensor data (3000 readings)...")
    df = generate_normal_data(n_samples=3000)
    
    # Inject anomalies
    print("[2/3] Injecting anomalies...")
    df = inject_anomalies(df)
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "iot_sensor_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"[3/3] Saved to: {output_path}")
    print(f"\n  Total rows:        {len(df)}")
    print(f"  Columns:           {list(df.columns)}")
    print(f"  Date range:        {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Temperature range: {df['temperature'].min()} to {df['temperature'].max()}")
    print(f"  Humidity range:    {df['humidity'].min()} to {df['humidity'].max()}")
    print(f"  Sensors:           {df['sensor_id'].unique().tolist()}")
    print(f"\n  Anomaly types injected:")
    print(f"    • Temperature spikes:  30 readings")
    print(f"    • Humidity impossible:  20 readings")
    print(f"    • Stuck sensors:       3 windows (~105 readings)")
    print(f"    • Future timestamps:   10 readings")
    print(f"    • Gradual drift:       60 readings")
    print("=" * 60)


if __name__ == "__main__":
    main()
