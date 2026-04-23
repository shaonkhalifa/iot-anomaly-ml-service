# IoT Anomaly Detection - ML Service

This repository contains the machine learning component of the MSc Project on **Unsupervised Machine Learning for Industrial Sensor Anomaly Detection**.

## 🚀 Features
- **Data Preprocessing**: Advanced 9-stage pipeline including domain validation and feature engineering.
- **Unsupervised Models**:
  - **Isolation Forest**: Contamination-based anomaly detection.
  - **One-Class SVM**: Boundary-based outlier detection.
  - **K-Means Clustering**: Centroid-distance based anomaly flagging.
- **REST API**: Flask-based endpoints for real-time predictions.

## 🛠️ Tech Stack
- Python 3.10+
- Scikit-learn
- Pandas / Numpy
- Flask

## 🏁 Quick Start
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Train models
python train.py

# Run API
python -m flask --app app.routes run --port 5001
```

## 📊 Endpoints
- `GET /health`: Model status and health check.
- `POST /predict`: Single data point prediction.
- `POST /predict/compare`: Compare results across all 3 models.
