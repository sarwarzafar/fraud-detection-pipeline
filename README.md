# Real-Time Fintech Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)

An end-to-end Machine Learning pipeline and microservice for credit card fraud detection. Designed to handle severe class imbalance in transaction streams, provide feature interpretability for compliance teams, and serve real-time predictions via a Dockerized FastAPI endpoint.

---

## Project Overview

In financial transaction networks, fraudulent transactions account for a minute fraction of total volume (e.g., ~0.17%). Standard classification accuracy fails to measure performance in such heavily skewed domains. 

This project implements a robust production-ready machine learning pipeline that:
- Preprocesses raw financial data while preserving outlier signals using non-parametric scaling (`RobustScaler`).
- Trains an ensemble classifier optimized for imbalanced target distributions using balanced class weights.
- Evaluates model performance using **Precision-Recall AUC (PR-AUC)** to prioritize high precision (minimizing false positives/customer friction) and actionable recall.
- Provides model interpretability through feature importance analysis to explain model decisions.
- Packages the inference workflow into a lightweight REST API containerized with Docker for seamless deployment.

---

## Key Performance Metrics

Evaluated on a test set of **56,962 transactions** containing 98 fraud cases (preserving the 0.172% fraud distribution):

| Metric | Score | Impact / Meaning |
| :--- | :--- | :--- |
| **Precision (Class 1)** | **90.70%** | When flagged as fraud, the alert is accurate 90.7% of the time (low false positive rate). |
| **Recall (Class 1)** | **79.59%** | Correctly identifies ~80% of all fraudulent transactions automatically. |
| **F1-Score (Class 1)** | **0.8478** | High harmonic mean between Precision and Recall. |
| **PR-AUC Score** | **0.8453** | Demonstrates strong predictive capacity across operational classification thresholds. |

---

## Model Interpretability & Key Risk Drivers

In regulatory financial environments, "black-box" model predictions are insufficient. Feature importance extraction revealed the top 5 predictive drivers influencing fraud likelihood:

1. **V14 (20.56%)** – Primary structural indicator of anomalous transaction behavior.
2. **V10 (12.26%)** – Key latent representation of user profile deviations.
3. **V4 (11.90%)** – Strongly correlated with velocity and high-risk transaction channels.
4. **V17 (8.46%)** – Late-stage risk signal in secondary feature space.
5. **V12 (7.69%)** – Transaction amount and channel pattern marker.

---

## Tech Stack & Architecture

- **Language:** Python 3.11
- **Data & ML Libraries:** Pandas, NumPy, Scikit-Learn, Joblib
- **Visualization:** Matplotlib, Seaborn
- **API Framework:** FastAPI, Uvicorn, Pydantic
- **Containerization & Deployment:** Docker, Docker Compose

---

## Repository Structure

```text
fraud-detection-pipeline/
│
├── .gitignore              # Configured to ignore large datasets and local venv
├── 01_EDA.ipynb            # Data ingestion, preprocessing, training, & interpretability
├── app.py                  # FastAPI REST service for real-time inference
├── Dockerfile              # Container spec for the application
├── docker-compose.yml      # Orchestration config for quick deployment
├── requirements.txt        # Python dependencies
│
└── models/                 # Saved model artifacts (generated locally)
    ├── fraud_model.pkl
    └── scaler.pkl
```

---

## Quick Start & Local Deployment

### Option A: Running locally with Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sarwarzafar/fraud-detection-pipeline.git
   cd fraud-detection-pipeline
   ```

2. **Spin up the API container:**
   ```bash
   docker compose up --build
   ```
   The service will be live at `http://localhost:8000`. Access interactive API documentation at `http://localhost:8000/docs`.

---

### Option B: Running without Docker

1. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Launch FastAPI server:**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

---

## API Endpoint Reference

### POST `/predict`
Evaluates transaction feature vector and returns fraud prediction probability.

#### Request Example:
```json
{
  "features": [-1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 0, 149.62]
}
```

#### Response Example:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0012
}
```