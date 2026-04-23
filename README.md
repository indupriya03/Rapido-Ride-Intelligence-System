# ⚡ Rapido Ride Intelligence System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-End--to--End-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-red?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-Tuned-brightgreen)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![Status](https://img.shields.io/badge/Status-Completed-success)

> An end-to-end machine learning system for ride-hailing intelligence — covering cancellation prediction, fare forecasting, demand classification, and driver performance analysis — backed by a live inference engine and an interactive Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Use Cases](#use-cases)
- [Model Performance](#model-performance)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Dashboard](#dashboard)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Tech Stack](#tech-stack)

---

## Overview

Ride-hailing platforms experience dynamic, real-time fluctuations driven by time, location, weather, traffic, and driver behavior. This project builds a production-style ML system to tackle four distinct prediction problems, all derived from Rapido's operational data:

- Predict whether a ride will be completed, cancelled, or incomplete
- Forecast the fare for each vehicle type
- Classify demand zones as high or low
- Detect whether a driver is likely to be delayed

Each use case has its own dedicated feature set, trained model, Optuna-tuned variant, and calibrated probability threshold — all served through a unified inference API and a branded analytics dashboard.

---

## Use Cases

| # | Use Case | Task | Target |
|---|----------|------|--------|
| UC1 | Booking Outcome | Multi-class Classification | Completed / Cancelled / Incomplete |
| UC2 | Fare Forecasting | Regression (per vehicle) | Predicted fare (Cab / Auto / Bike) |
| UC3 | Cancellation Risk | Binary Classification | Cancelled or Not |
| UC4 | Driver Delay Risk | Binary Classification | On Time or Delayed |

All classification models output calibrated probabilities and apply optimized decision thresholds (UC3: 0.749, UC4: 0.700) to maximize operational usefulness.

---

## Model Performance

| Use Case | Metric | Baseline | Tuned | Winner |
|----------|--------|----------|-------|--------|
| UC1 – Booking Outcome | Weighted F1 | 0.7473 | **0.7486** | Tuned |
| UC2 – Fare (Cab) | R² | **0.9949** | — | Baseline |
| UC2 – Fare (Auto) | R² | **0.9953** | — | Baseline |
| UC2 – Fare (Bike) | R² | **0.9955** | — | Baseline |
| UC3 – Cancellation Risk | ROC-AUC | 0.8637 | **0.8640** | Tuned |
| UC4 – Driver Delay | ROC-AUC | 0.9549 | **0.9997** | Tuned |

UC2 fare models achieve near-perfect regression accuracy (R² > 0.994) out of the box — no tuning required. UC4 sees a dramatic jump from 0.95 to **0.9997 ROC-AUC** after Optuna hyperparameter search.

---

## Architecture

```
Raw CSV Data
     │
     ▼
Database Setup           ← scripts/setup_database.py
     │
     ▼
EDA & Cleaning           ← notebooks/1_data_profiling.ipynb
                            notebooks/2_data_cleaning.ipynb
                            notebooks/3_eda.ipynb
     │
     ▼
Feature Engineering      ← src/feature_engineering/
  Zone 1: base features  (temporal, behavioral, driver stats)
  Zone 2: config/join    (cross-table aggregations)
  Zone 3: final pipeline (scaling, encoding, SHAP feature selection)
     │
     ▼
Model Selection          ← run_model_selection.py
  Logistic Regression, Random Forest, XGBoost, LightGBM
     │
     ▼
Hyperparameter Tuning    ← run_tuning.py  (Optuna, 100 trials)
     │
     ▼
Evaluation & Artifacts   ← outputs/  (confusion matrices, feature importance)
     │
     ▼
Batch Inference          ← run_predict.py → scripts/insert_predictions.py
     │
     ▼
Streamlit Dashboard      ← app/app.py
  Overview · Predictions · Analytics · Strategy
```

---

## Project Structure

```
Rapido-Ride-Intelligence-System/
│
├── app/                         # Streamlit dashboard
│   ├── app.py                   # Entry point + theme engine
│   ├── sections/
│   │   ├── overview.py          # KPI cards, city/hour charts
│   │   ├── predictions.py       # Batch results + live inference tab
│   │   ├── analytics.py         # Deep EDA visualizations
│   │   └── strategy.py          # Business recommendations
│   └── utils/
│       ├── charts.py            # Reusable Plotly chart helpers
│       ├── db.py                # SQLAlchemy query runner
│       ├── queries.py           # All SQL queries
│       └── theme.py             # Dark / light CSS injector
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering/     # Zone 1–3 pipeline modules
│   ├── modeling/                # Model definitions, trainers, selection, I/O
│   ├── tuning/                  # Optuna tuners for XGBoost & LightGBM
│   └── inference/
│       ├── predictor.py         # End-to-end single-row inference
│       └── preprocessor.py      # Row preprocessing for live prediction
│
├── notebooks/                   # Data profiling, cleaning, EDA
├── data/
│   ├── raw/                     # Original source CSVs
│   └── cleaned/                 # Post-cleaning datasets
│
├── models/                      # Trained .pkl models + scalers + thresholds
├── outputs/                     # Metrics, confusion matrices, feature plots
├── images/                      # EDA visualizations (100+ charts)
├── scripts/                     # DB setup + batch prediction insert
│
├── run_feature_engineering.py
├── run_model_selection.py
├── run_training.py
├── run_tuning.py
├── run_predict.py
└── requirements.txt
```

---

## ML Pipeline

### Feature Engineering

Features are built across three processing zones:

- **Zone 1** — Base features: temporal encodings (hour, weekday, peak flag, season), driver stats (acceptance rate, delay rate, incomplete rate), customer stats (tenure, cancel flag), and ride attributes (distance, surge, wait time).
- **Zone 2** — Aggregation config: cross-table joins merging bookings, customers, drivers, and location demand signals.
- **Zone 3** — Final pipeline: standard scaling, categorical encoding, SHAP-guided feature selection, and per-use-case train/test splits.

Each use case trains on ~40 engineered features selected via correlation analysis and SHAP importance.

### Model Selection

Four model families are evaluated per use case on held-out validation data:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

The best-performing architecture per use case is selected and saved as `*_baseline.pkl`.

### Hyperparameter Tuning

Optuna runs 100 trials per model (XGBoost + LightGBM) for UC1, UC3, and UC4 using Bayesian optimization. The best parameters are saved to `outputs/tuned_params.json` and retrained models saved as `*_tuned.pkl`. Final models (`*_final.pkl`) are selected by comparing baseline vs. tuned metrics.

### Inference

`src/inference/predictor.py` provides a clean `predict(row, use_case)` API:

```python
from src.inference.predictor import predict

result = predict(
    row={"city": "Bangalore", "vehicle_type": "Cab", "hour": 18, ...},
    use_case="UC3",
    return_proba=True,
)
# {'use_case': 'UC3', 'prediction': 1, 'label': 'Cancelled',
#  'threshold': 0.749, 'probability': 0.812}
```

Supports all six use cases. UC3 and UC4 apply stored probability thresholds instead of the default 0.5 cutoff.

### Explainability

Every trained model stores SHAP values (`shap_importance_*.pkl`) and correlation reports (`corr_report_*.pkl`) for post-hoc analysis and feature validation.

---

## Dashboard

```bash
streamlit run app/app.py
```

The dashboard is organized into four pages:

| Page | Contents |
|------|----------|
| 🏠 Overview | Total rides, avg fare, cancellation rate, avg surge, hourly trends, city breakdown, vehicle performance |
| 🔮 Predictions | Batch prediction KPIs, cancellation risk tiers, fare accuracy by city, live single-ride inference form |
| 📊 Analytics | Deep EDA charts — correlation heatmaps, weather/traffic/peak pattern analysis, rating distributions |
| 🎯 Strategy | Data-driven business recommendations for driver allocation, surge pricing, and retention |

Supports dark/light mode toggle with a fully themed Plotly + CSS layer.

---

## Installation

```bash
git clone https://github.com/indupriya03/Rapido-Ride-Intelligence-System.git
cd Rapido-Ride-Intelligence-System
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `shap`, `imbalanced-learn`, `streamlit`, `sqlalchemy`, `matplotlib`, `seaborn`, `joblib`

---

## Running the Pipeline

```bash
# 1. Run data cleaning notebook (produces data/cleaned/)
jupyter nbconvert --to notebook --execute notebooks/2_data_cleaning.ipynb

# 2. Set up the database (requires cleaned data)
python scripts/setup_database.py

# 3. Build features
python run_feature_engineering.py

# 4. Select best baseline model
python run_model_selection.py

# 5. Train final baselines
python run_training.py

# 6. Tune hyperparameters (Optuna)
python run_tuning.py

# 7. Generate predictions & insert to DB
python run_predict.py
python scripts/insert_predictions.py

# 8. Launch dashboard
streamlit run app/app.py
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Imbalanced Classes | imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Storage | SQLAlchemy + SQL |
| Serialization | Joblib |

---

## Author

**Indupriya Chidambararaj**  
Machine Learning Engineering Project — Rapido Ride Intelligence System
