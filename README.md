# 🛡️ Enterprise Fraud Detection System — Adey Innovations Inc.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end, leakage-proof machine learning system designed to detect fraudulent transactions in **e-commerce** and **banking** channels. Features a production-ready **FastAPI REST Service**, an automated **Fraud Decision Engine**, **SHAP Explainability**, a **Multi-Page Streamlit Dashboard**, and a **Pytest Validation Suite**.

---

## 📌 1. Project Overview & Business Vision

Financial fraud represents a multi-billion dollar threat to digital commerce. For **Adey Innovations Inc.**, maintaining high-precision fraud detection without disrupting legitimate user transactions is essential for operational integrity and brand trust.

### Key Capabilities:
* **Leakage-Proof Machine Learning**: Applies SMOTE class balancing **strictly on training splits** (`X_train`) to prevent synthetic data contamination and ensure realistic model validation.
* **Domain Signal Feature Engineering**: Extracts temporal behavior (`time_since_signup`, `hour_of_day`), velocity (`spending_speed`), and explicit business flags (`is_new_user`, `is_rapid_spender`, `is_high_value`).
* **Automated Fraud Decision Engine**: Maps model probabilities and business policy rules into deterministic action decisions (`ALLOW`, `REVIEW`, `BLOCK`) with risk bands (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and reason codes.
* **Model Transparency via SHAP**: Provides feature contribution attributions for individual transaction risk scoring to fulfill regulatory compliance.
* **Real-Time REST API Serving**: High-performance FastAPI endpoints (`/predict`, `/predict/batch`) for payment gateway integration.
* **Interactive Control Dashboard**: Multi-page Streamlit application with live CSV batch scoring, transaction inspection, risk alert streams, and threshold tuning sliders.

---

## 📁 2. Repository Architecture

```text
fraud-detection-project/
├── data/
│   ├── raw/                           # Original datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│   └── processed/                     # Cleaned, feature-engineered preprocessed datasets
├── models/                            # Trained model artifacts & metadata
│   ├── gbdt_model.pkl                 # Serialized GBDT classifier model
│   ├── lightgbm_model.txt             # LightGBM booster artifact
│   ├── logistic_regression_model.pkl  # Baseline Logistic Regression model
│   └── model_metadata.json            # Model metrics, feature names, & cost analysis
├── src/                               # Core Production Package
│   ├── config/                        # Settings & central configuration
│   │   └── settings.py                # Paths, thresholds ($0.35$), seeds ($42$), cost params
│   ├── data/                          # Data Contracts & Quality Validation
│   │   ├── data_contract.py           # Pydantic Transaction schemas
│   │   ├── validation.py              # Data quality & schema validator
│   │   ├── ingestion.py               # Raw data loader & synthetic fallback
│   │   └── synthetic_generator.py     # Deterministic synthetic transaction fixture generator
│   ├── preprocessing/                 # Data Preprocessing Pipeline
│   │   └── preprocessor.py            # Point-in-time cleaning, binary search IP lookup, scaling, OHE
│   ├── features/                      # Feature Engineering & Registry
│   │   ├── engineering.py             # Velocity, time_since_signup, is_new_user, is_rapid_spender
│   │   └── registry.py                # Feature metadata registry
│   ├── models/                        # Model Training & Inference Engine
│   │   ├── train.py                   # Leakage-proof training with SMOTE strictly on X_train
│   │   ├── predict.py                 # Single & batch inference predictor
│   │   └── evaluate.py                # ROC-AUC, AUC-PR, cost-sensitive financial analysis
│   ├── decision/                      # Fraud Decision Engine
│   │   ├── rules.py                   # Hard business policy rules
│   │   └── risk_engine.py             # ALLOW / REVIEW / BLOCK decision & risk band engine
│   ├── explainability/                # Explainable AI (XAI)
│   │   └── shap_explainer.py          # SHAP attribution engine & risk impact calculations
│   ├── api/                           # Production Serving Microservice
│   │   └── main.py                    # FastAPI REST application
│   ├── monitoring/                    # Telemetry & Observability
│   │   └── metrics.py                 # Request telemetry & fraud flag rates
│   └── utils/                         # Utilities
│       └── logger.py                  # Structured JSON logging
├── dashboard/                         # Multi-Page Control Center
│   ├── app.py                         # Streamlit entry point with cached dataset reads
│   └── pages/                         # Multi-page views
│       ├── 1_📊_Overview.py            # Dataset overview & histograms
│       ├── 2_🚨_Fraud_Alerts.py         # Live alert stream filtering BLOCK / REVIEW
│       ├── 3_🔍_Transaction_Investigation.py # CSV upload for live predictions & SHAP plots
│       ├── 4_📈_Model_Performance.py   # AUC-PR, ROC curves, confusion matrix, threshold slider
│       ├── 5_📋_Data_Quality.py        # Schema validation metrics & null reports
│       └── 6_⚙️_System_Health.py        # API latency, health status, & metadata
├── scripts/                           # CLI Execution Pipelines
│   ├── run_pipeline.py                # 1-Command end-to-end pipeline orchestrator
│   ├── train_model.py                 # CLI model training script
│   └── evaluate_model.py              # CLI model evaluation script
├── tests/                             # Pytest Suite
│   ├── unit/                          # Unit tests for validation, preprocessor, features, decision, API
│   ├── integration/                   # End-to-end pipeline integration tests
│   └── fixtures/                      # Test transaction fixtures
├── requirements.txt                   # Pinned production dependencies
└── README.md                          # Project documentation
```

---

## ⚙️ 3. Installation & Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Dagiayy/fraud-detection-project.git
cd fraud-detection-project
git checkout production-fraud-system
```

### 2. Create and Activate Virtual Environment
```bash
# Using venv
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Production Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 4. Usage & Execution Commands

### A. Run Full End-to-End Pipeline Orchestrator
Executes data generation/loading, preprocessor transformation, leakage-proof training (SMOTE on train set only), evaluation, and model artifact serialization:
```bash
python scripts/run_pipeline.py
```

### B. Run Pytest Automated Validation Suite
Executes unit tests and integration tests:
```bash
python -m pytest tests/ -v
```

### C. Launch Real-Time FastAPI REST Microservice
Starts the production API server at `http://127.0.0.1:8000`:
```bash
uvicorn src.api.main:app --reload --port 8000
```
> **Swagger Interactive API Documentation**: Visit `http://127.0.0.1:8000/docs` in your browser.

### D. Launch Multi-Page Control Dashboard
Starts the Streamlit web dashboard:
```bash
streamlit run dashboard/app.py
```

---

## 🔌 5. REST API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Scores a single transaction payload; returns probability score, decision (`ALLOW`/`REVIEW`/`BLOCK`), risk band, and SHAP risk factors. |
| `POST` | `/predict/batch` | Scores a batch array of transaction payloads. |
| `GET` | `/health` | Microservice health check status (`HEALTHY`). |
| `GET` | `/ready` | Readiness check verifying model artifact availability. |
| `GET` | `/model/info` | Returns model metadata, version, decision threshold, and feature schema. |

### Sample `POST /predict` Request Payload:
```json
{
  "user_id": 12345,
  "signup_time": "2025-01-01 00:00:00",
  "purchase_time": "2025-01-01 00:05:00",
  "purchase_value": 150.0,
  "age": 30,
  "ip_address": 1234567,
  "source": "Ads",
  "browser": "Chrome",
  "sex": "M"
}
```

### Sample `POST /predict` Response:
```json
{
  "transaction_id": "8f3b2a19-4c12-4e90-b1d2-098765432100",
  "timestamp": "2025-01-01T00:05:01",
  "score_results": {
    "decision": "BLOCK",
    "risk_band": "CRITICAL",
    "fraud_probability": 0.8645,
    "threshold_used": 0.35,
    "reason_codes": [
      "RULE_INSTANT_PURCHASE_AFTER_SIGNUP",
      "MODEL_HIGH_FRAUD_PROBABILITY_(0.865)"
    ],
    "hard_rule_triggered": true
  },
  "explanation": {
    "top_positive_risk_factors": [
      "spending_speed",
      "is_new_user",
      "is_high_value"
    ],
    "top_negative_protective_factors": [
      "time_since_signup"
    ]
  }
}
```

---

## 📈 6. Financial Cost-Sensitive Evaluation Framework

The system incorporates a business-oriented cost evaluation model beyond simple accuracy:

$$\text{Net Savings} = \text{Baseline Fraud Losses} - \left( \text{FP} \times C_{\text{FP}} + \text{FN} \times C_{\text{FN}} + (\text{FP} + \text{TP}) \times C_{\text{Review}} \right)$$

* **$C_{\text{FP}} = \$10.00$**: False positive friction cost.
* **$C_{\text{FN}} = \$150.00$**: Average uncaptured fraud loss.
* **$C_{\text{Review}} = \$5.00$**: Manual analyst investigation cost.

---

## 📄 License & Attribution

Developed for **Adey Innovations Inc.** under the MIT License.
