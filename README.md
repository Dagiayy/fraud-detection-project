# 🛡️ Enterprise Fraud Detection System — Adey Innovations Inc.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](https://pytest.org/)

> An end-to-end, leakage-proof machine learning system designed to detect fraudulent transactions in **e-commerce** and **banking** channels. Supports **2 separate execution workflows**: a fast local ML pipeline (Option 1) and a full enterprise production infrastructure stack (Option 2).

---

## 🔀 DUAL EXECUTION WORKFLOW OPTIONS

Choose the execution workflow that matches your environment requirements:

```text
                                  ┌────────────────────────────────────────────────────────┐
                                  │      SELECT EXECUTION WORKFLOW OPTION                  │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                    ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                    ▼                                                                                   ▼
   🔹 OPTION 1: STANDARD ML PIPELINE                                                  🚀 OPTION 2: ENTERPRISE PRODUCTION STACK
   (Fast Local Execution - No Docker Needed)                                          (Docker + API Key Security + Drift + Redis)
   ─────────────────────────────────────────                                          ───────────────────────────────────────────
   • Standard leakage-proof training pipeline                                          • Containerized multi-service via Docker Compose
   • Fast pytest validation suite                                                      • API Key Authentication (X-API-Key header)
   • Lightweight Streamlit control dashboard                                           • SHA-256 PII Hashing for Security Compliance
   • Standard FastAPI microservice (No auth)                                          • Population Stability Index (PSI) Data Drift Audit
                                                                                      • Online Feature Store (Feast/Redis) Scaffolding
                                                                                      • Analyst Review Feedback Active Learning Loop
```

---

## 🚀 1. Workflow Execution Guide

### 🔹 OPTION 1: Standard ML Pipeline Workflow (Fast Local Execution)
Use this option for fast local testing, standard model training, and lightweight UI exploration:

1. **Run Standard Training & Evaluation Pipeline**:
   ```bash
   python scripts/run_pipeline.py --mode standard
   ```

2. **Run Pytest Validation Suite**:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

4. **Launch Standard REST API (No Auth)**:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

---

### 🚀 OPTION 2: Enterprise Production Stack (Docker + Security + MLOps)
Use this option for production deployments, containerized infrastructure, security auditing, and live drift monitoring:

1. **Run Production Pipeline & Data Drift Audit**:
   ```bash
   python scripts/run_pipeline.py --mode production
   ```

2. **Launch Multi-Container Stack via Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   *Services launched:*
   * **FastAPI Secured Microservice**: `http://localhost:8000/docs`
   * **Streamlit Control Dashboard**: `http://localhost:8501`
   * **Redis Feature Store Cache**: `localhost:6379`

3. **Query Secured API Endpoints** (requires `X-API-Key` header):
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: adey-fraud-secret-key-2025" \
        -d '{
              "user_id": 12345,
              "signup_time": "2025-01-01 00:00:00",
              "purchase_time": "2025-01-01 00:05:00",
              "purchase_value": 150.0,
              "age": 30,
              "ip_address": 1234567,
              "source": "Ads",
              "browser": "Chrome",
              "sex": "M"
            }'
   ```

---

## 📌 2. Project Capabilities & Architecture

### Core System Features:
* **Zero Data-Leakage Training**: Applies SMOTE class balancing **strictly on training splits** (`X_train`) after train-test splitting.
* **Domain Feature Engineering**: Extracts point-in-time features: `spending_speed` ($\text{purchase\_value} / \text{time\_since\_signup}$), `time_since_signup`, `is_new_user` ($\le 24\text{h}$), `is_rapid_spender` ($> 95\text{th percentile}$), `is_high_value` ($> \$100$).
* **Automated Fraud Decision Engine**: Evaluates model probabilities and business policy rules to output `ALLOW`, `REVIEW`, or `BLOCK` decisions with risk bands (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and reason codes.
* **Model Explainability (SHAP)**: Individual feature attribution breakdown for every scored transaction.
* **Population Stability Index (PSI) Drift Monitoring**: Detects feature distribution shifts comparing online inference data against baseline reference.
* **Active Learning Feedback Loop**: Analyst verdict recording (`CONFIRMED_FRAUD`, `FALSE_POSITIVE`, `WHITELISTED`) integrated directly into the investigation dashboard.

---

## 📁 3. Repository Architecture

```text
fraud-detection-project/
├── Dockerfile.api                     # Docker image for FastAPI REST Microservice
├── Dockerfile.dashboard               # Docker image for Streamlit Dashboard
├── docker-compose.yml                 # Multi-container orchestration (API + Dashboard + Redis)
├── .dockerignore                      # Docker build ignore file
├── data/
│   ├── raw/                           # Raw datasets (Fraud_Data.csv, IpAddress_to_Country.csv)
│   └── processed/                     # Cleaned preprocessed datasets
├── models/                            # Serialized model artifacts & metadata
│   ├── gbdt_model.pkl                 # Trained GBDT model
│   ├── lightgbm_model.txt             # LightGBM booster model
│   ├── logistic_regression_model.pkl  # Baseline Logistic Regression model
│   └── model_metadata.json            # Model metrics, feature names, & cost analysis
├── src/                               # Canonical Package
│   ├── config/
│   │   └── settings.py                # Dual-workflow config (paths, seeds, API keys, thresholds)
│   ├── data/
│   │   ├── data_contract.py           # Pydantic Transaction schemas
│   │   ├── validation.py              # Data quality & schema validator
│   │   ├── ingestion.py               # Data loader & synthetic fallback
│   │   └── synthetic_generator.py     # Deterministic synthetic dataset generator
│   ├── preprocessing/
│   │   └── preprocessor.py            # Point-in-time cleaning, IP binary search lookup, OHE, scaling
│   ├── features/
│   │   ├── engineering.py             # Velocity, time_since_signup, is_new_user, is_rapid_spender
│   │   ├── registry.py                # Feature metadata registry
│   │   └── feature_store.py           # Feast/Redis online feature store client
│   ├── models/
│   │   ├── train.py                   # Leakage-proof training with SMOTE strictly on X_train
│   │   ├── predict.py                 # Single & batch inference predictor
│   │   └── evaluate.py                # ROC-AUC, AUC-PR, cost-sensitive financial analysis
│   ├── decision/
│   │   ├── rules.py                   # Hard business policy rules
│   │   ├── risk_engine.py             # ALLOW / REVIEW / BLOCK decision engine
│   │   └── feedback.py                # Analyst feedback store for active learning retraining
│   ├── explainability/
│   │   └── shap_explainer.py          # SHAP attribution engine & risk impact calculations
│   ├── api/
│   │   ├── main.py                    # FastAPI REST application
│   │   └── auth.py                    # API Key authentication & X-API-Key security
│   ├── monitoring/
│   │   ├── metrics.py                 # Request telemetry & fraud flag rates
│   │   └── drift.py                   # Population Stability Index (PSI) drift detector
│   └── utils/
│       ├── logger.py                  # Structured JSON logging
│       └── security.py                # SHA-256 PII hashing utility
├── dashboard/                         # Multi-Page Control Center
│   ├── app.py                         # Streamlit entry point with workflow selector
│   └── pages/                         # Multi-page views
│       ├── 1_📊_Overview.py            # Dataset overview & histograms
│       ├── 2_🚨_Fraud_Alerts.py         # Live alert stream filtering BLOCK / REVIEW
│       ├── 3_🔍_Transaction_Investigation.py # CSV upload, SHAP plots, & Analyst Feedback buttons
│       ├── 4_📈_Model_Performance.py   # AUC-PR, ROC curves, confusion matrix, threshold slider
│       ├── 5_📋_Data_Quality.py        # Schema validation metrics & null reports
│       └── 6_⚙️_System_Health.py        # API latency, health status, & metadata
├── scripts/                           # CLI Execution Pipelines
│   ├── run_pipeline.py                # Orchestrator supporting --mode standard & --mode production
│   ├── train_model.py                 # CLI model training script
│   └── evaluate_model.py              # CLI model evaluation script
├── tests/                             # Pytest Suite (14 unit & integration tests)
├── requirements.txt                   # Pinned production dependencies
└── README.md                          # Project documentation
```

---

## 🔌 4. REST API Endpoint Reference

| Method | Endpoint | Security | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/predict` | Header `X-API-Key` | Scores single transaction payload; returns fraud probability, decision (`ALLOW`/`REVIEW`/`BLOCK`), risk band, and SHAP risk factors. |
| `POST` | `/predict/batch` | Header `X-API-Key` | Scores batch array of transaction payloads. |
| `GET` | `/health` | None | Microservice health check status (`HEALTHY`). |
| `GET` | `/ready` | None | Readiness check verifying model artifact availability. |
| `GET` | `/model/info` | Header `X-API-Key` | Returns model metadata, version, decision threshold, and feature schema. |

---

## 📄 License & Attribution

Developed for **Adey Innovations Inc.** under the MIT License.
