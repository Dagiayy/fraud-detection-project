# Fraud Detection Project - Adey Innovations Inc.

## 📌 Overview

This project is designed to detect fraudulent transactions in **e-commerce** and **banking domains**. Fraud detection is critical for minimizing financial loss, maintaining trust, and improving customer experience.

**Objectives:**

* Accurately detect fraudulent transactions using machine learning.
* Handle **highly imbalanced data** typical of fraud datasets.
* Engineer meaningful features to capture temporal, behavioral, and geolocation patterns.
* Provide **interpretable model insights** using SHAP to explain predictions.
* Maintain a **modular, reproducible pipeline** for preprocessing, EDA, feature engineering, model training, and explainability.

---

## 📁 Project Structure

```
fraud-detection-project/
│
├── data/
│   ├── raw/                      # Original datasets (e.g., Fraud_Data.csv, IpAddress_to_Country.csv)
│   └── processed/                # Cleaned, feature-engineered, and preprocessed data
│
├── src/
│   ├── utils/
│   │   ├── preprocessor.py       # Full preprocessing pipeline: cleaning, encoding, scaling, class balancing
│   │   ├── feature_engineering.py # Functions for time features, velocity, transaction frequency
│   │   ├── eda_utils.py          # Functions to plot fraud distributions, correlation heatmaps, PCA
│   │   └── refactored_preprocessor.py # Modular, refactored preprocessing class
│   │
│   ├── models/
│   │   └── train_models.py       # Model training, evaluation, and serialization
│   │
│   └── notebooks/
│       ├── 01_eda_fraud_data.ipynb        # Exploratory Data Analysis and Feature Engineering
│       ├── 02_model_training.ipynb        # Model training and comparison (Logistic Regression, LightGBM)
│       └── 03_model_explainability.ipynb  # SHAP-based model interpretability
│
├── tests/                          # Unit tests for preprocessing, feature engineering
│
├── reports/                        # Project write-ups, analysis reports, insights
├── models/                         # Serialized models (.pkl, .txt)
├── requirements.txt                # Python dependencies
└── README.md                        # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd fraud-detection-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt --timeout 100
```

> **Note:** If handling large CSVs, ensure enough RAM or consider using **sample datasets** in `data/raw/` for testing.

---

## 🚀 Usage Guide

### 1. Data Preprocessing

Run the preprocessing script to clean, transform, and prepare data for modeling:

```bash
python src/utils/run_preprocessing.py \
    --fraud_path data/raw/Fraud_Data.csv \
    --ip_path data/raw/IpAddress_to_Country.csv \
    --credit_path data/raw/creditcard.csv \
    --out data/processed/enhanced_processed_fraud_data.csv \
    --balance smote
```

**Pipeline transformations include:**

* Filling missing values & dropping duplicates.
* **Datetime features:** Extract `hour_of_day`, `day_of_week`, `time_since_signup`.
* **Velocity:** `spending_speed = purchase_value / time_since_signup`.
* **Transaction frequency:** Count of transactions per user.
* **IP-to-country enrichment** for geolocation.
* **Categorical encoding:** One-hot encoding for source, browser, sex, country.
* **Scaling numerical features**.
* **Class balancing** with SMOTE (or undersampling).

➡️ Output file: `data/processed/enhanced_processed_fraud_data.csv`

---

### 2. Exploratory Data Analysis & Feature Engineering

Visualize dataset characteristics and engineer additional features:

```bash
python src/utils/run_eda.py --data data/processed/enhanced_processed_fraud_data.csv
```

**Key insights:**

* Temporal fraud trends (hourly, daily, weekly)
* Fraud counts and rates by country
* Distribution of behavioral features (`spending_speed`, `transaction_frequency`)
* Correlation heatmaps for numeric features
* PCA analysis for credit card features (V1-V28)

> These EDA steps support **feature engineering** for robust modeling.

---

### 3. Model Training & Evaluation

Train, evaluate, and save classification models:

```bash
jupyter notebook src/notebooks/02_model_training.ipynb
```

**Models trained:**

* Logistic Regression
* LightGBM (best performance)

**Evaluation metrics:**

* Precision, Recall, F1-Score (fraud recall prioritized)
* ROC-AUC & PR-AUC

**Serialized models:**

* `models/logistic_regression_model.pkl`
* `models/lightgbm_model.txt`

> LightGBM achieved the best trade-off between recall and precision for fraud detection.

---

### 4. Model Explainability

Use SHAP to interpret model decisions:

```bash
jupyter notebook src/notebooks/03_model_explainability.ipynb
```

**Insights:**

* SHAP summary plots highlight **global feature importance**.
* SHAP force plots show **individual prediction contributions**.
* Fraud likelihood increases with:

  * High transaction amounts
  * Short time since signup
  * Certain countries or device/browser behaviors

> Model transparency ensures compliance with regulations and builds trust.

---

## 🔍 Key Features

* **Class Imbalance Handling:** SMOTE/undersampling applied carefully.
* **Modular Feature Engineering:** Temporal, behavioral, and geolocation signals.
* **Explainable ML:** SHAP visualizations for transparency.
* **Reproducibility:** Scripts, notebooks, and tests ensure workflow can be rerun easily.
* **Unit Testing:** `tests/` folder contains smoke tests for preprocessing pipeline.

---

## 🧩 Dependencies

* Python 3.8+
* pandas, numpy
* scikit-learn
* lightgbm
* shap
* imbalanced-learn
* matplotlib, seaborn
* jupyter

Install via:

```bash
pip install -r requirements.txt
```

---

## 📝 Notes & Best Practices

* Large CSVs (`creditcard.csv`, `enhanced_processed_fraud_data.csv`) **exceed GitHub limits**. Use Git LFS or keep datasets locally.
* SMOTE is applied **only to training sets** to prevent data leakage.
* High-cardinality columns (`user_id`, `device_id`, `ip_address`) are dropped or transformed to avoid memory issues.
* All preprocessing, EDA, and modeling steps are automated for reproducibility.

---

## 📈 Results

* **LightGBM:** Best recall and AUC for detecting fraud.
* **SHAP analysis:** Confirmed alignment of features with real-world fraud patterns.
* Model achieves balance between **accuracy, recall, and interpretability**.

---

## 🧪 Testing

Run unit tests to validate preprocessing and feature engineering:

```bash
pytest tests/
```

Tests cover:

* Preprocessor pipeline (smoke tests)
* Feature engineering functions
* Output consistency

---

