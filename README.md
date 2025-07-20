
# Fraud Detection Project - Adey Innovations Inc.

## Overview

This project aims to build robust, interpretable machine learning models to detect fraudulent activities in e-commerce and banking transactions. The core objectives include:

* Enhancing fraud detection accuracy.
* Tackling class imbalance effectively.
* Engineering meaningful and insightful features.
* Maintaining a balance between security and user experience.

---

## Project Structure

```
fraud-detection-project/
│
├── data/
│   ├── raw/                     # Original datasets (e.g., Fraud_Data.csv, IpAddress_to_Country.csv)
│   └── processed/               # Cleaned, feature-engineered, and preprocessed datasets
│
├── src/
│   ├── utils/
│   │   └── preprocessor.py      # Preprocessing pipeline: cleaning, encoding, scaling, and balancing
│   │
│   ├── models/
│   │   └── train_models.py      # Model training and evaluation logic (coming soon)
│   │
│   └── notebooks/
│       └── 01_eda_fraud_data.ipynb   # EDA and feature engineering notebook
│
├── reports/                     # Visualizations, analysis results, and project reports
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation (this file)
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd fraud-detection-project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Use

### Preprocess the Data

Run the preprocessing pipeline to clean, encode, scale, and balance your dataset:

```bash
python src/utils/preprocessor.py
```

* Inputs: `data/raw/Fraud_Data.csv`
* Outputs: `data/processed/processed_fraud_data.csv`
* Includes: datetime handling, SMOTE oversampling, encoding (with high-cardinality management), and scaling

---

### Exploratory Data Analysis

Launch the EDA notebook:

```bash
jupyter notebook src/notebooks/01_eda_fraud_data.ipynb
```

* Understand feature distributions, trends, and anomalies
* Engineer features like:

  * Time since signup/login
  * IP-to-country mapping
  * Device/browser indicators

---

### Model Training (Coming Soon)

* Will include training with:

  * Logistic Regression
  * Random Forest
  * LightGBM
* Evaluation with:

  * AUC-PR
  * F1-score
* Explanation with:

  * SHAP (model interpretability)

---

## Key Features

* Class Imbalance Handling: SMOTE + undersampling
* Feature Engineering: Time-based, geolocation, and behavior-derived features
* Modular Architecture: Clean separation of code components
* Explainability-First: SHAP integration for interpretability (planned)

---

## Requirements

* Python 3.8+
* `pandas`, `scikit-learn`, `imbalanced-learn`
* `matplotlib`, `seaborn`
* `jupyter`

See full list in `requirements.txt`.

---

## Notes

* High-cardinality columns (e.g., `user_id`, `ip_address`) are dropped to reduce memory usage.
* Avoid applying SMOTE on the test set to prevent data leakage.
* The modeling component is still under development.

---
