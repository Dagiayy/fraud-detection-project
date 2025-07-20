# Fraud Detection Project - Adey Innovations Inc.

## Overview

This project aims to build robust fraud detection models for e-commerce and banking transactions. The main objectives include improving fraud detection accuracy while balancing security and user experience by addressing class imbalance, engineering meaningful features, and applying interpretable machine learning models.

---

## Project Structure

```
fraud-detection-project/
│
├── data/
│   ├── raw/                  # Original datasets (e.g., Fraud_Data.csv, IpAddress_to_Country.csv)
│   └── processed/            # Cleaned, feature-engineered, and preprocessed data ready for modeling
│
├── src/
│   ├── utils/
│   │   └── preprocessor.py   # Data cleaning, feature encoding, scaling, and class balancing pipeline
│   │
│   ├── models/
│   │   └── train_models.py   # Model training, evaluation, and persistence (to be implemented)
│   │
│   └── notebooks/
│       └── 01_eda_fraud_data.ipynb  # Exploratory Data Analysis and Feature Engineering
│
├── reports/                  # Project reports, analyses, and visualizations
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation (this file)
```

---

## Setup Instructions

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd fraud-detection-project
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preprocessing

Run the preprocessing script to clean, encode, scale, and balance the dataset:

```bash
python src/utils/preprocessor.py
```

* This script loads raw or intermediate processed data (`data/raw/Fraud_Data.csv` or `data/processed/processed_fraud_data.csv`).
* It performs transformations including handling datetime features, encoding categorical variables (while managing high-cardinality columns), scaling numeric features, and balancing classes using SMOTE.
* The final processed dataset is saved in `data/processed/processed_fraud_data.csv` for downstream modeling.

### 2. Exploratory Data Analysis & Feature Engineering

Launch the Jupyter notebook for detailed EDA and feature engineering:

```bash
jupyter notebook src/notebooks/01_eda_fraud_data.ipynb
```

* Investigate data distributions, missing values, and correlations.
* Engineer new features such as time-based metrics (hour of day, day of week, time since signup) and geolocation mapping from IP addresses.

### 3. Model Training and Evaluation *(Planned)*

* Scripts will train machine learning models such as Logistic Regression, Random Forest, and LightGBM.
* Models will be evaluated with metrics appropriate for imbalanced datasets like AUC-PR and F1-Score.
* SHAP (SHapley Additive exPlanations) will be integrated for model interpretability and explainability.

---

## Key Features

* **Class Imbalance Handling:** Implements SMOTE oversampling and random undersampling to address imbalanced fraud classes.
* **Robust Feature Engineering:** Includes temporal features, IP-based geolocation, and careful encoding of categorical variables while managing high-cardinality features to avoid memory issues.
* **Modular Pipeline:** Clean separation of preprocessing, exploratory analysis, and modeling components for maintainability and extensibility.
* **Explainable AI:** Planned integration of SHAP to make model decisions interpretable.

---

## Dependencies

* Python 3.8+
* pandas
* scikit-learn
* imbalanced-learn
* matplotlib, seaborn (for visualization in notebooks)
* Jupyter Notebook

Full dependency list in `requirements.txt`.

---

## Notes

* The preprocessing script drops high-cardinality columns like `user_id`, `device_id`, and `ip_address` during encoding to prevent memory overflow issues.
* Always apply data balancing (e.g., SMOTE) **only on training data** during model development to avoid data leakage.
* This project is a work in progress — model training and deployment scripts will be added in upcoming iterations.

---


