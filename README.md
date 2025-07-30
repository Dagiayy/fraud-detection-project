
# Fraud Detection Project - Adey Innovations Inc.

## Overview

This project aims to build robust fraud detection models for e-commerce and banking transactions. The objectives include improving fraud detection accuracy while balancing security and user experience by:
- Addressing class imbalance,
- Engineering meaningful features, and
- Applying interpretable machine learning models using SHAP.

---

## Project Structure

```

fraud-detection-project/
│
├── data/
│   ├── raw/                      # Original datasets (e.g., Fraud\_Data.csv, IpAddress\_to\_Country.csv)
│   └── processed/                # Cleaned, feature-engineered, and preprocessed data
│
├── src/
│   ├── utils/
│   │   └── preprocessor.py       # Data cleaning, feature encoding, scaling, and balancing pipeline
│   │
│   ├── models/
│   │   └── train\_models.py       # Model training and evaluation pipeline
│   │
│   └── notebooks/
│       ├── 01\_eda\_fraud\_data.ipynb        # EDA and Feature Engineering
│       ├── 02\_model\_training.ipynb        # Model training and comparison (Logistic Regression, LightGBM)
│       └── 03\_model\_explainability.ipynb  # SHAP-based model interpretability
│
├── reports/                    # Project write-ups, insights, and submission reports
├── models/                     # Serialized models (e.g., .pkl, .txt)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)

````

---

## Setup Instructions

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd fraud-detection-project
````

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

Key processing steps:

* Datetime feature extraction
* IP-based geolocation enrichment
* Categorical encoding (with high-cardinality filtering)
* Scaling and SMOTE-based balancing

Final output: `data/processed/processed_fraud_data.csv`

---

### 2. Exploratory Data Analysis & Feature Engineering

Run the EDA notebook:

```bash
jupyter notebook src/notebooks/01_eda_fraud_data.ipynb
```

Insights include:

* Distributions of transaction types, devices, and categories
* Fraud patterns across time, location, and user behavior
* Engineered features such as:

  * `hour`, `day_of_week`, `time_since_signup`
  * Mapped country from IP address

---

### 3. Model Training and Evaluation

Train and evaluate classification models:

```bash
jupyter notebook src/notebooks/02_model_training.ipynb
```

Models include:

* Logistic Regression
* LightGBM (selected as best performer)

Metrics:

* ROC-AUC, PR-AUC
* F1-Score, precision, recall (with focus on fraud class)

Best model is saved to:

* `models/lightgbm_model.txt`
* `models/logistic_regression_model.pkl`

---

### 4. Model Explainability (SHAP)

Visualize and explain model predictions using SHAP:

```bash
jupyter notebook src/notebooks/03_model_explainability.ipynb
```

Interpretability analysis includes:

* **SHAP Summary Plot**: Global feature importance across all predictions
* **SHAP Force Plot**: Local explanation of individual fraud predictions

Key insights:

* Certain features (e.g., transaction amount, signup day, country risk level) significantly influence fraud detection
* Force plots provide transparency into how the model flags individual transactions

---

## Key Features

* ✅ **Class Imbalance Handling**: Uses SMOTE and undersampling techniques
* ✅ **Feature Engineering**: Includes geolocation, temporal, and behavior-based features
* ✅ **Interpretable Models**: SHAP-based insights into LightGBM decision-making
* ✅ **Modular Pipeline**: Preprocessing, EDA, modeling, and explainability are separated for clarity

---

## Dependencies

* Python 3.8+
* pandas, numpy
* scikit-learn
* lightgbm
* imbalanced-learn
* shap
* matplotlib, seaborn
* jupyter

All listed in `requirements.txt`

---

## Notes

* `user_id`, `device_id`, and `ip_address` are excluded to avoid overfitting and memory overload during encoding.
* Class balancing is only applied on the training set to prevent leakage.
* All modeling and SHAP explainability steps are now complete.

---
