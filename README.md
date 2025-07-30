
# Fraud Detection Project - Adey Innovations Inc.

## 📌 Overview

This project aims to build robust fraud detection models for e-commerce and banking transactions. The objectives include improving fraud detection accuracy while balancing security and user experience by:
- Addressing class imbalance,
- Engineering meaningful features, and
- Applying interpretable machine learning models using SHAP.

---

## 📁 Project Structure

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

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd fraud-detection-project
````

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt --timeout 100
```

---

## 🚀 Usage Guide

### 1. Data Preprocessing

Run the preprocessing script to clean, encode, scale, and balance the dataset:

```bash
python src/utils/preprocessor.py
```

Key transformations:

* Datetime feature extraction
* IP-to-country enrichment
* One-hot or label encoding for categorical variables
* Feature scaling and class balancing via SMOTE

➡️ Final output: `data/processed/processed_fraud_data.csv`

---

### 2. Exploratory Data Analysis & Feature Engineering

Launch the notebook to visualize data insights and engineer new features:

```bash
jupyter notebook src/notebooks/01_eda_fraud_data.ipynb
```

Highlights:

* Temporal fraud patterns (e.g., weekend, signup delay)
* Transaction volume per country and category
* Creation of behavior-driven features:

  * `hour`, `day_of_week`, `time_since_signup`, etc.
  * Country risk derived from IP address

---

### 3. Model Training and Evaluation

Train and evaluate classification models in:

```bash
jupyter notebook src/notebooks/02_model_training.ipynb
```

Models used:

* Logistic Regression
* LightGBM (selected best model)

Evaluation metrics:

* Precision, Recall, F1-Score (emphasis on recall for fraud)
* ROC-AUC and PR-AUC

📁 Models saved as:

* `models/logistic_regression_model.pkl`
* `models/lightgbm_model.txt`

---

### 4. Model Explainability with SHAP

Interpret model behavior using SHAP:

```bash
jupyter notebook src/notebooks/03_model_explainability.ipynb
```

Visuals and insights:

* **SHAP Summary Plot**: Global feature importance
* **SHAP Force Plot**: How individual features push predictions toward fraud or non-fraud

Key Findings:

* Fraud likelihood increases with:

  * Higher transaction amounts
  * Shorter time since signup
  * Certain risky countries or device behaviors

SHAP enhances trust and compliance by offering transparency into decision logic.

---

## 🔍 Key Features

* ✅ **Class Imbalance Handling**: SMOTE and undersampling applied only to training sets
* ✅ **Robust Feature Engineering**: Incorporates time, geolocation, and behavioral signals
* ✅ **Explainability**: SHAP plots for local and global interpretability
* ✅ **Modular Workflow**: Clean separation of scripts, notebooks, and data assets
* ✅ **Reproducibility**: All preprocessing, training, and evaluation steps are automated and reproducible

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

Install all using:

```bash
pip install -r requirements.txt
```

---

## 📝 Notes

* High-cardinality columns (`user_id`, `device_id`, `ip_address`) are excluded during encoding to prevent memory overload.
* SMOTE is applied **only to training data** to avoid data leakage.
* All model and SHAP steps are complete and reproducible in notebooks.

---

## 📈 Results

* **LightGBM** achieved the best recall and AUC for identifying fraudulent transactions.
* **SHAP** analysis confirmed that model decisions align with real-world fraud signals.
* Project balances performance and interpretability, making it suitable for regulated environments.

---


