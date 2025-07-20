
# Fraud Detection Project - Adey Innovations Inc.

## Overview

This project aims to build robust, interpretable machine learning models to detect fraudulent activities in e-commerce and banking transactions. The core objectives include:

* Enhancing fraud detection accuracy.
* Tackling class imbalance effectively.
* Engineering meaningful and insightful features.
* Maintaining a balance between security and user experience.
* Providing a modular, maintainable, and extensible codebase for fraud analytics.

---

## Project Structure

```

fraud-detection-project/
│
├── data/
│   ├── raw/                     # Original datasets (e.g., Fraud\_Data.csv, IpAddress\_to\_Country.csv)
│   └── processed/               # Cleaned, feature-engineered, and preprocessed datasets
│
├── src/
│   ├── utils/
│   │   └── preprocessor.py      # Preprocessing pipeline: cleaning, encoding, scaling, and balancing
│   │
│   ├── models/
│   │   └── train\_models.py      # Model training and evaluation scripts
│   │
│   └── notebooks/
│       └── 01\_eda\_fraud\_data.ipynb   # EDA and feature engineering notebook
│
├── reports/                     # Visualizations, analysis results, and project reports
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation (this file)

````

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd fraud-detection-project
````

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Use

### 1. Data Preprocessing

Run the preprocessing script to clean, encode, scale, and balance the dataset:

```bash
python src/utils/preprocessor.py
```

* **Input:** Raw dataset at `data/raw/Fraud_Data.csv`
* **Output:** Processed dataset saved to `data/processed/processed_fraud_data.csv`
* **Features:** Handles datetime conversion, categorical encoding (with memory-safe handling of high-cardinality columns), scaling, and class balancing using SMOTE and undersampling.

---

### 2. Exploratory Data Analysis (EDA)

Open the EDA notebook for detailed data exploration and feature engineering:

```bash
jupyter notebook src/notebooks/01_eda_fraud_data.ipynb
```

* Visualize data distributions, detect anomalies, and examine correlations.
* Engineer advanced features such as:

  * Time-based features (e.g., time since signup, hour of day)
  * Geolocation mapping from IP addresses
  * Device/browser usage indicators

---

### 3. Model Training and Evaluation

Train and evaluate machine learning models using:

```bash
python src/models/train_models.py
```

* Models: Logistic Regression and LightGBM
* Metrics: AUC-PR, F1-Score, Confusion Matrix, and Classification Reports
* Visualizes Precision-Recall curves automatically.
* **Note:** Model persistence (saving/loading models) functionality will be added soon.

---

## Key Features

* **Class Imbalance Handling:** Combines SMOTE oversampling with random undersampling to balance classes.
* **Feature Engineering:** Incorporates temporal, geolocation, and behavioral features for richer data representation.
* **Modular Codebase:** Organized into reusable, maintainable components.
* **Explainability:** Future integration of SHAP to provide transparent model insights.

---

## Requirements

* Python 3.8+
* `pandas`, `scikit-learn`, `imbalanced-learn`
* `matplotlib`, `seaborn`
* `jupyter`

Refer to `requirements.txt` for full details.

---

## Notes

* High-cardinality columns such as `user_id`, `device_id`, and `ip_address` are dropped during preprocessing to prevent memory issues.
* Apply data balancing only on training data to avoid data leakage.
* Model saving/loading features are under development.
* Contributions and suggestions are welcome to improve this project.

---

## Contact & Contributions

For questions or contributions, please open an issue or submit a pull request.

---

```


