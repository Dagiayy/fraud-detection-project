# src/models/train_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt


def load_data(filepath):
    """Load preprocessed dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def prepare_data(df, target_col='class', test_size=0.2, random_state=42):
    """Split features and target, then train-test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred_probs, y_pred_labels):
    """Evaluate model with AUC-PR, F1, Confusion Matrix and print results."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred_labels)
    cm = confusion_matrix(y_true, y_pred_labels)

    print("\n=== Evaluation Metrics ===")
    print(f"AUC-PR Score: {auc_pr:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_labels))

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return auc_pr, f1, cm


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    """Train LightGBM model."""
    model = LGBMClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    # 1. Load preprocessed data
    data_path = "data/processed/processed_fraud_data.csv"
    df = load_data(data_path)
    print(f"✅ Data loaded: {df.shape}")

    # 2. Prepare data: split into features and target
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"✅ Data split - Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 3. Train Logistic Regression
    print("\n🚀 Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    y_probs_lr = lr_model.predict_proba(X_test)[:, 1]
    y_preds_lr = lr_model.predict(X_test)

    print("\n📊 Logistic Regression Evaluation:")
    evaluate_model(y_test, y_probs_lr, y_preds_lr)

    # 4. Train LightGBM
    print("\n🚀 Training LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train)
    y_probs_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
    y_preds_lgbm = lgbm_model.predict(X_test)

    print("\n📊 LightGBM Evaluation:")
    evaluate_model(y_test, y_probs_lgbm, y_preds_lgbm)

    # 5. Compare models
    print("\n✅ Model training and evaluation completed.")
    print("🔍 Compare AUC-PR and F1-score to choose the best model.")
