# src/models/train.py
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from src.config import settings
from src.data.ingestion import load_raw_data
from src.preprocessing.preprocessor import FraudPreprocessor
from src.models.evaluate import evaluate_models, cost_sensitive_evaluation

def balance_training_set(X_train: pd.DataFrame, y_train: np.ndarray, random_seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """Applies oversampling strictly on training data."""
    if HAS_SMOTE:
        smote = SMOTE(random_state=random_seed)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    else:
        # Fallback random oversampling
        np.random.seed(random_seed)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return X_train, y_train
        n_samples = max(len(pos_idx), len(neg_idx))
        pos_res = np.random.choice(pos_idx, size=n_samples, replace=True)
        neg_res = np.random.choice(neg_idx, size=n_samples, replace=True)
        res_idx = np.concatenate([pos_res, neg_res])
        np.random.shuffle(res_idx)
        return X_train.iloc[res_idx].reset_index(drop=True), y_train[res_idx]

def train_pipeline() -> Dict[str, Any]:
    """
    Executes the leakage-proof model training pipeline:
    1. Load raw dataset & preprocess features.
    2. Split into train & test splits BEFORE resampling.
    3. Apply oversampling strictly on X_train.
    4. Train LightGBM / HistGradientBoosting & Logistic Regression models.
    5. Evaluate strictly on untouched test split.
    6. Save models and metadata JSON.
    """
    # 1. Load Data
    df_raw, df_ip = load_raw_data()
    
    # 2. Preprocess (No SMOTE here)
    preprocessor = FraudPreprocessor(df_ip=df_ip)
    df_processed = preprocessor.clean_and_transform(df_raw, is_training=True)
    
    target_col = settings.TARGET_COLUMN
    if target_col not in df_processed.columns:
        raise ValueError(f"Target column '{target_col}' missing from preprocessed data.")
        
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col].values
    
    # 3. Leakage-Proof Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=settings.TEST_SIZE, 
        random_state=settings.RANDOM_SEED, 
        stratify=y
    )
    
    # 4. Apply Oversampling ONLY on X_train
    X_train_res, y_train_res = balance_training_set(X_train, y_train, random_seed=settings.RANDOM_SEED)
    
    # 5. Train GBDT Model
    if HAS_LIGHTGBM:
        gbdt_model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        )
        gbdt_model.fit(X_train_res, y_train_res)
        model_type_str = "LightGBM Classifier"
    else:
        gbdt_model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            random_state=settings.RANDOM_SEED
        )
        gbdt_model.fit(X_train_res, y_train_res)
        model_type_str = "HistGradientBoosting Classifier"
    
    # 6. Train Logistic Regression Model
    lr_model = LogisticRegression(max_iter=1000, random_state=settings.RANDOM_SEED)
    lr_model.fit(X_train_res, y_train_res)
    
    # 7. Evaluate on UNTOUCHED Test Split
    y_probs_gbdt = gbdt_model.predict_proba(X_test)[:, 1]
    eval_metrics = evaluate_models(y_test, y_probs_gbdt, threshold=settings.DEFAULT_DECISION_THRESHOLD)
    cost_metrics = cost_sensitive_evaluation(y_test, y_probs_gbdt, threshold=settings.DEFAULT_DECISION_THRESHOLD)
    
    # 8. Save Artifacts
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if HAS_LIGHTGBM and hasattr(gbdt_model, "booster_"):
        gbdt_model.booster_.save_model(str(settings.LIGHTGBM_MODEL_PATH))
    else:
        joblib.dump(gbdt_model, str(settings.MODELS_DIR / "gbdt_model.pkl"))

    joblib.dump(lr_model, str(settings.LOGISTIC_REGRESSION_MODEL_PATH))
    
    metadata = {
        "model_type": model_type_str,
        "feature_names": list(X.columns),
        "target_name": target_col,
        "evaluation_metrics": eval_metrics,
        "financial_cost_analysis": cost_metrics,
        "train_samples_raw": len(X_train),
        "train_samples_resampled": len(X_train_res),
        "test_samples_untouched": len(X_test),
        "decision_threshold": settings.DEFAULT_DECISION_THRESHOLD
    }
    
    with open(settings.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[SUCCESS] Training completed successfully! Models saved to {settings.MODELS_DIR}")
    return metadata

if __name__ == "__main__":
    train_pipeline()
