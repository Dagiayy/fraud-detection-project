# src/models/predict.py
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from src.config import settings
from src.preprocessing.preprocessor import FraudPreprocessor
from src.data.ingestion import load_raw_data

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

class FraudPredictor:
    """Inference engine for single and batch transaction scoring."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        # Load IP map for geolocation lookup
        _, df_ip = load_raw_data()
        self.preprocessor = FraudPreprocessor(df_ip=df_ip)
        
        # Load model artifacts
        gbdt_pkl = settings.MODELS_DIR / "gbdt_model.pkl"
        if HAS_LIGHTGBM and settings.LIGHTGBM_MODEL_PATH.exists():
            self.model = lgb.Booster(model_file=str(settings.LIGHTGBM_MODEL_PATH))
        elif gbdt_pkl.exists():
            self.model = joblib.load(gbdt_pkl)
        elif settings.LOGISTIC_REGRESSION_MODEL_PATH.exists():
            self.model = joblib.load(settings.LOGISTIC_REGRESSION_MODEL_PATH)
            
        if settings.MODEL_METADATA_PATH.exists():
            with open(settings.MODEL_METADATA_PATH, "r") as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])

    def predict_proba(self, df_raw: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            # Deterministic heuristic if model not trained yet
            from src.features.engineering import extract_all_features
            df_feat = extract_all_features(df_raw.copy())
            speeds = df_feat.get("spending_speed", pd.Series([0]*len(df_feat))).values
            times = df_feat.get("time_since_signup", pd.Series([24]*len(df_feat))).values
            probs = np.clip(0.05 + (speeds / 20.0) + (1.0 / np.maximum(times, 1.0)), 0.01, 0.95)
            return probs

        df_proc = self.preprocessor.clean_and_transform(df_raw, is_training=False)
        
        # Ensure exact feature alignment
        if self.feature_names:
            for col in self.feature_names:
                if col not in df_proc.columns:
                    df_proc[col] = 0
            df_proc = df_proc[self.feature_names]

        if HAS_LIGHTGBM and isinstance(self.model, lgb.Booster):
            probs = self.model.predict(df_proc)
        else:
            probs = self.model.predict_proba(df_proc)[:, 1]
            
        return probs
