# dashboard/utils/prediction.py
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FraudPredictor:
    """
    Real-time and batch fraud prediction class.

    Responsibilities:
      - Load trained model
      - Accept a new transaction (dict) or DataFrame
      - Preprocess input similarly to training
      - Predict fraud probabilities
    """

    def __init__(self, model_path: str, scaler: Optional[StandardScaler] = None) -> None:
        self.model_path = model_path
        self.model = self._load_model()
        self.scaler = scaler  # optional, in case numeric scaling needed

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            logger.info("Model loaded successfully from %s", self.model_path)
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def predict(
        self, 
        data: Union[Dict[str, Any], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Predict fraud for a single transaction dict or batch DataFrame.

        Returns DataFrame with:
            - fraud_probability
            - predicted_class (0/1)
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        df_proc = self._preprocess_input(df)

        # Predict probabilities
        proba = self.model.predict_proba(df_proc)[:, 1]
        df_result = df.copy()
        df_result["fraud_probability"] = proba
        df_result["predicted_class"] = (proba >= 0.5).astype(int)

        return df_result

    def _preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply necessary transformations:
          - Datetime conversion
          - Feature engineering (time_since_signup, hour_of_day, day_of_week)
          - Velocity & transaction frequency
          - Categorical encoding
          - Scaling numeric features
        """
        df_proc = df.copy()

        # --- Datetime features ---
        for col in ["signup_time", "purchase_time"]:
            if col in df_proc.columns:
                df_proc[col] = pd.to_datetime(df_proc[col], errors="coerce")

        df_proc["hour_of_day"] = df_proc["purchase_time"].dt.hour
        df_proc["day_of_week"] = df_proc["purchase_time"].dt.dayofweek
        df_proc["time_since_signup"] = (
            df_proc["purchase_time"] - df_proc["signup_time"]
        ).dt.total_seconds() / 3600.0
        df_proc["time_since_signup"].replace(0, np.nan, inplace=True)

        # --- Velocity ---
        if "purchase_value" in df_proc.columns:
            df_proc["spending_speed"] = df_proc["purchase_value"] / df_proc["time_since_signup"]
            df_proc["spending_speed"].fillna(0, inplace=True)

        # --- Transaction frequency ---
        if "user_id" in df_proc.columns:
            df_proc["transaction_frequency"] = df_proc.groupby("user_id")["purchase_time"].transform("count")
        else:
            df_proc["transaction_frequency"] = 1  # default for single/batch unknown users

        # --- Encode categorical columns ---
        cat_cols = [c for c in ["source", "browser", "sex", "country"] if c in df_proc.columns]
        if cat_cols:
            df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=False)

        # --- Scale numeric columns ---
        if self.scaler is not None:
            num_cols = df_proc.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                df_proc[num_cols] = self.scaler.transform(df_proc[num_cols])

        # Drop unused columns
        drop_cols = ["signup_time", "purchase_time", "ip_address", "user_id"]
        df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns], inplace=True, errors="ignore")

        return df_proc
