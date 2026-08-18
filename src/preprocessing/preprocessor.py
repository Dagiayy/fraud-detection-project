# src/preprocessing/preprocessor.py
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
from src.config import settings

class FraudPreprocessor:
    """
    Point-in-time leakage-proof preprocessor.
    Applies cleaning, IP-to-country mapping, categorical encoding, and feature scaling.
    CRITICAL: Does NOT perform SMOTE or resample features to guarantee zero data leakage.
    """

    def __init__(self, df_ip: Optional[pd.DataFrame] = None):
        self.df_ip = df_ip
        self.scaler = StandardScaler()
        self.fitted_columns: Optional[List[str]] = None
        self.bounds_list: List[Tuple[int, int, str]] = []
        if df_ip is not None and not df_ip.empty:
            self._prepare_ip_bounds(df_ip)

    def _prepare_ip_bounds(self, df_ip: pd.DataFrame):
        lows = df_ip["lower_bound_ip_address"].astype(int).values
        highs = df_ip["upper_bound_ip_address"].astype(int).values
        countries = df_ip["country"].values
        self.bounds_list = sorted(list(zip(lows, highs, countries)), key=lambda x: x[0])

    def lookup_country(self, ip: int) -> str:
        if not self.bounds_list:
            return "Unknown"
        left, right = 0, len(self.bounds_list) - 1
        while left <= right:
            mid = (left + right) // 2
            low, high, country = self.bounds_list[mid]
            if low <= ip <= high:
                return str(country)
            elif ip < low:
                right = mid - 1
            else:
                left = mid + 1
        return "Unknown"

    def clean_and_transform(self, df_raw: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df = df_raw.copy()
        
        # 1. Handle datetimes
        for col in ["signup_time", "purchase_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # 2. IP Geolocation Lookup
        if "ip_address" in df.columns:
            df["ip_int"] = df["ip_address"].fillna(0).astype(int)
            df["country"] = df["ip_int"].apply(self.lookup_country)
            df.drop(columns=["ip_int", "ip_address"], inplace=True, errors="ignore")
        elif "country" not in df.columns:
            df["country"] = "Unknown"

        # 3. Clean duplicates & drop raw identifiers
        if is_training:
            df.drop_duplicates(inplace=True)
        for col in ["user_id", "device_id", "transaction_id"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # 4. Feature Engineering Integration
        from src.features.engineering import extract_all_features
        df = extract_all_features(df)

        # 5. One-Hot Encoding
        categorical_cols = [c for c in settings.CATEGORICAL_FEATURES if c in df.columns]
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

        # Drop any remaining raw datetime columns
        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        if datetime_cols:
            df.drop(columns=datetime_cols, inplace=True)

        # Align columns during transform
        if is_training:
            self.fitted_columns = [c for c in df.columns if c != settings.TARGET_COLUMN]
        else:
            if self.fitted_columns:
                target_col = settings.TARGET_COLUMN
                has_target = target_col in df.columns
                y = df[target_col] if has_target else None
                
                # Reindex features to match fitted columns
                for col in self.fitted_columns:
                    if col not in df.columns:
                        df[col] = 0
                df = df[self.fitted_columns]
                if has_target and y is not None:
                    df[target_col] = y

        # 6. Scaling Numerical Features
        num_cols = [c for c in settings.NUMERICAL_FEATURES if c in df.columns]
        if num_cols:
            if is_training or not hasattr(self.scaler, "mean_"):
                df[num_cols] = self.scaler.fit_transform(df[num_cols])
            else:
                df[num_cols] = self.scaler.transform(df[num_cols])

        return df
