# src/utils/refactored_preprocessor.py
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

@dataclass
class Paths:
    fraud_path: str
    ip_path: str
    credit_path: Optional[str] = None


class FraudPreprocessor:
    """
    Refactored preprocessor for fraud detection data.
    Handles cleaning, feature engineering, geolocation, encoding, scaling, and balancing.
    """

    def __init__(self, paths: Paths):
        self.df_fraud = pd.read_csv(paths.fraud_path)
        self.df_ip = pd.read_csv(paths.ip_path)
        self.df_credit = pd.read_csv(paths.credit_path) if paths.credit_path else None
        self.scaler = StandardScaler()

    # -------------------------
    # Basic Cleaning
    # -------------------------
    def handle_missing_values(self):
        self.df_fraud.fillna(self.df_fraud.median(numeric_only=True), inplace=True)
        self.df_fraud.dropna(inplace=True)
        if self.df_credit is not None:
            self.df_credit.fillna(0, inplace=True)

    def convert_datetimes(self):
        for col in ["signup_time", "purchase_time"]:
            if col in self.df_fraud.columns:
                self.df_fraud[col] = pd.to_datetime(self.df_fraud[col])

    def clean_data(self):
        self.df_fraud.drop_duplicates(inplace=True)
        for col in ["user_id", "device_id"]:
            if col in self.df_fraud.columns:
                self.df_fraud.drop(columns=col, inplace=True)
        if self.df_credit is not None:
            self.df_credit.drop_duplicates(inplace=True)

    # -------------------------
    # Feature Engineering
    # -------------------------
    def add_time_features(self):
        df = self.df_fraud
        df["hour_of_day"] = df["purchase_time"].dt.hour
        df["day_of_week"] = df["purchase_time"].dt.dayofweek
        df["time_since_signup"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600
        self.df_fraud = df

    def calculate_velocity(self):
        df = self.df_fraud
        df["time_since_signup"].replace(0, np.nan, inplace=True)
        df["spending_speed"] = df["purchase_value"] / df["time_since_signup"]
        df["spending_speed"].fillna(0, inplace=True)
        self.df_fraud = df

    def add_transaction_frequency(self):
        df = self.df_fraud
        if "user_id" in df.columns:
            df["transaction_frequency"] = df.groupby("user_id")["purchase_time"].transform("count")
        self.df_fraud = df

    # -------------------------
    # Geolocation
    # -------------------------
    def merge_country(self):
        df = self.df_fraud
        df["ip_int"] = df["ip_address"].apply(lambda x: int(x) if not pd.isna(x) else 0)
        self.df_ip["lower_bound_ip_address"] = self.df_ip["lower_bound_ip_address"].astype(int)
        self.df_ip["upper_bound_ip_address"] = self.df_ip["upper_bound_ip_address"].astype(int)

        bounds = list(
            zip(
                self.df_ip["lower_bound_ip_address"],
                self.df_ip["upper_bound_ip_address"],
                self.df_ip["country"],
            )
        )

        def lookup(ip):
            left, right = 0, len(bounds) - 1
            while left <= right:
                mid = (left + right) // 2
                low, high, country = bounds[mid]
                if low <= ip <= high:
                    return country
                elif ip < low:
                    right = mid - 1
                else:
                    left = mid + 1
            return "Unknown"

        df["country"] = df["ip_int"].apply(lookup)
        df.drop(columns=["ip_int", "ip_address"], inplace=True)
        self.df_fraud = df

    # -------------------------
    # Encoding, Scaling, Balancing
    # -------------------------
    def encode_categorical(self, categorical_columns: Optional[List[str]] = None):
        df = self.df_fraud
        if categorical_columns is None:
            categorical_columns = ["source", "browser", "sex", "country"]
        cols = [c for c in categorical_columns if c in df.columns]
        if cols:
            df = pd.get_dummies(df, columns=cols, drop_first=False)
        self.df_fraud = df

    def scale_features(self, numerical_columns: List[str]):
        df = self.df_fraud
        cols = [c for c in numerical_columns if c in df.columns]
        df[cols] = self.scaler.fit_transform(df[cols])
        self.df_fraud = df

    def balance_classes(self, target_col="class", method="smote"):
        df = self.df_fraud
        X = df.drop(columns=[target_col])
        # Drop datetime columns
        datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        if datetime_cols:
            X = X.drop(columns=datetime_cols)
        y = df[target_col]

        sampler = SMOTE(random_state=42) if method == "smote" else RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        self.df_fraud = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)

    # -------------------------
    # Save
    # -------------------------
    def save_processed_data(self, output_path: str):
        self.df_fraud.to_csv(output_path, index=False)
