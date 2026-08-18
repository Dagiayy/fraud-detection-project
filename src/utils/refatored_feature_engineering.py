# src/feature_engineering.py
from __future__ import annotations

import numpy as np # type: ignore
import pandas as pd # type: ignore


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour_of_day, day_of_week, and time_since_signup (hours).

    Expects 'signup_time' and 'purchase_time' as datetime columns.
    """
    required = ["signup_time", "purchase_time"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required datetime column: '{c}'")
    out = df.copy()
    out["hour_of_day"] = out["purchase_time"].dt.hour
    out["day_of_week"] = out["purchase_time"].dt.dayofweek
    out["time_since_signup"] = (
        out["purchase_time"] - out["signup_time"]
    ).dt.total_seconds() / 3600.0
    return out


def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    spending_speed = purchase_value / time_since_signup (safe division).
    """
    if "purchase_value" not in df.columns or "time_since_signup" not in df.columns:
        raise ValueError("purchase_value and time_since_signup must be present.")
    out = df.copy()
    out["time_since_signup"] = out["time_since_signup"].replace(0, np.nan)
    out["spending_speed"] = out["purchase_value"] / out["time_since_signup"]
    out["spending_speed"] = out["spending_speed"].fillna(0.0)
    return out


def add_transaction_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transaction_frequency per user (count of transactions).
    Requires 'user_id' and 'purchase_time'.
    """
    out = df.copy()
    if "user_id" in out.columns and "purchase_time" in out.columns:
        out["transaction_frequency"] = out.groupby("user_id")["purchase_time"].transform(
            "count"
        )
    else:
        out["transaction_frequency"] = 1
    return out
