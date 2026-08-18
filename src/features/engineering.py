# src/features/engineering.py
import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features: hour_of_day, day_of_week, time_since_signup."""
    out = df.copy()
    if "purchase_time" in out.columns and "signup_time" in out.columns:
        p_time = pd.to_datetime(out["purchase_time"])
        s_time = pd.to_datetime(out["signup_time"])
        out["hour_of_day"] = p_time.dt.hour
        out["day_of_week"] = p_time.dt.dayofweek
        out["time_since_signup"] = (p_time - s_time).dt.total_seconds() / 3600.0
        out["time_since_signup"] = out["time_since_signup"].clip(lower=0.0)
    else:
        if "hour_of_day" not in out.columns:
            out["hour_of_day"] = 12
        if "day_of_week" not in out.columns:
            out["day_of_week"] = 2
        if "time_since_signup" not in out.columns:
            out["time_since_signup"] = 24.0
    return out

def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate spending_speed = purchase_value / max(time_since_signup, 0.001)."""
    out = df.copy()
    if "purchase_value" in out.columns and "time_since_signup" in out.columns:
        safe_time = out["time_since_signup"].replace(0, np.nan).fillna(0.01)
        safe_time = np.maximum(safe_time, 0.001)
        out["spending_speed"] = out["purchase_value"] / safe_time
        out["spending_speed"] = out["spending_speed"].fillna(0.0)
    elif "spending_speed" not in out.columns:
        out["spending_speed"] = 0.0
    return out

def add_target_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary indicator signals explicitly required by business target specifications."""
    out = df.copy()
    # is_new_user: Signup within 24 hours of purchase
    if "time_since_signup" in out.columns:
        out["is_new_user"] = (out["time_since_signup"] <= 24.0).astype(int)
    else:
        out["is_new_user"] = 0
        
    # is_rapid_spender: Speed exceeds 95th percentile or threshold 10.0
    if "spending_speed" in out.columns:
        p95 = out["spending_speed"].quantile(0.95) if len(out) > 10 else 10.0
        out["is_rapid_spender"] = (out["spending_speed"] > p95).astype(int)
    else:
        out["is_rapid_spender"] = 0
        
    # is_high_value: Purchase over $100 threshold
    if "purchase_value" in out.columns:
        out["is_high_value"] = (out["purchase_value"] > 100.0).astype(int)
    else:
        out["is_high_value"] = 0

    return out

def add_transaction_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction_frequency per user."""
    out = df.copy()
    if "user_id" in out.columns and "purchase_time" in out.columns:
        out["transaction_frequency"] = out.groupby("user_id")["purchase_time"].transform("count")
    elif "transaction_frequency" not in out.columns:
        out["transaction_frequency"] = 1
    return out

def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master feature extraction pipeline."""
    df = add_time_features(df)
    df = calculate_velocity(df)
    df = add_target_binary_features(df)
    df = add_transaction_frequency(df)
    return df
