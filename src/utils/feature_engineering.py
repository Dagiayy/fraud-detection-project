import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features: hour_of_day, day_of_week, time_since_signup."""
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600  # In hours
    return df

def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate spending_speed = purchase_value / time_since_signup (handle zero division)."""
    df['time_since_signup'] = df['time_since_signup'].replace(0, np.nan)  # Avoid division by zero
    df['spending_speed'] = df['purchase_value'] / df['time_since_signup']
    df['spending_speed'] = df['spending_speed'].fillna(0)  # Fix: Non-inplace fill
    return df