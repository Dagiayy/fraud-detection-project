# src/data/synthetic_generator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_transactions(num_records: int = 2000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic transactions for offline testing and pipeline validation.
    Includes normal behavior, rapid spenders, new users, and high-value anomalies.
    """
    np.random.seed(random_seed)
    
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    
    user_ids = np.random.randint(100000, 200000, size=num_records)
    ages = np.random.randint(18, 70, size=num_records)
    sources = np.random.choice(["Ads", "SEO", "Direct"], size=num_records, p=[0.4, 0.4, 0.2])
    browsers = np.random.choice(["Chrome", "FireFox", "IE", "Safari", "Opera"], size=num_records, p=[0.45, 0.25, 0.15, 0.10, 0.05])
    genders = np.random.choice(["M", "F"], size=num_records, p=[0.52, 0.48])
    
    # Generate realistic IP bounds (fit in standard 32-bit integer range)
    ip_addresses = np.random.randint(1000000, 2147000000, size=num_records, dtype=np.int64)
    
    # Signup vs Purchase times
    signup_offsets = np.random.exponential(scale=100, size=num_records)  # hours before base
    purchase_offsets = signup_offsets - np.random.exponential(scale=50, size=num_records)
    # Ensure purchase is after signup
    purchase_offsets = np.maximum(purchase_offsets, 0)
    
    signup_times = [base_time + timedelta(hours=float(so)) for so in signup_offsets]
    purchase_times = [st + timedelta(hours=float(po)) for st, po in zip(signup_times, purchase_offsets)]
    
    # Purchase values (lognormal)
    purchase_values = np.round(np.random.lognormal(mean=3.5, sigma=0.8, size=num_records), 2)
    purchase_values = np.clip(purchase_values, 9.0, 350.0)
    
    # Ground truth fraud simulation (based on domain rules: rapid spending + new account)
    time_since_signup_hours = [(pt - st).total_seconds() / 3600.0 for st, pt in zip(signup_times, purchase_times)]
    spending_speeds = [pv / max(tss, 0.01) for pv, tss in zip(purchase_values, time_since_signup_hours)]
    
    fraud_probs = []
    for tss, speed, pv in zip(time_since_signup_hours, spending_speeds, purchase_values):
        prob = 0.05
        if tss < 24:
            prob += 0.35
        if speed > 10.0:
            prob += 0.40
        if pv > 150:
            prob += 0.15
        fraud_probs.append(min(prob, 0.95))
        
    labels = (np.random.rand(num_records) < np.array(fraud_probs)).astype(int)
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "signup_time": [st.strftime("%Y-%m-%d %H:%M:%S") for st in signup_times],
        "purchase_time": [pt.strftime("%Y-%m-%d %H:%M:%S") for pt in purchase_times],
        "purchase_value": purchase_values,
        "age": ages,
        "ip_address": ip_addresses,
        "source": sources,
        "browser": browsers,
        "sex": genders,
        "class": labels
    })
    
    return df
