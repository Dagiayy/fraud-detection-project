# tests/fixtures/synthetic_transactions.py
import pytest
import pandas as pd

@pytest.fixture
def sample_transaction_df():
    return pd.DataFrame({
        "user_id": [101, 102],
        "signup_time": ["2025-01-01 00:00:00", "2025-01-01 00:00:00"],
        "purchase_time": ["2025-01-01 00:05:00", "2025-01-05 00:00:00"],  # row 1 is 5 mins after signup
        "purchase_value": [150.0, 45.0],
        "age": [30, 45],
        "ip_address": [1000000, 2000000],
        "source": ["Ads", "Direct"],
        "browser": ["Chrome", "FireFox"],
        "sex": ["M", "F"],
        "class": [1, 0]
    })
