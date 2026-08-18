# tests/unit/test_preprocessor.py
import pandas as pd
from src.preprocessing.preprocessor import FraudPreprocessor

def test_preprocessor_pipeline_smoke():
    df_raw = pd.DataFrame({
        "user_id": [1, 2],
        "signup_time": ["2025-01-01 00:00:00", "2025-01-01 00:00:00"],
        "purchase_time": ["2025-01-01 01:00:00", "2025-01-02 02:00:00"],
        "purchase_value": [50.0, 150.0],
        "age": [30, 40],
        "ip_address": [1000, 2000],
        "source": ["Ads", "Direct"],
        "browser": ["Chrome", "FireFox"],
        "sex": ["M", "F"],
        "class": [0, 1]
    })
    
    preprocessor = FraudPreprocessor()
    df_proc = preprocessor.clean_and_transform(df_raw, is_training=True)
    
    assert not df_proc.empty
    assert "spending_speed" in df_proc.columns
    assert "is_new_user" in df_proc.columns
    assert "user_id" not in df_proc.columns

def test_preprocessor_empty_df():
    df_empty = pd.DataFrame(columns=['user_id', 'signup_time', 'purchase_time', 'purchase_value', 'age', 'ip_address', 'source', 'browser', 'sex'])
    p = FraudPreprocessor()
    df_out = p.clean_and_transform(df_empty, is_training=False)
    assert df_out.empty
