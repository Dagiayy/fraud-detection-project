# tests/unit/test_features.py
import pandas as pd
from src.features.engineering import extract_all_features, add_target_binary_features

def test_feature_engineering_signals():
    df = pd.DataFrame({
        "user_id": [1, 2],
        "signup_time": ["2025-01-01 00:00:00", "2025-01-01 00:00:00"],
        "purchase_time": ["2025-01-01 01:00:00", "2025-01-10 00:00:00"],  # row 1 is 1 hour, row 2 is 216 hours
        "purchase_value": [150.0, 30.0]
    })
    
    df_feat = extract_all_features(df)
    
    assert "time_since_signup" in df_feat.columns
    assert "spending_speed" in df_feat.columns
    assert "is_new_user" in df_feat.columns
    assert "is_high_value" in df_feat.columns
    
    # Row 0: signup 1 hour ago -> is_new_user = 1, purchase $150 -> is_high_value = 1
    assert df_feat.loc[0, "is_new_user"] == 1
    assert df_feat.loc[0, "is_high_value"] == 1
    
    # Row 1: signup 216 hours ago -> is_new_user = 0
    assert df_feat.loc[1, "is_new_user"] == 0

def test_transaction_frequency():
    df = pd.DataFrame({'user_id': [1, 1], 'purchase_time': ['2025-01-01', '2025-01-02']})
    df_out = extract_all_features(df)
    assert df_out['transaction_frequency'].iloc[0] == 2
