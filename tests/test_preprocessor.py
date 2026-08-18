# tests/test_preprocessor.py
from __future__ import annotations
import os
import pandas as pd

from src.preprocessor import FraudPreprocessor, Paths


def test_pipeline_smoke(tmp_path):
    # Put tiny CSVs in tmp_path for a real test, or mock in CI.
    fraud = tmp_path / "fraud.csv"
    ip = tmp_path / "ip.csv"

    pd.DataFrame(
        {
            "user_id": [1, 1],
            "signup_time": ["2020-01-01 00:00:00", "2020-01-02 00:00:00"],
            "purchase_time": ["2020-01-01 01:00:00", "2020-01-02 02:00:00"],
            "purchase_value": [100, 200],
            "age": [30, 31],
            "ip_address": [10, 20],
            "source": ["Ads", "Direct"],
            "browser": ["Chrome", "Firefox"],
            "sex": ["M", "F"],
            "class": [0, 1],
        }
    ).to_csv(fraud, index=False)

    pd.DataFrame(
        {
            "lower_bound_ip_address": [0, 15],
            "upper_bound_ip_address": [14, 30],
            "country": ["A", "B"],
        }
    ).to_csv(ip, index=False)

    out = tmp_path / "out.csv"
    pp = FraudPreprocessor(Paths(str(fraud), str(ip)))
    pp.load_data()
    df = pp.preprocess(balance_method="none")
    assert not df.empty
    pp.save_processed_data(str(out))
    assert os.path.exists(out)
