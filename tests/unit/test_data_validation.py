# tests/unit/test_data_validation.py
import pandas as pd
from src.data.data_contract import TransactionSchema
from src.data.validation import DataQualityValidator

def test_transaction_schema_valid():
    txn = TransactionSchema(
        user_id=123,
        signup_time="2025-01-01 00:00:00",
        purchase_time="2025-01-01 01:00:00",
        purchase_value=50.0,
        age=30,
        ip_address=1234567,
        source="Ads",
        browser="Chrome",
        sex="M"
    )
    assert txn.user_id == 123
    assert txn.purchase_value == 50.0

def test_data_quality_validator():
    df = pd.DataFrame({
        "user_id": [1, 2],
        "signup_time": ["2025-01-01", "2025-01-02"],
        "purchase_time": ["2025-01-01", "2025-01-02"],
        "purchase_value": [10.0, 20.0],
        "age": [25, 35],
        "ip_address": [100, 200],
        "source": ["Ads", "SEO"],
        "browser": ["Chrome", "IE"],
        "sex": ["M", "F"]
    })
    validator = DataQualityValidator(df)
    report = validator.validate_schema()
    assert report["is_valid"] is True
    assert report["total_records"] == 2

def test_age_boundary_validation():
    df = pd.DataFrame({'user_id': [1], 'signup_time': ['2025-01-01'], 'purchase_time': ['2025-01-01'], 'purchase_value': [10.0], 'age': [15], 'ip_address': [100], 'source': ['Ads'], 'browser': ['Chrome'], 'sex': ['M']})
    v = DataQualityValidator(df)
    rep = v.validate_schema()
    assert rep['invalid_ages'] == 1

def test_negative_purchase_value():
    df = pd.DataFrame({'user_id': [1], 'signup_time': ['2025-01-01'], 'purchase_time': ['2025-01-01'], 'purchase_value': [-50.0], 'age': [30], 'ip_address': [100], 'source': ['Ads'], 'browser': ['Chrome'], 'sex': ['M']})
    v = DataQualityValidator(df)
    rep = v.validate_schema()
    assert rep['invalid_amounts'] == 1
