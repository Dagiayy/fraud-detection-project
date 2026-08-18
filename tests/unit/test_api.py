# tests/unit/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"

def test_api_predict():
    payload = {
        "user_id": 12345,
        "signup_time": "2025-01-01 00:00:00",
        "purchase_time": "2025-01-01 00:05:00",
        "purchase_value": 120.0,
        "age": 32,
        "ip_address": 1234567,
        "source": "Ads",
        "browser": "Chrome",
        "sex": "M"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "score_results" in data
    assert data["score_results"]["decision"] in ["ALLOW", "REVIEW", "BLOCK"]
