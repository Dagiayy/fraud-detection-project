# tests/unit/test_decision_engine.py
import pandas as pd
from src.decision.risk_engine import FraudDecisionEngine

def test_decision_engine_allow():
    engine = FraudDecisionEngine(decision_threshold=0.35)
    row = pd.Series({"spending_speed": 0.5, "time_since_signup": 100.0, "purchase_value": 20.0})
    result = engine.evaluate_transaction(row, probability=0.05)
    assert result["decision"] == "ALLOW"
    assert result["risk_band"] == "LOW"

def test_decision_engine_review():
    engine = FraudDecisionEngine(decision_threshold=0.35)
    row = pd.Series({"spending_speed": 4.0, "time_since_signup": 10.0, "purchase_value": 150.0})
    result = engine.evaluate_transaction(row, probability=0.45)
    assert result["decision"] == "REVIEW"

def test_decision_engine_block():
    engine = FraudDecisionEngine(decision_threshold=0.35)
    row = pd.Series({"spending_speed": 25.0, "time_since_signup": 0.05, "purchase_value": 300.0})
    result = engine.evaluate_transaction(row, probability=0.85)
    assert result["decision"] == "BLOCK"
    assert result["risk_band"] == "CRITICAL"

def test_risk_band_critical():
    engine = FraudDecisionEngine()
    assert engine.get_risk_band(0.90) == 'CRITICAL'
