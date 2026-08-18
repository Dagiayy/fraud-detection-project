# src/decision/risk_engine.py
import pandas as pd
from typing import Dict, Any, List
from src.config import settings
from src.decision.rules import evaluate_business_rules

class FraudDecisionEngine:
    """
    Production Risk Decision Engine.
    Combines machine learning fraud probabilities with business policy rules
    to output deterministic decisions (ALLOW, REVIEW, BLOCK) and risk bands.
    """

    def __init__(self, decision_threshold: float = settings.DEFAULT_DECISION_THRESHOLD):
        self.threshold = decision_threshold

    def get_risk_band(self, probability: float) -> str:
        for band, (low, high) in settings.RISK_BANDS.items():
            if low <= probability <= high:
                return band
        return "CRITICAL" if probability >= 0.8 else "LOW"

    def evaluate_transaction(self, row: pd.Series, probability: float) -> Dict[str, Any]:
        rule_reasons, hard_block = evaluate_business_rules(row)
        risk_band = self.get_risk_band(probability)

        # Decision Matrix Logic
        if hard_block or probability >= 0.70:
            decision = "BLOCK"
        elif probability >= self.threshold or risk_band in ["MEDIUM", "HIGH"]:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        all_reasons = rule_reasons.copy()
        if probability >= self.threshold:
            all_reasons.append(f"MODEL_HIGH_FRAUD_PROBABILITY_({probability:.3f})")

        return {
            "decision": decision,
            "risk_band": risk_band,
            "fraud_probability": round(float(probability), 4),
            "threshold_used": self.threshold,
            "reason_codes": all_reasons,
            "hard_rule_triggered": hard_block
        }
