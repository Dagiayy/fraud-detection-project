# src/decision/feedback.py
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

class AnalystFeedbackStore:
    """Store recording human analyst review feedback for active learning retraining."""

    def __init__(self):
        self.feedback_records: List[Dict[str, Any]] = []

    def record_feedback(
        self,
        transaction_id: str,
        user_id: int,
        analyst_decision: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record analyst verdict: CONFIRMED_FRAUD, FALSE_POSITIVE, or WHITELISTED.
        """
        valid_decisions = {"CONFIRMED_FRAUD", "FALSE_POSITIVE", "WHITELISTED"}
        if analyst_decision not in valid_decisions:
            analyst_decision = "CONFIRMED_FRAUD"

        record = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "analyst_decision": analyst_decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes or ""
        }
        self.feedback_records.append(record)
        return record

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        return self.feedback_records

feedback_store = AnalystFeedbackStore()
