# src/monitoring/metrics.py
from typing import Dict, Any

class SystemTelemetry:
    """Telemetry collector for API latency, request throughput, and model health."""

    def __init__(self):
        self.total_requests = 0
        self.flagged_fraud_count = 0
        self.allowed_count = 0

    def record_prediction(self, decision: str):
        self.total_requests += 1
        if decision in ["BLOCK", "REVIEW"]:
            self.flagged_fraud_count += 1
        else:
            self.allowed_count += 1

    def get_summary(self) -> Dict[str, Any]:
        rate = (self.flagged_fraud_count / self.total_requests * 100.0) if self.total_requests > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "flagged_fraud_count": self.flagged_fraud_count,
            "allowed_count": self.allowed_count,
            "fraud_flag_rate_pct": round(rate, 2)
        }

telemetry = SystemTelemetry()
