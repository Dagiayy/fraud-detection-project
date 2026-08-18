# src/monitoring/drift.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between reference baseline and online actual data.
    PSI < 0.1: No significant drift.
    0.1 <= PSI < 0.2: Moderate drift.
    PSI >= 0.2: Significant drift detected.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    quantiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, quantiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf
    
    expected_counts, _ = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=buckets)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Avoid zero division
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_value, 4))

class DataDriftDetector:
    """Drift detector evaluating feature distributions against baseline reference."""

    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df

    def detect_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        drift_flagged_count = 0
        
        num_cols = self.reference_df.select_dtypes(include=np.number).columns.tolist()
        for col in num_cols:
            if col in current_df.columns:
                ref_vals = self.reference_df[col].dropna().values
                curr_vals = current_df[col].dropna().values
                
                psi = calculate_psi(ref_vals, curr_vals)
                drift_status = "STABLE"
                if psi >= 0.20:
                    drift_status = "SIGNIFICANT_DRIFT"
                    drift_flagged_count += 1
                elif psi >= 0.10:
                    drift_status = "MODERATE_DRIFT"

                results[col] = {
                    "psi_score": psi,
                    "status": drift_status
                }
                
        return {
            "feature_drift_reports": results,
            "total_features_monitored": len(results),
            "features_with_significant_drift": drift_flagged_count,
            "overall_drift_status": "ALERT" if drift_flagged_count > 0 else "STABLE"
        }
