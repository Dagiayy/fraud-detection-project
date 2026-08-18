# tests/unit/test_drift.py
import numpy as np
import pandas as pd
from src.monitoring.drift import calculate_psi, DataDriftDetector

def test_calculate_psi_stable():
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(0, 1, 1000)
    psi = calculate_psi(ref, curr)
    assert psi < 0.10  # Stable

def test_calculate_psi_drifted():
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(5, 1, 1000)  # Major shift
    psi = calculate_psi(ref, curr)
    assert psi >= 0.20  # Significant drift

def test_drift_detector():
    df_ref = pd.DataFrame({"spending_speed": np.random.normal(2, 0.5, 500)})
    df_curr = pd.DataFrame({"spending_speed": np.random.normal(2, 0.5, 500)})
    
    detector = DataDriftDetector(df_ref)
    report = detector.detect_drift(df_curr)
    
    assert report["overall_drift_status"] == "STABLE"
    assert "spending_speed" in report["feature_drift_reports"]
