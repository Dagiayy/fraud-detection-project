# dashboard/pages/6_⚙️_System_Health.py
import streamlit as st
from datetime import datetime
from src.config import settings
from src.monitoring.metrics import telemetry

st.set_page_config(page_title="System Health", layout="wide")
st.title("⚙️ Service Telemetry & Production Health")

summary = telemetry.get_summary()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Scored Requests", summary["total_requests"])
with col2:
    st.metric("Flagged Fraud Rate", f"{summary['fraud_flag_rate_pct']}%")
with col3:
    st.metric("Model Status", "LOADED" if settings.LIGHTGBM_MODEL_PATH.exists() or settings.LOGISTIC_REGRESSION_MODEL_PATH.exists() else "HEURISTIC")

st.subheader("Model Artifact Metadata & Paths")
st.json({
    "project_root": str(settings.PROJECT_ROOT),
    "lightgbm_model_path": str(settings.LIGHTGBM_MODEL_PATH),
    "metadata_path": str(settings.MODEL_METADATA_PATH),
    "default_decision_threshold": settings.DEFAULT_DECISION_THRESHOLD,
    "risk_bands": settings.RISK_BANDS,
    "cost_parameters": {
        "false_positive_cost": settings.COST_FALSE_POSITIVE,
        "false_negative_cost": settings.COST_FALSE_NEGATIVE,
        "review_cost": settings.COST_REVIEW
    }
})
