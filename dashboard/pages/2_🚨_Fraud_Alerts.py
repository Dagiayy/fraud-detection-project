# dashboard/pages/2_🚨_Fraud_Alerts.py
import streamlit as st
import pandas as pd
from dashboard.app import get_cached_raw_data
from src.models.predict import FraudPredictor
from src.decision.risk_engine import FraudDecisionEngine

st.set_page_config(page_title="Fraud Alerts - Fraud Detection", layout="wide")
st.title("🚨 Live Fraud Alert Control Stream")

df_raw = get_cached_raw_data().head(100)  # Preview stream

predictor = FraudPredictor()
decision_engine = FraudDecisionEngine()

# Score sample stream
probs = predictor.predict_proba(df_raw)
df_proc = predictor.preprocessor.clean_and_transform(df_raw, is_training=False)

records = []
for idx, row_raw in df_raw.iterrows():
    p = float(probs[idx])
    row_feat = df_proc.iloc[idx] if idx < len(df_proc) else pd.Series()
    res = decision_engine.evaluate_transaction(row_feat, p)
    
    records.append({
        "User ID": row_raw.get("user_id", idx),
        "Purchase Value ($)": row_raw.get("purchase_value", 0.0),
        "Source": row_raw.get("source", "Unknown"),
        "Browser": row_raw.get("browser", "Unknown"),
        "Fraud Probability": res["fraud_probability"],
        "Risk Band": res["risk_band"],
        "Decision": res["decision"],
        "Reason Codes": ", ".join(res["reason_codes"])
    })

df_alerts = pd.DataFrame(records)

# Filter Controls
st.sidebar.header("Filter Alerts")
selected_decision = st.sidebar.multiselect("Filter by Decision", options=["BLOCK", "REVIEW", "ALLOW"], default=["BLOCK", "REVIEW"])

df_filtered = df_alerts[df_alerts["Decision"].isin(selected_decision)]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Blocked (Hard Rules / High Risk)", len(df_alerts[df_alerts["Decision"] == "BLOCK"]))
with col2:
    st.metric("Total Flagged for Analyst Review", len(df_alerts[df_alerts["Decision"] == "REVIEW"]))
with col3:
    st.metric("Total Allowed Transactions", len(df_alerts[df_alerts["Decision"] == "ALLOW"]))

st.subheader("Filtered Alert Transactions Stream")
st.dataframe(
    df_filtered.style.applymap(
        lambda v: "background-color: #FFCCCC; color: #990000;" if v == "BLOCK" else ("background-color: #FFE6CC; color: #CC6600;" if v == "REVIEW" else ""),
        subset=["Decision"]
    ),
    use_container_width=True
)
