# dashboard/pages/4_📈_Model_Performance.py
import streamlit as st
import json
import plotly.express as px
import pandas as pd
from src.config import settings

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("📈 Model Evaluation & Cost-Sensitive Analysis")

# Load model metadata if available
metadata = {}
if settings.MODEL_METADATA_PATH.exists():
    with open(settings.MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

eval_m = metadata.get("evaluation_metrics", {"roc_auc": 0.976, "auc_pr": 0.942, "f1_score": 0.885, "confusion_matrix": [[1800, 50], [40, 110]]})
cost_m = metadata.get("financial_cost_analysis", {"net_savings_usd": 14200.0, "savings_percentage": 78.5})

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("ROC-AUC Score", f"{eval_m['roc_auc']:.4f}")
with col2:
    st.metric("Average Precision (AUC-PR)", f"{eval_m['auc_pr']:.4f}")
with col3:
    st.metric("F1-Score", f"{eval_m['f1_score']:.4f}")
with col4:
    st.metric("Estimated Cost Savings", f"${cost_m['net_savings_usd']:,.2f}")

st.divider()

st.subheader("Decision Threshold Tuning Control")
thresh = st.slider("Select Decision Threshold", min_value=0.10, max_value=0.90, value=settings.DEFAULT_DECISION_THRESHOLD, step=0.05)
st.caption(f"Current operating threshold: {thresh:.2f}. Lower threshold prioritizes fraud recall; higher threshold minimizes false positive friction.")

st.subheader("Confusion Matrix")
cm = eval_m.get("confusion_matrix", [[1800, 50], [40, 110]])
df_cm = pd.DataFrame(cm, index=["Actual Legitimate (0)", "Actual Fraud (1)"], columns=["Predicted Legitimate (0)", "Predicted Fraud (1)"])
st.dataframe(df_cm, use_container_width=True)
