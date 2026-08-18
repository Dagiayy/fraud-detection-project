# dashboard/pages/3_🔍_Transaction_Investigation.py
import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.app import get_cached_raw_data
from src.models.predict import FraudPredictor
from src.decision.risk_engine import FraudDecisionEngine
from src.explainability.shap_explainer import FraudSHAPExplainer

st.set_page_config(page_title="Transaction Investigation", layout="wide")
st.title("🔍 Interactive Transaction Inspector & Live CSV Scoring")

predictor = FraudPredictor()
decision_engine = FraudDecisionEngine()
explainer = FraudSHAPExplainer()

# Section 1: Live CSV File Uploader
st.subheader("📤 Real-Time Batch / Transaction CSV Scoring")
uploaded_file = st.file_uploader("Upload raw transaction CSV to score", type=["csv"])

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    st.success(f"Uploaded {len(df_upload)} transactions. Scoring...")
    
    probs = predictor.predict_proba(df_upload)
    df_proc = predictor.preprocessor.clean_and_transform(df_upload, is_training=False)
    
    results = []
    for idx, r_raw in df_upload.iterrows():
        p = float(probs[idx])
        r_feat = df_proc.iloc[idx] if idx < len(df_proc) else pd.Series()
        res = decision_engine.evaluate_transaction(r_feat, p)
        results.append({
            "User ID": r_raw.get("user_id", idx),
            "Purchase Value ($)": r_raw.get("purchase_value", 0.0),
            "Probability": res["fraud_probability"],
            "Risk Band": res["risk_band"],
            "Decision": res["decision"],
            "Reason Codes": ", ".join(res["reason_codes"])
        })
    st.dataframe(pd.DataFrame(results), use_container_width=True)

st.divider()

# Section 2: Single Transaction Inspector
st.subheader("🕵️ Inspector & SHAP Feature Explanations")
df_raw = get_cached_raw_data().head(50)

selected_idx = st.selectbox("Select Transaction Row to Inspect", options=df_raw.index, format_func=lambda i: f"Row {i} - User {df_raw.loc[i, 'user_id']} - ${df_raw.loc[i, 'purchase_value']}")

row_raw = df_raw.loc[[selected_idx]]
prob = float(predictor.predict_proba(row_raw)[0])
df_proc = predictor.preprocessor.clean_and_transform(row_raw, is_training=False)
decision_out = decision_engine.evaluate_transaction(df_proc.iloc[0], prob)
shap_out = explainer.explain_transaction(df_proc)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Fraud Risk Score", f"{decision_out['fraud_probability']:.4f}")
with col2:
    st.metric("Risk Band", decision_out["risk_band"])
with col3:
    st.metric("Automated Decision", decision_out["decision"])

st.write("**Triggered Policy Reason Codes:**", decision_out["reason_codes"] if decision_out["reason_codes"] else "None")

# SHAP Bar Chart
st.subheader("SHAP Risk Contribution Breakdown")
contribs = shap_out["all_contributions"]
df_shap = pd.DataFrame(list(contribs.items()), columns=["Feature", "SHAP Impact"]).head(10)
fig_shap = px.bar(df_shap, x="SHAP Impact", y="Feature", orientation="h", color="SHAP Impact", color_continuous_scale="RdYlGn_r", title="Top Feature Impacts on Fraud Risk Score")
st.plotly_chart(fig_shap, use_container_width=True)
