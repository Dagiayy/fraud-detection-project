# dashboard/app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from src.config import settings
from src.data.ingestion import load_raw_data
from src.preprocessing.preprocessor import FraudPreprocessor
from src.features.engineering import extract_all_features

# Streamlit Page Config
st.set_page_config(
    page_title="💳 Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Workflow Mode Selector
st.sidebar.header("⚙️ Execution Workflow Mode")
workflow_choice = st.sidebar.radio(
    "Select Active Workflow:",
    options=["Workflow 1: Standard ML Pipeline", "Workflow 2: Production Enterprise Stack"],
    index=0
)

if workflow_choice == "Workflow 1: Standard ML Pipeline":
    st.info("🔹 **WORKFLOW 1 ACTIVE**: Standard local ML pipeline execution. Running in lightweight mode.")
else:
    st.success("🚀 **WORKFLOW 2 ACTIVE**: Production Enterprise Stack enabled (API Auth + Drift Monitoring + Active Learning).")

# Global Environment Banner
st.warning("⚠️ **DEMO / SYNTHETIC DATA ENVIRONMENT** — Real production data is pending. All metrics represent fixture validation.")

# Data & Model Caching Helper Functions
@st.cache_data
def get_cached_processed_data() -> pd.DataFrame:
    df_raw, df_ip = load_raw_data()
    preprocessor = FraudPreprocessor(df_ip=df_ip)
    df_proc = preprocessor.clean_and_transform(df_raw, is_training=True)
    
    country_cols = [c for c in df_proc.columns if c.startswith("country_")]
    if country_cols:
        df_proc["country"] = df_proc[country_cols].idxmax(axis=1).str.replace("country_", "")
    else:
        df_proc["country"] = "Unknown"
        
    return df_proc

@st.cache_data
def get_cached_raw_data() -> pd.DataFrame:
    df_raw, _ = load_raw_data()
    return df_raw

# Main Page Layout
st.title("🛡️ Enterprise Fraud Detection Control Center")
st.markdown(f"""
Currently running under **{workflow_choice}**.  
Use the sidebar on the left to navigate through real-time alerts, transaction investigation tools, model performance evaluations, and system health metrics.
""")

df_proc = get_cached_processed_data()

# Top Summary KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions", f"{len(df_proc):,}")
with col2:
    fraud_count = int((df_proc[settings.TARGET_COLUMN] == 1).sum()) if settings.TARGET_COLUMN in df_proc.columns else 0
    st.metric("Fraudulent Transactions", f"{fraud_count:,}")
with col3:
    legit_count = len(df_proc) - fraud_count
    st.metric("Legitimate Transactions", f"{legit_count:,}")
with col4:
    avg_speed = float(df_proc["spending_speed"].mean()) if "spending_speed" in df_proc.columns else 0.0
    st.metric("Avg Spending Velocity", f"{avg_speed:.2f}")

st.divider()
st.info("👈 **Select a page from the sidebar to begin analysis.**")