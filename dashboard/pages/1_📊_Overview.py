# dashboard/pages/1_📊_Overview.py
import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard.app import get_cached_raw_data, get_cached_processed_data

st.set_page_config(page_title="Overview - Fraud Detection", layout="wide")
st.title("📊 Dataset Overview & Distribution Analysis")

df_raw = get_cached_raw_data()
df_proc = get_cached_processed_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Class Distribution (Before SMOTE)")
    if "class" in df_raw.columns:
        fig1 = px.histogram(df_raw, x="class", color="class", color_discrete_map={0: "green", 1: "red"}, title="Raw Class Imbalance (~9% Fraud)")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Demonstrates the severe class imbalance in raw transaction data prior to leakage-proof resampling.")

with col2:
    st.subheader("Purchase Value Distribution")
    if "purchase_value" in df_raw.columns:
        fig2 = px.histogram(df_raw, x="purchase_value", nbins=40, title="Purchase Value Range ($)", color_discrete_sequence=["#1B365D"])
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Most transactions fall in $20-$60 range, with high-value tail purchases representing elevated fraud exposure.")

st.subheader("Geographical Distribution (Top Countries)")
if "country" in df_proc.columns:
    country_counts = df_proc["country"].value_counts().head(15).reset_index()
    country_counts.columns = ["Country", "Transaction Count"]
    fig3 = px.bar(country_counts, x="Country", y="Transaction Count", title="Top 15 Countries by Transaction Volume", color_discrete_sequence=["#006699"])
    st.plotly_chart(fig3, use_container_width=True)
