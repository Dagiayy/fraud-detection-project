# dashboard/pages/5_📋_Data_Quality.py
import streamlit as st
import pandas as pd
from dashboard.app import get_cached_raw_data
from src.data.validation import DataQualityValidator

st.set_page_config(page_title="Data Quality", layout="wide")
st.title("📋 Data Quality & Schema Integrity Report")

df_raw = get_cached_raw_data()
validator = DataQualityValidator(df_raw)
report = validator.validate_schema()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Schema Status", "PASSED" if report["is_valid"] else "FAILED")
with col2:
    st.metric("Total Ingested Records", f"{report['total_records']:,}")
with col3:
    st.metric("Duplicate Rows", report["duplicate_rows"])

st.subheader("Null Value Distribution per Field")
df_nulls = pd.DataFrame(list(report["null_counts"].items()), columns=["Field", "Null Count"])
st.dataframe(df_nulls, use_container_width=True)

st.subheader("Schema Compliance Summary")
st.json({
    "missing_required_columns": report["missing_columns"],
    "invalid_amounts_detected": report["invalid_amounts"],
    "invalid_ages_detected": report["invalid_ages"],
    "schema_validation_passed": report["is_valid"]
})
