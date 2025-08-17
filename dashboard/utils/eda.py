import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------
# Class Distribution
# -------------------------
def plot_class_distribution(df: pd.DataFrame):
    st.subheader("Class Distribution")
    sns.countplot(x='class', data=df)
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------
# Correlation Heatmap
# -------------------------
def plot_correlation_heatmap(df: pd.DataFrame):
    st.subheader("Correlation Heatmap")
    num_cols = df.select_dtypes(include='number').columns
    plt.figure(figsize=(12,8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------
# Fraud by Aggregations
# -------------------------
def plot_fraud_by_country(df: pd.DataFrame):
    if 'country' in df.columns:
        st.subheader("Fraud by Country")
        fraud_country = df.groupby('country')['class'].sum().sort_values(ascending=False)
        st.bar_chart(fraud_country)

def plot_fraud_by_source(df: pd.DataFrame):
    if 'source' in df.columns:
        st.subheader("Fraud by Source")
        fraud_source = df.groupby('source')['class'].sum()
        st.bar_chart(fraud_source)

def plot_fraud_by_browser(df: pd.DataFrame):
    if 'browser' in df.columns:
        st.subheader("Fraud by Browser")
        fraud_browser = df.groupby('browser')['class'].sum()
        st.bar_chart(fraud_browser)

def plot_fraud_by_hour(df: pd.DataFrame):
    if 'hour_of_day' in df.columns:
        st.subheader("Fraud by Hour of Day")
        fraud_hour = df.groupby('hour_of_day')['class'].sum()
        st.line_chart(fraud_hour)

# -------------------------
# Summary Metrics
# -------------------------
def summary_metrics(df: pd.DataFrame):
    st.subheader("Top Risk Metrics")
    st.metric("Total Transactions", df.shape[0])
    st.metric("Total Fraud Cases", int(df['class'].sum()))
    if 'country' in df.columns:
        top_country = df.groupby('country')['class'].sum().idxmax()
        st.metric("High-Risk Country", top_country)
    if 'source' in df.columns:
        top_source = df.groupby('source')['class'].sum().idxmax()
        st.metric("High-Risk Source", top_source)
    if 'hour_of_day' in df.columns:
        top_hour = df.groupby('hour_of_day')['class'].sum().idxmax()
        st.metric("High-Risk Hour", top_hour)
