from __future__ import annotations

import os
from pathlib import Path
from typing import List
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
import shap
import joblib

# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(
    page_title="💳 Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Fraud Detection Dashboard")
st.markdown("""
This interactive dashboard provides actionable fraud detection insights for non-technical stakeholders.
It helps identify high-risk transactions, reduce financial losses, and enhance trust with transparent decision-making.
""")

# -------------------------
# LOAD DATA
# -------------------------
DATA_PATH = Path("data/processed/enhanced_processed_fraud_data.csv")
RAW_DATA_PATH = Path("data/raw/Fraud_Data.csv")
CREDIT_DATA_PATH = Path("data/raw/creditcard.csv")

if not DATA_PATH.exists() or not RAW_DATA_PATH.exists() or not CREDIT_DATA_PATH.exists():
    st.error(f"Dataset not found at `{DATA_PATH}`, `{RAW_DATA_PATH}`, or `{CREDIT_DATA_PATH}`")
    st.stop()

df = pd.read_csv(DATA_PATH)
raw_df = pd.read_csv(RAW_DATA_PATH)
df_credit = pd.read_csv(CREDIT_DATA_PATH)

# Reconstruct 'country' column from one-hot encoding
country_cols = [c for c in df.columns if c.startswith("country_")]
df['country'] = df[country_cols].idxmax(axis=1).str.replace("country_", "")

# Convert class to string for plotting
df['class_str'] = df['class'].astype(str)

# -------------------------
# SIDEBAR: Key Metrics & Filters
# -------------------------
st.sidebar.header("📊 Key Metrics")
st.sidebar.metric("Total Transactions", len(df))
st.sidebar.metric("Fraudulent Transactions", (df['class'] == 1).sum())
st.sidebar.metric("Legitimate Transactions", (df['class'] == 0).sum())
st.sidebar.metric("Avg Spending Speed", f"{df['spending_speed'].mean():.2f}")

st.sidebar.header("🔍 Filters")
selected_class = st.sidebar.selectbox("Filter by Transaction Type", ["All", "Fraudulent", "Legitimate"])
selected_country = st.sidebar.multiselect(
    "Filter by Country", 
    options=sorted(df['country'].unique()), 
    default=None
)
time_range = st.sidebar.slider(
    "Filter by Time Since Signup (seconds)", 
    min_value=float(df['time_since_signup'].min()), 
    max_value=float(df['time_since_signup'].max()), 
    value=(float(df['time_since_signup'].min()), float(df['time_since_signup'].max()))
)

# Apply filters
filtered_df = df.copy()
if selected_class == "Fraudulent":
    filtered_df = filtered_df[filtered_df['class'] == 1]
elif selected_class == "Legitimate":
    filtered_df = filtered_df[filtered_df['class'] == 0]

if selected_country:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_country)]

filtered_df = filtered_df[
    (filtered_df['time_since_signup'] >= time_range[0]) & 
    (filtered_df['time_since_signup'] <= time_range[1])
]

# -------------------------
# TABS
# -------------------------
tab_overview, tab_risk_factors, tab_time_analysis, tab_geography, tab_top_insights, tab_model_performance = st.tabs([
    "📊 Overview",
    "🔍 Risk Factors",
    "⏰ Time-Based Analysis",
    "🌍 Geography & Sources",
    "🚨 Top-Risk Insights",
    "📈 Model Performance"
])

# -------------------------
# TAB 1: Overview
# -------------------------
with tab_overview:
    st.header("Class Distribution Before SMOTE")
    fig = px.histogram(raw_df, x='class', title="Fraud vs Non-Fraud (Before SMOTE)", color='class', color_discrete_map={0: 'green', 1: 'red'})
    fig.update_layout(xaxis_title="Fraud Class", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    This chart highlights the severe class imbalance before applying SMOTE. Fraud cases are rare (~9%), making it challenging for traditional models to learn patterns effectively.  
    **Business Impact:** Without balancing, the model would prioritize detecting legitimate transactions, leading to missed frauds and significant financial losses.
    """)

    st.header("Class Distribution After SMOTE")
    fig = px.pie(
        filtered_df, 
        names='class_str', 
        title="Fraud vs Non-Fraud Transactions (After SMOTE)", 
        color='class_str',
        color_discrete_map={'0': 'green', '1': 'red'},
        labels={'class_str': 'Fraud Class'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    After applying SMOTE, the dataset is balanced, enabling the model to learn equally from both fraudulent and legitimate transactions.  
    **Business Impact:** This ensures the system detects fraud effectively without over-penalizing legitimate users, reducing false positives and financial risk.
    """)

    st.header("Purchase Value Distribution")
    fig = px.histogram(raw_df, x='purchase_value', nbins=50, title="Distribution of Purchase Value", color='class', color_discrete_map={0: 'green', 1: 'red'})
    fig.update_layout(xaxis_title="Purchase Value", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    Most transactions involve small to medium purchases, but outliers (high-value transactions) may indicate fraud.  
    **Business Impact:** Monitoring high-value transactions helps mitigate large-scale financial losses caused by single fraudulent activities.
    """)

# -------------------------
# TAB 2: Risk Factors
# -------------------------
with tab_risk_factors:
    st.header("Spending Speed vs Fraud Probability")
    fig = px.scatter(
        filtered_df, 
        x='spending_speed', 
        y='class', 
        title="Spending Speed vs Fraud Probability", 
        color='class_str',
        color_discrete_map={'0': 'green', '1': 'red'},
        labels={'spending_speed': 'Spending Speed', 'class': 'Fraud Probability'}
    )
    fig.update_layout(xaxis_title="Spending Speed", yaxis_title="Fraud Probability")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    Higher spending speeds are strongly correlated with fraud, often indicating bot activity or stolen cards.  
    **Business Impact:** Implementing real-time alerts for rapid spending reduces exposure to automated attacks and minimizes financial losses.
    """)

    st.header("Time Since Signup vs Fraud Probability")
    fig = px.box(
        filtered_df, 
        x='class_str', 
        y='time_since_signup', 
        title="Time Since Signup vs Fraud Probability", 
        color='class_str',
        color_discrete_map={'0': 'green', '1': 'red'},
        labels={'class_str': 'Fraud Class', 'time_since_signup': 'Time Since Signup (seconds)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    New accounts (short time since signup) are more likely to be fraudulent, suggesting attackers test stolen credentials quickly.  
    **Business Impact:** Applying extra verification steps for new users prevents unauthorized access and protects sensitive data.
    """)

# -------------------------
# TAB 3: Time-Based Analysis
# -------------------------
with tab_time_analysis:
    st.header("Transactions by Hour of Day")
    raw_df['hour_of_day'] = pd.to_datetime(raw_df['purchase_time']).dt.hour
    hour_counts = raw_df.groupby(['hour_of_day', 'class']).size().reset_index(name='count')
    fig = px.bar(
        hour_counts, 
        x='hour_of_day', 
        y='count', 
        color='class', 
        barmode='group', 
        title="Transactions by Hour of Day",
        labels={'hour_of_day': 'Hour of Day', 'count': 'Number of Transactions'},
        color_discrete_map={0: 'green', 1: 'red'}
    )
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Number of Transactions")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    Fraud occurs throughout the day, but certain hours may see higher activity.  
    **Business Impact:** Allocating monitoring resources during peak fraud hours enhances detection efficiency and reduces operational costs.
    """)

    st.header("Transactions by Day of Week")
    # Unscale and map day_of_week
    day_mean = 3  # Example mean (adjust based on raw data)
    day_std = 2   # Example std (adjust based on raw data)
    df['day_of_week_plot'] = (df['day_of_week'] * day_std + day_mean).round().astype(int)
    df['day_of_week_plot'] = df['day_of_week_plot'].clip(0, 6)
    weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['day_of_week_plot'] = df['day_of_week_plot'].map(weekday_map)
    day_counts = df.groupby(['day_of_week_plot', 'class_str']).size().reset_index(name='count')
    fig = px.bar(
        day_counts, 
        x='day_of_week_plot', 
        y='count', 
        color='class_str', 
        barmode='group', 
        title="Transactions by Day of Week",
        labels={'day_of_week_plot': 'Day of Week', 'count': 'Number of Transactions'},
        color_discrete_map={'0': 'green', '1': 'red'}
    )
    fig.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Transactions")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    Certain days (e.g., weekends) may have higher fraud rates due to increased transaction volumes.  
    **Business Impact:** Strengthening fraud prevention during high-risk days protects revenue and maintains customer trust.
    """)

    st.header("Correlation Heatmap")
    num_cols = ['purchase_value', 'time_since_signup', 'spending_speed', 'transaction_frequency', 'hour_of_day', 'day_of_week']
    corr = df[num_cols + ['class']].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap: Fraud vs Features")
    st.pyplot(fig)
    st.markdown("""
    **Interpretation:**  
    This heatmap shows how strongly each feature correlates with fraud. High positive/negative values indicate strong relationships (e.g., spending_speed may correlate with fraud).  
    **Business Impact:** Focusing on highly correlated features improves model accuracy and targeted fraud prevention strategies.
    """)

# -------------------------
# TAB 4: Geography & Sources
# -------------------------
with tab_geography:
    st.header("Top Countries by Fraud Cases")
    fraud_country = filtered_df[filtered_df['class'] == 1]['country'].value_counts().head(15)
    fig = px.bar(
        fraud_country, 
        title="Top Countries by Fraud Cases", 
        labels={'index': 'Country', 'value': 'Number of Fraud Cases'}
    )
    fig.update_layout(xaxis_title="Country", yaxis_title="Number of Fraud Cases")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    Identifies countries with the highest fraud activity, enabling geolocation-based risk assessments.  
    **Business Impact:** Targeted monitoring and stricter verification for high-risk regions reduce exposure to global fraud threats.
    """)

    st.header("Transaction Sources by Fraud Rate")
    source_cols = ['source_Ads', 'source_Direct', 'source_SEO']
    filtered_df['source'] = filtered_df[source_cols].idxmax(axis=1).str.replace('source_', '')
    fraud_by_source = filtered_df.groupby('source')['class'].mean().sort_values(ascending=False)
    fig = px.bar(
        fraud_by_source, 
        title="Fraud Rate by Source", 
        labels={'index': 'Source', 'value': 'Fraud Rate'}
    )
    fig.update_layout(xaxis_title="Source", yaxis_title="Fraud Rate")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    Certain acquisition sources (e.g., SEO) have higher fraud rates than others (e.g., Ads).  
    **Business Impact:** Adjusting marketing strategies and implementing source-specific fraud checks minimize risks while maintaining user acquisition.
    """)

# -------------------------
# TAB 5: Top-Risk Insights
# -------------------------
with tab_top_insights:
    st.header("Top Devices by Fraud Cases")
    if 'device_id' in filtered_df.columns:
        top_devices = filtered_df[filtered_df['class'] == 1]['device_id'].value_counts().head(10)
        fig = px.bar(
            top_devices, 
            title="Top Devices by Fraud Cases", 
            labels={'index': 'Device ID', 'value': 'Number of Fraud Cases'}
        )
        fig.update_layout(xaxis_title="Device ID", yaxis_title="Number of Fraud Cases")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Interpretation:**  
        Identifies devices used in multiple fraud attempts, supporting device fingerprinting strategies.  
        **Business Impact:** Blacklisting suspicious devices prevents future fraudulent activities, reducing operational overhead.
        """)
    else:
        st.warning("The dataset does not contain a 'device_id' column. This insight is unavailable.")
        st.markdown("""
        **Alternative Insight:** Consider analyzing fraud by browser or source instead, as device_id is dropped to reduce high-cardinality features.
        """)

    st.header("Top Browsers by Fraud Rate")
    browser_cols = ['browser_Chrome', 'browser_FireFox', 'browser_IE', 'browser_Opera', 'browser_Safari']
    filtered_df['browser'] = filtered_df[browser_cols].idxmax(axis=1).str.replace('browser_', '')
    fraud_by_browser = filtered_df.groupby('browser')['class'].mean().sort_values(ascending=False)
    fig = px.bar(
        fraud_by_browser, 
        title="Fraud Rate by Browser", 
        labels={'index': 'Browser', 'value': 'Fraud Rate'}
    )
    fig.update_layout(xaxis_title="Browser", yaxis_title="Fraud Rate")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    Certain browsers may have higher fraud rates, guiding browser-specific fraud mitigation efforts.  
    **Business Impact:** Enhancing security measures for high-risk browsers improves detection accuracy and reduces false positives.
    """)


# -------------------------
# TAB 6: Model Performance
# -------------------------
with tab_model_performance:
    st.header("Model Performance and Explainability")

    # Load and preprocess data for modeling
    df_fraud = pd.read_csv("data/processed/enhanced_processed_fraud_data.csv")
    time_threshold = 24 * 60 * 60  # 24 hours in seconds
    spending_speed_threshold = df_fraud['spending_speed'].quantile(0.95)
    df_fraud['is_new_user'] = (df_fraud['time_since_signup'] < time_threshold).astype(int)
    df_fraud['is_rapid_spender'] = (df_fraud['spending_speed'] > spending_speed_threshold).astype(int)
    X = df_fraud.drop(columns=['class'])
    y = df_fraud['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Sanitize column names
    def sanitize_columns(df):
        df = df.copy()
        df.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in df.columns]
        return df
    X_train = sanitize_columns(X_train)
    X_test = sanitize_columns(X_test)

    # Train models
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)
    lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight='balanced', random_state=42)
    lgb_model.fit(X_train, y_train)

    # Evaluation function
    def evaluate_model(model, X_test, y_test, name="Model"):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(f"--- {name} ---")
        st.text(f"Precision (0): {report['0']['precision']:.2f}, Recall (0): {report['0']['recall']:.2f}")
        st.text(f"Precision (1): {report['1']['precision']:.2f}, Recall (1): {report['1']['recall']:.2f}")
        st.text(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.2f}")
        st.text(f"Average Precision (AUC-PR): {average_precision_score(y_test, y_proba):.2f}")
        cm = confusion_matrix(y_test, y_pred)
        st.text(f"Confusion Matrix:\n{cm}")

    # Evaluate models
    st.subheader("Logistic Regression Performance")
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    st.markdown("""
    **Interpretation:**  
    Logistic Regression provides a baseline performance with balanced precision and recall. High ROC-AUC indicates good overall discrimination between classes.  
    **Business Impact:** Useful for quick deployment but may need enhancement for rare fraud detection.
    """)

    st.subheader("LightGBM Performance")
    evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    st.markdown("""
    **Interpretation:**  
    LightGBM, a gradient boosting model, typically outperforms Logistic Regression with higher recall for fraud cases due to its ability to handle complex patterns.  
    **Business Impact:** Improves fraud detection accuracy, reducing financial losses from missed frauds.
    """)

    # SHAP Explainability
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    st.subheader("SHAP Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, ax=ax)
    st.pyplot(fig)
    st.markdown("""
    **Interpretation:**  
    This plot shows the most influential features driving fraud predictions (e.g., spending_speed, time_since_signup). Red indicates higher values increase fraud likelihood.  
    **Business Impact:** Focuses fraud prevention efforts on key risk factors, optimizing resource allocation.
    """)

    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model_with_new_features.pkl")
    joblib.dump(lgb_model, model_path)
    st.success(f"✅ Updated Model Saved to {model_path}")

    st.subheader("PCA Analysis for Credit Card Dataset")
    X_credit = df_credit[[f"V{i}" for i in range(1, 29)]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_credit)
    pca = PCA(n_components=10)
    pca.fit(X_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), explained_variance, marker='o')
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Credit Card PCA Explained Variance")
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("""
    **Interpretation:**  
    This plot shows how many principal components explain the variance in credit card features (V1-V28). Most variance is captured within 5-10 components.  
    **Business Impact:** Simplifies complex data for modeling, improving computational efficiency without losing key fraud signals.
    """)

# -------------------------
# Notes
# -------------------------
st.sidebar.markdown("**Last Updated:** 04:49 PM EAT, August 17, 2025")