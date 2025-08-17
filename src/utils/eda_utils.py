# src/eda_utils.py
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_class_distribution(df: pd.DataFrame, target: str = "class") -> None:
    """Bar plot of class imbalance."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe.")
    sns.countplot(x=target, data=df)
    plt.title("Class Distribution")
    plt.show()


def plot_corr_heatmap(df: pd.DataFrame, include_target: bool = True, target: str = "class") -> None:
    """Numeric correlation heatmap."""
    cols = df.select_dtypes(include="number").columns.tolist()
    if include_target and target in df.columns and target not in cols:
        cols.append(target)
    corr = df[cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (numeric)")
    plt.show()


def _ensure_country_column(df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct 'country' column from one-hot columns if needed.
    Returns a pd.Series of country values aligned to df.index.
    """
    if "country" in df.columns:
        return df["country"]

    country_cols = [c for c in df.columns if c.startswith("country_")]
    if not country_cols:
        raise ValueError("No 'country' or one-hot 'country_*' columns found.")
    return df[country_cols].idxmax(axis=1).str.replace("country_", "", regex=False)


def plot_fraud_count_by_country(df: pd.DataFrame, target: str = "class", top_k: int = 20) -> None:
    """Top-k countries by fraud count."""
    country = _ensure_country_column(df)
    tmp = df.copy()
    tmp["country"] = country
    agg = tmp.groupby("country")[target].sum().sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=agg.index, y=agg.values)
    plt.title(f"Top {top_k} Countries by Fraud Count")
    plt.ylabel("Fraud Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_fraud_rate_by_country(df: pd.DataFrame, target: str = "class", top_k: int = 20, min_txn: int = 10) -> None:
    """Top-k countries by fraud rate with minimum transaction threshold."""
    country = _ensure_country_column(df)
    tmp = df.copy()
    tmp["country"] = country
    grp = tmp.groupby("country")[target]
    rate = grp.mean()
    size = grp.size()
    rate = rate[size >= min_txn].sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=rate.index, y=rate.values)
    plt.title(f"Top {top_k} Countries by Fraud Rate (min {min_txn} txns)")
    plt.ylabel("Fraud Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_fraud_by_hour(df: pd.DataFrame, target: str = "class") -> None:
    """Fraud rate by hour of day."""
    if "hour_of_day" not in df.columns:
        raise ValueError("'hour_of_day' not found. Run time feature engineering first.")
    rate = df.groupby("hour_of_day")[target].mean()
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=rate.index, y=rate.values, marker="o")
    plt.title("Fraud Rate by Hour of Day")
    plt.ylabel("Fraud Rate")
    plt.xlabel("Hour of Day (0-23)")
    plt.show()


def plot_fraud_by_day(df: pd.DataFrame, target: str = "class") -> None:
    """Fraud rate by day of week (0=Mon)."""
    if "day_of_week" not in df.columns:
        raise ValueError("'day_of_week' not found. Run time feature engineering first.")
    rate = df.groupby("day_of_week")[target].mean()
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=rate.index, y=rate.values, marker="o")
    plt.title("Fraud Rate by Day of Week (0=Mon)")
    plt.ylabel("Fraud Rate")
    plt.xlabel("Day of Week")
    plt.show()


def plot_spending_speed_distribution(df: pd.DataFrame, target: str = "class") -> None:
    """Distribution of spending_speed split by class."""
    if "spending_speed" not in df.columns:
        raise ValueError("'spending_speed' not found. Compute velocity first.")
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="spending_speed", hue=target, bins=50, kde=True, log_scale=True)
    plt.title("Spending Speed Distribution by Class")
    plt.xlabel("spending_speed (purchase_value / time_since_signup)")
    plt.show()


def pca_variance_plot(credit_df: pd.DataFrame, n_components: int = 10) -> None:
    """
    PCA explained variance for anonymized credit card features V1..V28.
    """
    cols = [f"V{i}" for i in range(1, 29)]
    missing = [c for c in cols if c not in credit_df.columns]
    if missing:
        raise ValueError(f"Credit card dataframe missing columns: {missing}")
    scaler = StandardScaler()
    X = scaler.fit_transform(credit_df[cols])
    pca = PCA(n_components=n_components).fit(X)
    cum = pca.explained_variance_ratio_.cumsum()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, n_components + 1), y=cum, marker="o")
    plt.title("PCA Cumulative Explained Variance (V1..V28)")
    plt.xlabel("# Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.grid(True)
    plt.show()
