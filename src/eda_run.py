# scripts/run_eda.py
from __future__ import annotations

import argparse
import pandas as pd

from src.eda_utils import (
    plot_class_distribution,
    plot_corr_heatmap,
    plot_fraud_count_by_country,
    plot_fraud_rate_by_country,
    plot_fraud_by_day,
    plot_fraud_by_hour,
    plot_spending_speed_distribution,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EDA visuals for processed fraud dataset.")
    p.add_argument("--data", type=str, required=True, help="Path to processed fraud CSV")
    p.add_argument("--target", type=str, default="class")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)

    plot_class_distribution(df, target=args.target)
    plot_corr_heatmap(df, include_target=True, target=args.target)
    # These will reconstruct 'country' if one-hot encoded:
    plot_fraud_count_by_country(df, target=args.target, top_k=20)
    plot_fraud_rate_by_country(df, target=args.target, top_k=20, min_txn=10)
    # Time and velocity views
    if "hour_of_day" in df.columns:
        plot_fraud_by_hour(df, target=args.target)
    if "day_of_week" in df.columns:
        plot_fraud_by_day(df, target=args.target)
    if "spending_speed" in df.columns:
        plot_spending_speed_distribution(df, target=args.target)


if __name__ == "__main__":
    main()
