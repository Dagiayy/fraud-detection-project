# scripts/run_preprocessing.py
from __future__ import annotations  # must be first

import argparse
import os
import sys

# -------------------------
# Add project root to sys.path
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# -------------------------
# Correct import path
# -------------------------
from src.utils.refactored_preprocessor import FraudPreprocessor, Paths

# -------------------------
# Argument parser
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fraud preprocessing pipeline.")
    
    # Default paths
    default_fraud = os.path.join(PROJECT_ROOT, "data/raw/Fraud_Data.csv")
    default_ip = os.path.join(PROJECT_ROOT, "data/raw/IpAddress_to_Country.csv")
    default_credit = os.path.join(PROJECT_ROOT, "data/raw/creditcard.csv")
    default_out = os.path.join(PROJECT_ROOT, "data/processed/enhanced_processed_fraud_data.csv")
    
    parser.add_argument("--fraud_path", type=str, default=default_fraud, help="Path to Fraud_Data.csv")
    parser.add_argument("--ip_path", type=str, default=default_ip, help="Path to IP-to-Country CSV")
    parser.add_argument("--credit_path", type=str, default=default_credit, help="Path to creditcard.csv (optional)")
    parser.add_argument("--out", type=str, default=default_out, help="Path to save processed CSV")
    parser.add_argument("--balance", type=str, default="smote", choices=["smote", "under", "none"], help="Resampling method")
    
    return parser.parse_args()

# -------------------------
# Main function
# -------------------------
def main() -> None:
    args = parse_args()
    
    paths = Paths(fraud_path=args.fraud_path, ip_path=args.ip_path, credit_path=args.credit_path)
    pp = FraudPreprocessor(paths)
    
    pp.load_data()
    pp.preprocess(target_col="class", balance_method=None if args.balance == "none" else args.balance)
    pp.save_processed_data(args.out)
    
    print(f"✅ Preprocessing complete. Saved: {args.out}")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
