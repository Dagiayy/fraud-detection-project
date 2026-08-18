# scripts/run_pipeline.py
from __future__ import annotations
import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import settings
from src.models.train import train_pipeline
from src.data.ingestion import load_raw_data
from src.monitoring.drift import DataDriftDetector

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fraud Detection Pipeline.")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument(
        "--mode", 
        type=str, 
        default="standard", 
        choices=["standard", "production"], 
        help="Workflow Mode: 'standard' (Workflow 1: Fast ML Pipeline) or 'production' (Workflow 2: Production + Drift + Feature Store)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == "standard":
        print("==================================================================")
        print("[WORKFLOW 1] STANDARD ML PIPELINE (FAST LOCAL EXECUTION)")
        print("==================================================================")
        metadata = train_pipeline()
        print("\n--- Pipeline Execution Summary ---")
        print(f"* Model Type: {metadata['model_type']}")
        print(f"* Train Samples (Resampled): {metadata['train_samples_resampled']}")
        print(f"* Test Samples (Untouched): {metadata['test_samples_untouched']}")
        print(f"* Test ROC-AUC Score: {metadata['evaluation_metrics']['roc_auc']:.4f}")
        print(f"* Test Average Precision (AUC-PR): {metadata['evaluation_metrics']['auc_pr']:.4f}")
        print(f"* Estimated Net Cost Savings: ${metadata['financial_cost_analysis']['net_savings_usd']:,.2f}")
        print("\n[SUCCESS] Standard ML Pipeline Execution Completed!")

    elif args.mode == "production":
        print("==================================================================")
        print("[WORKFLOW 2] ENTERPRISE PRODUCTION STACK (SECURITY + DRIFT + MLOPS)")
        print("==================================================================")
        # Enable API Security
        settings.API_SECURITY_ENABLED = True
        
        # 1. Execute Leakage-Proof Training
        metadata = train_pipeline()
        
        # 2. Execute Data Drift Audit
        df_raw, _ = load_raw_data()
        detector = DataDriftDetector(df_raw)
        drift_report = detector.detect_drift(df_raw)
        
        print("\n--- Production Stack Summary ---")
        print(f"* Active Model: {metadata['model_type']}")
        print(f"* API Security Enabled: {settings.API_SECURITY_ENABLED}")
        print(f"* Valid API Keys Configured: {len(settings.VALID_API_KEYS)}")
        print(f"* Data Drift Status: {drift_report['overall_drift_status']}")
        print(f"* Features Monitored for Drift: {drift_report['total_features_monitored']}")
        print(f"* Estimated Financial Savings: ${metadata['financial_cost_analysis']['net_savings_usd']:,.2f}")
        print("\n[SUCCESS] Production Stack Pipeline Execution Completed!")

if __name__ == "__main__":
    main()
