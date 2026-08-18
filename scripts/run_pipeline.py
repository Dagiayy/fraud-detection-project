# scripts/run_pipeline.py
from __future__ import annotations
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.train import train_pipeline

def main():
    print("=== Starting End-to-End Leakage-Proof Fraud Detection Training Pipeline ===")
    metadata = train_pipeline()
    print("\n--- Pipeline Execution Summary ---")
    print(f"* Model Type: {metadata['model_type']}")
    print(f"* Train Samples (Resampled): {metadata['train_samples_resampled']}")
    print(f"* Test Samples (Untouched): {metadata['test_samples_untouched']}")
    print(f"* Test ROC-AUC Score: {metadata['evaluation_metrics']['roc_auc']:.4f}")
    print(f"* Test Average Precision (AUC-PR): {metadata['evaluation_metrics']['auc_pr']:.4f}")
    print(f"* Estimated Net Cost Savings: ${metadata['financial_cost_analysis']['net_savings_usd']:,.2f}")
    print("\n[SUCCESS] End-to-End Pipeline Execution Completed!")

if __name__ == "__main__":
    main()
