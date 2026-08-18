# src/models/evaluate.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.config import settings

def evaluate_models(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.35) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics including ROC-AUC, AUC-PR, F1, and confusion matrix."""
    y_preds = (y_probs >= threshold).astype(int)
    
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auc_pr = float(auc(recall, precision))
    roc_auc = float(roc_auc_score(y_true, y_probs))
    f1 = float(f1_score(y_true, y_preds, zero_division=0))
    cm = confusion_matrix(y_true, y_preds).tolist()
    
    # Precision-Recall threshold sweep to find optimal threshold
    best_thresh = threshold
    best_f1 = f1
    for t in np.linspace(0.1, 0.9, 17):
        p_temp = (y_probs >= t).astype(int)
        f_temp = f1_score(y_true, p_temp, zero_division=0)
        if f_temp > best_f1:
            best_f1 = f_temp
            best_thresh = t

    return {
        "roc_auc": roc_auc,
        "auc_pr": auc_pr,
        "f1_score": f1,
        "optimal_threshold": float(best_thresh),
        "best_f1_score": float(best_f1),
        "confusion_matrix": cm,
        "threshold_used": threshold
    }

def cost_sensitive_evaluation(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.35) -> Dict[str, float]:
    """Calculate financial cost savings based on cost parameters."""
    y_preds = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Financial cost calculation
    total_cost = (
        fp * settings.COST_FALSE_POSITIVE +
        fn * settings.COST_FALSE_NEGATIVE +
        (fp + tp) * settings.COST_REVIEW
    )
    
    # Baseline cost (if no model used and all frauds missed)
    baseline_cost = (tp + fn) * settings.COST_FALSE_NEGATIVE
    net_savings = max(baseline_cost - total_cost, 0.0)
    savings_pct = (net_savings / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    return {
        "total_cost_usd": float(total_cost),
        "baseline_cost_usd": float(baseline_cost),
        "net_savings_usd": float(net_savings),
        "savings_percentage": float(savings_pct)
    }
