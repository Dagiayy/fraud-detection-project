# src/api/main.py
import uuid
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from typing import Dict, Any, List

from src.config import settings
from src.data.data_contract import TransactionSchema, BatchTransactionSchema
from src.data.validation import DataQualityValidator
from src.models.predict import FraudPredictor
from src.decision.risk_engine import FraudDecisionEngine
from src.explainability.shap_explainer import FraudSHAPExplainer
from src.api.auth import verify_api_key
from src.utils.security import hash_pii

app = FastAPI(
    title="💳 Real-Time Fraud Detection REST API",
    description="Enterprise API providing real-time transaction fraud scoring, decision rules, and SHAP explainability.",
    version="2.1.0"
)

predictor = FraudPredictor()
decision_engine = FraudDecisionEngine()
shap_explainer = FraudSHAPExplainer()

@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/health", tags=["Monitoring"])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "HEALTHY", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/ready", tags=["Monitoring"])
def readiness_check() -> Dict[str, Any]:
    """Readiness endpoint verifying model availability."""
    is_ready = predictor.model is not None or settings.PROCESSED_DATA_PATH.exists()
    return {
        "status": "READY" if is_ready else "NOT_READY",
        "model_loaded": predictor.model is not None,
        "preprocessor_ready": predictor.preprocessor is not None
    }

@app.get("/model/info", tags=["Metadata"])
def model_info(api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Returns model metadata and feature schema."""
    return {
        "model_type": "LightGBM Classifier (Production)",
        "version": "2.1.0",
        "decision_threshold": settings.DEFAULT_DECISION_THRESHOLD,
        "feature_count": len(predictor.feature_names),
        "feature_names": predictor.feature_names,
        "risk_bands": settings.RISK_BANDS
    }

@app.post("/predict", tags=["Inference"])
def predict_transaction(
    transaction: TransactionSchema, 
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Score a single transaction payload.
    Returns fraud probability, decision (ALLOW/REVIEW/BLOCK), risk band, and SHAP explanations.
    """
    txn_dict = transaction.model_dump(by_alias=True)
    df_single = pd.DataFrame([txn_dict])

    # Validate Schema
    validator = DataQualityValidator(df_single)
    val_report = validator.validate_schema()
    if not val_report["is_valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid transaction payload. Missing columns: {val_report['missing_columns']}")

    # 1. Score Probability
    probs = predictor.predict_proba(df_single)
    prob = float(probs[0])

    # 2. Extract Preprocessed row for SHAP & Decision Engine
    df_proc = predictor.preprocessor.clean_and_transform(df_single, is_training=False)
    row_feat = df_proc.iloc[0] if not df_proc.empty else pd.Series()

    # 3. Decision Engine
    decision_out = decision_engine.evaluate_transaction(row_feat, prob)

    # 4. SHAP Explanation
    shap_out = shap_explainer.explain_transaction(df_proc)

    # 5. PII Hashing for Security Audit
    hashed_user = hash_pii(str(transaction.user_id))

    return {
        "transaction_id": transaction.transaction_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hashed_user_id": hashed_user,
        "score_results": decision_out,
        "explanation": shap_out,
        "environment": "PRODUCTION_ENHANCED"
    }

@app.post("/predict/batch", tags=["Inference"])
def predict_batch(
    batch: BatchTransactionSchema,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Score a batch of transactions."""
    txns = [t.model_dump(by_alias=True) for t in batch.transactions]
    df_batch = pd.DataFrame(txns)

    probs = predictor.predict_proba(df_batch)
    df_proc = predictor.preprocessor.clean_and_transform(df_batch, is_training=False)

    results = []
    for idx, row_raw in df_batch.iterrows():
        p = float(probs[idx])
        row_p = df_proc.iloc[[idx]]
        decision_out = decision_engine.evaluate_transaction(row_p.iloc[0], p)
        shap_out = shap_explainer.explain_transaction(row_p)
        
        results.append({
            "user_id": int(row_raw.get("user_id", 0)),
            "hashed_user_id": hash_pii(str(row_raw.get("user_id", 0))),
            "purchase_value": float(row_raw.get("purchase_value", 0.0)),
            "score_results": decision_out,
            "top_risk_factors": shap_out["top_positive_risk_factors"]
        })

    return {
        "total_scored": len(results),
        "results": results
    }
