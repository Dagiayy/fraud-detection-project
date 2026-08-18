# tests/integration/test_pipeline.py
from src.data.synthetic_generator import generate_synthetic_transactions
from src.preprocessing.preprocessor import FraudPreprocessor
from src.models.predict import FraudPredictor
from src.decision.risk_engine import FraudDecisionEngine

def test_full_pipeline_integration():
    # 1. Generate synthetic data batch
    df_syn = generate_synthetic_transactions(num_records=50)
    assert len(df_syn) == 50
    
    # 2. Preprocess batch
    preprocessor = FraudPreprocessor()
    df_proc = preprocessor.clean_and_transform(df_syn, is_training=True)
    assert not df_proc.empty
    
    # 3. Score predictions
    predictor = FraudPredictor()
    probs = predictor.predict_proba(df_syn)
    assert len(probs) == 50
    
    # 4. Decision Engine
    decision_engine = FraudDecisionEngine()
    for idx, prob in enumerate(probs):
        res = decision_engine.evaluate_transaction(df_proc.iloc[idx], prob)
        assert res["decision"] in ["ALLOW", "REVIEW", "BLOCK"]
        assert res["risk_band"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
