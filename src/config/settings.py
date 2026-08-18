# src/config/settings.py
import os
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict

class Settings(BaseModel):
    # System Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # File Paths
    FRAUD_DATA_PATH: Path = DATA_RAW_DIR / "Fraud_Data.csv"
    IP_COUNTRY_PATH: Path = DATA_RAW_DIR / "IpAddress_to_Country.csv"
    CREDIT_CARD_PATH: Path = DATA_RAW_DIR / "creditcard.csv"
    PROCESSED_DATA_PATH: Path = DATA_PROCESSED_DIR / "enhanced_processed_fraud_data.csv"
    
    LIGHTGBM_MODEL_PATH: Path = MODELS_DIR / "lightgbm_model.txt"
    LOGISTIC_REGRESSION_MODEL_PATH: Path = MODELS_DIR / "logistic_regression_model.pkl"
    MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
    
    # Global Parameters
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    DEFAULT_DECISION_THRESHOLD: float = 0.35
    
    # API & Security (Configurable for Workflow 1 vs Workflow 2)
    # Standard Workflow = False (No Auth needed), Production Workflow = True
    API_SECURITY_ENABLED: bool = os.getenv("FRAUD_API_SECURITY_ENABLED", "False").lower() in ("true", "1", "t")
    VALID_API_KEYS: List[str] = ["adey-fraud-secret-key-2025", "dev-test-key"]
    HASH_SALT: str = "adey_innovations_salt_2025"
    
    # Feature Configuration
    NUMERICAL_FEATURES: List[str] = [
        "purchase_value", "age", "spending_speed", 
        "hour_of_day", "day_of_week", "time_since_signup", "transaction_frequency"
    ]
    CATEGORICAL_FEATURES: List[str] = ["source", "browser", "sex", "country"]
    TARGET_COLUMN: str = "class"
    
    # Risk Bands
    RISK_BANDS: Dict[str, tuple] = {
        "LOW": (0.0, 0.20),
        "MEDIUM": (0.20, 0.50),
        "HIGH": (0.50, 0.80),
        "CRITICAL": (0.80, 1.00)
    }
    
    # Financial Cost Parameters (Cost-Sensitive Evaluation)
    COST_FALSE_POSITIVE: float = 10.0  # Friction / review cost
    COST_FALSE_NEGATIVE: float = 150.0 # Average fraud loss
    COST_REVIEW: float = 5.0           # Manual investigation cost

settings = Settings()

    # Cache & Feature Store
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
