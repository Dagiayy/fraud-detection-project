# src/data/ingestion.py
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from src.config import settings

def load_raw_data(
    fraud_path: Optional[Path] = None,
    ip_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw fraud transactions and IP range datasets."""
    f_path = fraud_path or settings.FRAUD_DATA_PATH
    i_path = ip_path or settings.IP_COUNTRY_PATH

    if f_path.exists() and i_path.exists():
        df_fraud = pd.read_csv(f_path)
        df_ip = pd.read_csv(i_path)
    else:
        # Fallback to synthetic generator if raw CSVs are missing
        from src.data.synthetic_generator import generate_synthetic_transactions
        df_fraud = generate_synthetic_transactions(num_records=2000)
        df_ip = pd.DataFrame({
            "lower_bound_ip_address": [0, 1000000000, 2000000000],
            "upper_bound_ip_address": [999999999, 1999999999, 4000000000],
            "country": ["United States", "China", "Unknown"]
        })

    return df_fraud, df_ip
