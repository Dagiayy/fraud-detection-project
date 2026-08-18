# src/data/validation.py
import pandas as pd
from typing import Dict, Any, List

class DataQualityValidator:
    """Validator for schema integrity and data quality monitoring."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def validate_schema(self) -> Dict[str, Any]:
        required_cols = {"user_id", "signup_time", "purchase_time", "purchase_value", "age", "ip_address", "source", "browser", "sex"}
        missing_cols = required_cols - set(self.df.columns)
        
        null_counts = self.df.isnull().sum().to_dict()
        duplicate_rows = int(self.df.duplicated().sum())
        invalid_amounts = int((self.df["purchase_value"] <= 0).sum()) if "purchase_value" in self.df.columns else 0
        invalid_ages = int(((self.df["age"] < 18) | (self.df["age"] > 120)).sum()) if "age" in self.df.columns else 0

        is_valid = len(missing_cols) == 0 and invalid_amounts == 0

        return {
            "is_valid": is_valid,
            "total_records": len(self.df),
            "missing_columns": list(missing_cols),
            "null_counts": null_counts,
            "duplicate_rows": duplicate_rows,
            "invalid_amounts": invalid_amounts,
            "invalid_ages": invalid_ages,
        }
