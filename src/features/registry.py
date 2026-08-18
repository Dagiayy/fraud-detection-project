# src/features/registry.py
from typing import Dict, Any

FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "purchase_value": {
        "description": "Monetary value of the transaction in USD",
        "data_type": "float",
        "leakage_risk": "None",
        "source": "Transaction Payload"
    },
    "time_since_signup": {
        "description": "Hours elapsed between account signup and transaction execution",
        "data_type": "float",
        "leakage_risk": "None",
        "source": "Derived (purchase_time - signup_time)"
    },
    "spending_speed": {
        "description": "Transaction velocity computed as purchase_value / time_since_signup",
        "data_type": "float",
        "leakage_risk": "None",
        "source": "Derived Velocity"
    },
    "is_new_user": {
        "description": "Binary flag indicating signup within 24 hours of purchase",
        "data_type": "int",
        "leakage_risk": "None",
        "source": "Derived Rule Flag"
    },
    "is_rapid_spender": {
        "description": "Binary flag indicating spending speed exceeding 95th percentile",
        "data_type": "int",
        "leakage_risk": "None",
        "source": "Derived Rule Flag"
    },
    "is_high_value": {
        "description": "Binary flag indicating purchase value exceeding $100",
        "data_type": "int",
        "leakage_risk": "None",
        "source": "Derived Rule Flag"
    },
    "hour_of_day": {
        "description": "Hour of day (0-23) when purchase occurred",
        "data_type": "int",
        "leakage_risk": "None",
        "source": "Derived Temporal"
    },
    "day_of_week": {
        "description": "Day of week (0=Mon, 6=Sun) when purchase occurred",
        "data_type": "int",
        "leakage_risk": "None",
        "source": "Derived Temporal"
    },
    "transaction_frequency": {
        "description": "Count of transactions originated by user ID",
        "data_type": "int",
        "leakage_risk": "None",
        "source": "Derived User Count"
    }
}
