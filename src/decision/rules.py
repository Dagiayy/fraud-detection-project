# src/decision/rules.py
import pandas as pd
from typing import Dict, Any, List, Tuple

def evaluate_business_rules(row: pd.Series) -> Tuple[List[str], bool]:
    """
    Evaluates hard policy rules on a single transaction row.
    Returns (reason_codes, is_hard_block).
    """
    reasons = []
    hard_block = False

    time_since_signup = row.get("time_since_signup", 100.0)
    spending_speed = row.get("spending_speed", 0.0)
    purchase_value = row.get("purchase_value", 0.0)
    
    # Rule 1: Instant transaction after registration (< 0.1 hours = 6 minutes)
    if time_since_signup < 0.1 and purchase_value > 50:
        reasons.append("RULE_INSTANT_PURCHASE_AFTER_SIGNUP")
        hard_block = True

    # Rule 2: Extreme spending velocity (> 20.0)
    if spending_speed > 20.0:
        reasons.append("RULE_EXTREME_SPENDING_VELOCITY")
        hard_block = True

    # Rule 3: High value new account purchase
    if time_since_signup <= 24.0 and purchase_value > 200:
        reasons.append("RULE_NEW_ACCOUNT_HIGH_VALUE")

    # Rule 4: High spending speed flag
    if spending_speed > 5.0:
        reasons.append("RULE_SPENDING_SPEED_ELEVATED")

    return reasons, hard_block
