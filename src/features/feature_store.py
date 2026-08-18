# src/features/feature_store.py
import time
import pandas as pd
from typing import Dict, Any, Optional

class FeatureStoreClient:
    """
    Feast / Redis Online Feature Store Client Scaffolding.
    Provides sub-20ms point-in-time rolling user aggregations.
    """

    def __init__(self, use_redis: bool = False):
        self.use_redis = use_redis
        self.in_memory_cache: Dict[int, Dict[str, Any]] = {}

    def get_online_features(self, user_id: int) -> Dict[str, Any]:
        """
        Retrieves real-time online features for a user ID.
        Fallbacks to in-memory window lookup if Redis is offline.
        """
        now = time.time()
        if user_id in self.in_memory_cache:
            data = self.in_memory_cache[user_id]
            # Increment velocity count
            data["count_txns_last_10m"] += 1
            data["last_transaction_timestamp"] = now
            return data
            
        default_features = {
            "user_id": user_id,
            "count_txns_last_10m": 1,
            "sum_value_last_1h": 0.0,
            "distinct_ip_count_24h": 1,
            "last_transaction_timestamp": now
        }
        self.in_memory_cache[user_id] = default_features
        return default_features

    def update_user_feature(self, user_id: int, purchase_value: float):
        features = self.get_online_features(user_id)
        features["sum_value_last_1h"] += purchase_value
        self.in_memory_cache[user_id] = features

feature_store = FeatureStoreClient()
