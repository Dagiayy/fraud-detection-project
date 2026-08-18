# tests/unit/test_security.py
from src.utils.security import hash_pii

def test_hash_pii():
    h1 = hash_pii("12345")
    h2 = hash_pii("12345")
    h3 = hash_pii("67890")
    
    assert len(h1) == 64  # SHA-256 hex string length
    assert h1 == h2       # Deterministic hashing
    assert h1 != h3       # Distinct outputs
