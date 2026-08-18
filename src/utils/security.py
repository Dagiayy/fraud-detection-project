# src/utils/security.py
import hashlib
from src.config import settings

def hash_pii(value: str) -> str:
    """Hashes PII attributes (IP addresses, user IDs, device fingerprints) using SHA-256 with salt."""
    if not value:
        return ""
    salted_value = f"{settings.HASH_SALT}:{value}"
    return hashlib.sha256(salted_value.encode("utf-8")).hexdigest()
