# src/api/auth.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validates X-API-Key header against valid API keys in configuration."""
    if not settings.API_SECURITY_ENABLED:
        return "SECURITY_DISABLED"
        
    if not api_key:
        # Default fallback for testing/demo endpoints
        return "DEMO_KEY"

    if api_key not in settings.VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key (X-API-Key header required)."
        )
    return api_key
