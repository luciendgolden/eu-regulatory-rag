"""API key authentication dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from config import settings

# ---------------------------------------------------------------------------
# Security schemes
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_api_key_query = APIKeyQuery(name="api_key", auto_error=False)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def get_api_key(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> str:
    """Validate the API key supplied via header or query param.

    Raises HTTP 401 when the key is missing or invalid.
    """
    key = header_key or query_key

    # If no API_KEY is configured, authentication is disabled (dev mode).
    if not settings.api_key:
        return "dev"

    if not key or key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-API-Key header or api_key query param.",
        )
    return key


async def get_admin_api_key(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> str:
    """Validate the admin API key.

    Falls back to the regular API key when ADMIN_API_KEY is not configured.
    Raises HTTP 403 when the key is invalid.
    """
    key = header_key or query_key
    expected = settings.admin_api_key or settings.api_key

    # If neither key is configured, allow in dev mode.
    if not expected:
        return "dev"

    if not key or key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access denied. Invalid or missing admin API key.",
        )
    return key


# Shorthand Depends aliases for use in route signatures
RequireApiKey = Depends(get_api_key)
RequireAdminKey = Depends(get_admin_api_key)
