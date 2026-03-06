"""Health check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api.auth import get_api_key
from api.schemas import HealthResponse
from config import settings
from vectordb.qdrant_client import QdrantWrapper

logger = logging.getLogger(__name__)

router = APIRouter()

VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    _key: str = Depends(get_api_key),
) -> HealthResponse:
    """Return service health status.

    Pings Qdrant and reports whether the vector store is reachable.
    Always returns HTTP 200; callers should inspect the ``status`` field.
    Requires a valid API key (or dev mode when API_KEY is not configured).
    """
    qdrant_connected = False
    collections: list[str] = []

    try:
        wrapper = QdrantWrapper(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            vector_size=settings.vector_size,
        )
        col_list = wrapper._get_client().get_collections().collections
        collections = [c.name for c in col_list]
        qdrant_connected = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant health check failed: %s", exc)

    return HealthResponse(
        status="ok" if qdrant_connected else "degraded",
        qdrant_connected=qdrant_connected,
        collections=collections,
        version=VERSION,
    )
