"""Regulations listing endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api.auth import get_api_key
from api.schemas import RegulationInfo
from config import settings
from vectordb.qdrant_client import COLLECTION_NAME, QdrantWrapper

logger = logging.getLogger(__name__)

router = APIRouter()

# Static metadata for known regulations — augmented with live Qdrant counts
_KNOWN_REGULATIONS: dict[str, dict] = {
    "DORA": {
        "title": "Digital Operational Resilience Act (DORA)",
        "celex": "32022R2554",
    },
    "NIS2": {
        "title": "Network and Information Security Directive 2 (NIS2)",
        "celex": "32022L2555",
    },
}


@router.get(
    "/regulations",
    response_model=list[RegulationInfo],
    tags=["Regulations"],
    summary="List indexed regulations",
)
async def list_regulations(
    _key: str = Depends(get_api_key),
) -> list[RegulationInfo]:
    """Return metadata and article counts for all indexed regulations.

    Article counts are derived from Qdrant collection statistics.
    If Qdrant is unavailable, article counts default to 0.
    """
    # Attempt to fetch per-regulation counts from Qdrant
    counts: dict[str, int] = {reg: 0 for reg in _KNOWN_REGULATIONS}

    try:
        from qdrant_client.http import models as qm  # type: ignore

        wrapper = QdrantWrapper(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            vector_size=settings.vector_size,
        )
        client = wrapper._get_client()

        for reg_id in _KNOWN_REGULATIONS:
            result = client.count(
                collection_name=COLLECTION_NAME,
                count_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="regulation",
                            match=qm.MatchValue(value=reg_id),
                        )
                    ]
                ),
                exact=True,
            )
            counts[reg_id] = result.count
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch regulation counts from Qdrant: %s", exc)

    return [
        RegulationInfo(
            id=reg_id,
            title=meta["title"],
            celex=meta["celex"],
            article_count=counts[reg_id],
            last_indexed=None,  # TODO: persist last-ingested timestamp
        )
        for reg_id, meta in _KNOWN_REGULATIONS.items()
    ]
