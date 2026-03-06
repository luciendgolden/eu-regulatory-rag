"""Query endpoint — non-streaming JSON and SSE streaming."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

import rag.chain as chain_module
from rag.chain import get_chain
from api.auth import get_api_key
from api.schemas import QueryRequest, QueryResponse, SourceCitation

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_query_response(result: dict[str, Any]) -> QueryResponse:
    sources = [
        SourceCitation(
            regulation=s["regulation"],
            section_type=s["section_type"],
            section_number=s.get("section_number"),
            section_title=s.get("section_title"),
            score=s["score"],
        )
        for s in result.get("sources", [])
    ]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        query_time_ms=result["query_time_ms"],
    )


async def _sse_generator(
    question: str,
    regulation: str | None,
    section_type: str | None,
    top_k: int,
) -> AsyncIterator[str]:
    """Yield SSE-formatted events from the streaming chain."""
    chain = chain_module.get_chain()
    async for event in chain.stream_query(
        question=question,
        regulation=regulation,
        section_type=section_type,
        top_k=top_k,
    ):
        data = json.dumps(event, ensure_ascii=False)
        yield f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Submit a regulatory question",
    description=(
        "Ask a question about DORA or NIS2. "
        "Set `stream=true` to receive a Server-Sent Events stream."
    ),
)
async def query_endpoint(
    body: QueryRequest,
    _key: str = Depends(get_api_key),
) -> QueryResponse | StreamingResponse:
    """Handle a regulatory query.

    * **stream=false** (default): returns a full ``QueryResponse`` JSON body.
    * **stream=true**: returns an SSE stream where each event is a JSON object.
      Chunk events have shape ``{"chunk": "..."}``; the final event has shape
      ``{"done": true, "sources": [...], "query_time_ms": N}``.
    """
    regulation = body.regulation.value if body.regulation else None

    if body.stream:
        logger.info("Streaming query: %r", body.question[:80])
        generator = _sse_generator(
            question=body.question,
            regulation=regulation,
            section_type=body.section_type,
            top_k=body.top_k,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    logger.info("Non-streaming query: %r", body.question[:80])
    result = await chain_module.query(
        question=body.question,
        regulation=regulation,
        section_type=body.section_type,
        top_k=body.top_k,
        stream=False,
    )
    return _build_query_response(result)
