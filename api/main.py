"""FastAPI application exposing retrieval endpoints for the EU Regulatory RAG system."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.retriever import RegulatoryRetriever

logger = logging.getLogger(__name__)
RETRIEVAL_FAILED_DETAIL = "Retrieval failed."

app = FastAPI(
    title="EU Regulatory RAG API",
    version="0.1.0",
    description="REST API for health checks and regulation retrieval.",
)


class RetrieveRequest(BaseModel):
    """Request payload for semantic retrieval."""

    query: str = Field(..., min_length=1, description="User question or search query")
    regulation: Optional[str] = Field(default=None, description="Optional regulation filter")
    section_type: Optional[str] = Field(default=None, description="Optional section type filter")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of results")


class RetrievalResult(BaseModel):
    """Single retrieved chunk returned by the API."""

    score: float
    regulation: str
    celex: str
    section_type: str
    section_number: Optional[str] = None
    section_title: str
    chapter: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    text: str


class RetrieveResponse(BaseModel):
    """Response payload for semantic retrieval."""

    query: str
    results: list[RetrievalResult]
    context: str


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str
    service: str
    version: str


class RootResponse(BaseModel):
    """Basic service metadata."""

    service: str
    version: str
    endpoints: dict[str, str]


@lru_cache
def get_retriever() -> RegulatoryRetriever:
    """Return the shared default retriever instance."""
    return RegulatoryRetriever()


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    """Return basic API metadata."""
    return RootResponse(
        service="eu-regulatory-rag",
        version=app.version,
        endpoints={
            "health": "/health",
            "retrieve": "/retrieve",
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness endpoint for local development and containers."""
    return HealthResponse(
        status="ok",
        service="eu-regulatory-rag",
        version=app.version,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: RetrieveRequest,
    retriever: RegulatoryRetriever = Depends(get_retriever),
) -> RetrieveResponse:
    """Retrieve regulation chunks and assemble an LLM-ready context."""
    try:
        results: list[dict[str, Any]] = retriever.retrieve(
            query=request.query,
            regulation=request.regulation,
            section_type=request.section_type,
            top_k=request.top_k,
        )
        context = retriever.build_context(results)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=RETRIEVAL_FAILED_DETAIL) from exc

    return RetrieveResponse(query=request.query, results=results, context=context)
