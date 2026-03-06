"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RegulationEnum(str, Enum):
    DORA = "DORA"
    NIS2 = "NIS2"
    ALL = "all"


class RegulationQueryEnum(str, Enum):
    DORA = "DORA"
    NIS2 = "NIS2"


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""

    question: str = Field(..., min_length=1, description="The regulatory question to answer")
    regulation: Optional[RegulationQueryEnum] = Field(
        default=None, description="Restrict search to DORA or NIS2"
    )
    section_type: Optional[str] = Field(
        default=None, description="Restrict to section type (e.g. 'article', 'recital')"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    stream: bool = Field(default=False, description="Enable SSE streaming of LLM response")


class SourceCitation(BaseModel):
    """A single source citation included in query responses."""

    regulation: str
    section_type: str
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    score: float


class QueryResponse(BaseModel):
    """Response schema for non-streaming /query requests."""

    answer: str
    sources: list[SourceCitation]
    query_time_ms: int


# ---------------------------------------------------------------------------
# Regulations
# ---------------------------------------------------------------------------


class RegulationInfo(BaseModel):
    """Summary info for an indexed regulation."""

    id: str
    title: str
    celex: str
    article_count: int
    last_indexed: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request schema for the /ingest endpoint."""

    regulation: RegulationEnum = Field(
        default=RegulationEnum.ALL,
        description="Which regulation to ingest: DORA, NIS2, or all",
    )
    force: bool = Field(
        default=False,
        description="Force re-ingestion even if chunks already exist",
    )


class IngestResponse(BaseModel):
    """Acknowledgement that ingestion has been triggered."""

    status: str
    message: str
    regulation: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "ok" | "degraded" | "error"
    qdrant_connected: bool
    collections: list[str]
    version: str


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Consistent error response format."""

    error: str
    detail: str
    status_code: int
