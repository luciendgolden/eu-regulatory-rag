"""Custom exception handlers for consistent error responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception types
# ---------------------------------------------------------------------------


class QdrantConnectionError(Exception):
    """Raised when the Qdrant vector database is unreachable."""


class LLMError(Exception):
    """Raised when the language model fails to generate a response."""


class RegulatoryRAGError(Exception):
    """Generic application-level error."""


# ---------------------------------------------------------------------------
# Error response helper
# ---------------------------------------------------------------------------


def _error_response(
    status_code: int,
    error: str,
    detail: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail,
            "status_code": status_code,
        },
    )


# ---------------------------------------------------------------------------
# Register handlers
# ---------------------------------------------------------------------------


def register_error_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to *app*."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Validation error on %s: %s", request.url, exc.errors())
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="ValidationError",
            detail=str(exc.errors()),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        logger.warning("Pydantic validation error: %s", exc)
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="ValidationError",
            detail=str(exc.errors()),
        )

    @app.exception_handler(QdrantConnectionError)
    async def qdrant_error_handler(
        request: Request, exc: QdrantConnectionError
    ) -> JSONResponse:
        logger.error("Qdrant connection error: %s", exc)
        return _error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="QdrantConnectionError",
            detail=str(exc) or "Vector database is unavailable.",
        )

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM error: %s", exc)
        return _error_response(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error="LLMError",
            detail=str(exc) or "Language model failed to generate a response.",
        )

    @app.exception_handler(RegulatoryRAGError)
    async def rag_error_handler(
        request: Request, exc: RegulatoryRAGError
    ) -> JSONResponse:
        logger.error("RAG error: %s", exc)
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="RegulatoryRAGError",
            detail=str(exc) or "An internal error occurred.",
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s", request.url)
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="InternalServerError",
            detail="An unexpected error occurred.",
        )
