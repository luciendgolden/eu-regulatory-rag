"""FastAPI application — EU Regulatory RAG API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.errors import register_error_handlers
from api.routes import health, ingest, query, regulations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared services on startup; clean up on shutdown."""
    logger.info("Starting EU Regulatory RAG API…")

    # Pre-warm embedding service
    try:
        from embeddings.service import get_embedding_service

        svc = get_embedding_service()
        logger.info("Embedding service ready (vector_size=%d)", svc.vector_size)
        app.state.embedding_service = svc
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not initialise embedding service: %s", exc)
        app.state.embedding_service = None

    # Pre-warm Qdrant connection
    try:
        from config import settings
        from vectordb.qdrant_client import QdrantWrapper

        wrapper = QdrantWrapper(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            vector_size=settings.vector_size,
        )
        # Just verify the connection — don't recreate collections
        wrapper._get_client().get_collections()
        logger.info("Qdrant connection established at %s:%d", settings.qdrant_host, settings.qdrant_port)
        app.state.qdrant = wrapper
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not connect to Qdrant on startup: %s", exc)
        app.state.qdrant = None

    yield

    # Shutdown
    logger.info("Shutting down EU Regulatory RAG API.")
    app.state.embedding_service = None
    app.state.qdrant = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    from config import settings  # local import to allow settings override in tests

    app = FastAPI(
        title="EU Regulatory RAG API",
        description=(
            "Retrieval-Augmented Generation API for EU financial and cybersecurity "
            "regulations (DORA & NIS2). Submit natural-language compliance questions "
            "and receive grounded answers with citations."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------
    register_error_handlers(app)

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)
    app.include_router(regulations.router, prefix=prefix)
    app.include_router(ingest.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "EU Regulatory RAG API — see /docs for usage."}

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn: api.app:app)
# ---------------------------------------------------------------------------

app = create_app()
