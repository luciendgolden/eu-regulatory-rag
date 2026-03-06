"""Ingest endpoint — trigger re-ingestion of regulations (admin-protected)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from api.auth import get_admin_api_key
from api.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def _run_ingestion(regulation: str, force: bool) -> None:
    """Run the ingestion pipeline in a background task."""
    logger.info("Starting ingestion: regulation=%r, force=%s", regulation, force)
    try:
        from ingestion.pipeline import run_pipeline  # type: ignore

        await asyncio.to_thread(run_pipeline, regulation=regulation, force=force)
        logger.info("Ingestion complete for regulation=%r", regulation)
    except ImportError:
        logger.warning(
            "ingestion.pipeline.run_pipeline not available — skipping ingestion."
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingestion failed for regulation=%r: %s", regulation, exc)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Admin"],
    summary="Trigger re-ingestion (admin)",
    description=(
        "Trigger re-ingestion of DORA, NIS2, or all regulations. "
        "Requires admin API key. Ingestion runs in the background."
    ),
)
async def trigger_ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    _key: str = Depends(get_admin_api_key),
) -> IngestResponse:
    """Queue an ingestion run and return immediately.

    The ingestion pipeline runs asynchronously in a background task.
    Poll ``GET /api/v1/health`` and ``GET /api/v1/regulations`` to track
    progress via article counts.
    """
    regulation = body.regulation.value
    logger.info(
        "Ingest requested: regulation=%r, force=%s", regulation, body.force
    )
    background_tasks.add_task(_run_ingestion, regulation=regulation, force=body.force)

    return IngestResponse(
        status="accepted",
        message=(
            f"Ingestion of '{regulation}' has been queued. "
            "Check /api/v1/regulations for updated article counts."
        ),
        regulation=regulation,
    )
