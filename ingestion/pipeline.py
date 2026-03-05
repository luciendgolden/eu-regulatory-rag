"""Ingestion pipeline: fetch → parse → chunk → embed → store.

CLI usage:
    python -m ingestion.pipeline --regulation DORA
    python -m ingestion.pipeline --regulation all --force
    python -m ingestion.pipeline --regulation NIS2 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def run_pipeline(
    regulation_id: str,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the full ingestion pipeline for a single regulation.

    Args:
        regulation_id: Regulation key, e.g. "DORA" or "NIS2".
        force:         Re-ingest even if chunks already exist in Qdrant.
        dry_run:       Parse and chunk but do not write to Qdrant.
    """
    from config import settings  # noqa: PLC0415
    from embeddings.service import get_embedding_service  # noqa: PLC0415
    from ingestion.chunker import RegulationChunker  # noqa: PLC0415
    from ingestion.parsers.eurlex_parser import EurLexParser  # noqa: PLC0415
    from ingestion.sources.eurlex_client import EurLexClient  # noqa: PLC0415
    from ingestion.sources.regulations import get_regulation  # noqa: PLC0415
    from vectordb.qdrant_client import QdrantWrapper  # noqa: PLC0415

    meta = get_regulation(regulation_id)
    logger.info("Starting pipeline for %s (%s)", meta.short_name, meta.celex)

    # ------------------------------------------------------------------ #
    # 1. Fetch HTML
    # ------------------------------------------------------------------ #
    client = EurLexClient(
        username=settings.eurlex_username,
        password=settings.eurlex_password,
    )
    logger.info("Fetching %s …", meta.short_name)
    html = client.fetch_html(meta.celex)
    logger.info("Fetched %d bytes of HTML", len(html))

    # ------------------------------------------------------------------ #
    # 2. Parse
    # ------------------------------------------------------------------ #
    parser = EurLexParser(regulation_id=meta.regulation_id, celex_number=meta.celex)
    sections = parser.parse(html)
    logger.info("Parsed %d sections", len(sections))

    if not sections:
        logger.warning("No sections extracted for %s — aborting", regulation_id)
        return

    # ------------------------------------------------------------------ #
    # 3. Chunk
    # ------------------------------------------------------------------ #
    chunker = RegulationChunker()
    chunks = chunker.chunk_sections(sections, topics=meta.topics)
    logger.info("Produced %d chunks", len(chunks))

    if dry_run:
        logger.info("[DRY RUN] Skipping embedding and storage. First chunk preview:")
        if chunks:
            c = chunks[0]
            logger.info("  %s — %s — %d chars", c.regulation, c.section_title, len(c.text))
        return

    # ------------------------------------------------------------------ #
    # 4. Initialise Qdrant
    # ------------------------------------------------------------------ #
    db = QdrantWrapper(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        vector_size=settings.vector_size,
    )
    db.ensure_collection(recreate=False)

    # ------------------------------------------------------------------ #
    # 5. Filter already-indexed chunks (idempotency)
    # ------------------------------------------------------------------ #
    if not force:
        new_chunks = [c for c in chunks if not db.hash_exists(c.content_hash)]
        skipped = len(chunks) - len(new_chunks)
        if skipped:
            logger.info("Skipping %d already-indexed chunks (use --force to re-ingest)", skipped)
        chunks = new_chunks

    if not chunks:
        logger.info("Nothing new to index for %s", regulation_id)
        return

    # ------------------------------------------------------------------ #
    # 6. Embed
    # ------------------------------------------------------------------ #
    emb_service = get_embedding_service()
    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks …", len(chunks))
    vectors = emb_service.embed(texts)

    # ------------------------------------------------------------------ #
    # 7. Store
    # ------------------------------------------------------------------ #
    logger.info("Upserting %d vectors …", len(chunks))
    db.upsert_chunks(chunks, vectors)

    info = db.collection_info()
    logger.info(
        "Done. Collection '%s': %s points total.",
        info["name"],
        info.get("points_count", "?"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="EU Regulatory RAG — ingestion pipeline",
    )
    parser.add_argument(
        "--regulation",
        choices=["DORA", "NIS2", "all"],
        default="all",
        help="Which regulation to ingest (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if chunks already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Parse and chunk but do not write to Qdrant",
    )
    args = parser.parse_args(argv)

    from ingestion.sources.regulations import list_regulations  # noqa: PLC0415

    targets = list_regulations() if args.regulation == "all" else [args.regulation]

    errors = []
    for reg in targets:
        try:
            run_pipeline(reg, force=args.force, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for %s: %s", reg, exc, exc_info=True)
            errors.append(reg)

    if errors:
        logger.error("Failed regulations: %s", errors)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
