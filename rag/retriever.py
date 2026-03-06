"""RAG retriever — semantic search + context assembly with citations."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class RegulatoryRetriever:
    """Retrieve regulation chunks and format them into an LLM-ready context."""

    def __init__(
        self,
        embedding_service=None,
        qdrant_wrapper=None,
    ) -> None:
        """Initialise the retriever.

        If *embedding_service* or *qdrant_wrapper* are not provided, they are
        lazily constructed from ``config.Settings`` on first use.
        """
        self._emb = embedding_service
        self._db = qdrant_wrapper

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_embedding_service(self):
        if self._emb is None:
            from embeddings.service import get_embedding_service  # noqa: PLC0415

            self._emb = get_embedding_service()
        return self._emb

    def _get_db(self):
        if self._db is None:
            from config import settings  # noqa: PLC0415
            from vectordb.qdrant_client import QdrantWrapper  # noqa: PLC0415

            self._db = QdrantWrapper(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                vector_size=settings.vector_size,
            )
        return self._db

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        regulation: Optional[str] = None,
        section_type: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Embed *query* and return the top-k matching chunks.

        Args:
            query:        Natural-language question or keyword.
            regulation:   Restrict results to "DORA" or "NIS2" (optional).
            section_type: Restrict to a section type, e.g. "article" (optional).
            top_k:        Number of results to return.

        Returns:
            List of result dicts with keys:
                - score (float)
                - regulation, celex, section_type, section_number, section_title
                - chapter, topics (list)
                - text (str)
        """
        emb = self._get_embedding_service()
        db = self._get_db()

        logger.debug("Retrieving for query: %r", query)
        query_vector = emb.embed_one(query)

        raw_results = db.search(
            query_vector=query_vector,
            regulation=regulation,
            section_type=section_type,
            top_k=top_k,
        )

        results = []
        for hit in raw_results:
            payload = hit.get("payload", {})
            results.append(
                {
                    "score": hit["score"],
                    "regulation": payload.get("regulation", ""),
                    "celex": payload.get("celex", ""),
                    "section_type": payload.get("section_type", ""),
                    "section_number": payload.get("section_number"),
                    "section_title": payload.get("section_title", ""),
                    "chapter": payload.get("chapter"),
                    "topics": payload.get("topics", []),
                    "text": payload.get("text", ""),
                }
            )

        logger.info("Retrieved %d results for query %r", len(results), query[:80])
        return results

    def format_citation(self, result: dict[str, Any]) -> str:
        """Public helper — format a single retrieval result as a citation string.

        Delegates to the module-level :func:`_build_citation` helper so callers
        do not need to import private functions.

        Example::

            retriever.format_citation(result)
            # → "According to DORA — Article 5 (ICT risk management framework):"
        """
        return _build_citation(result)

    def build_context(self, results: list[dict[str, Any]]) -> str:
        """Format retrieved results into a structured context string.

        Citations follow the pattern:
            "According to DORA Article 5 (Digital operational resilience strategy):"

        Args:
            results: Output of :meth:`retrieve`.

        Returns:
            Multi-section context string ready for insertion into an LLM prompt.
        """
        if not results:
            return "No relevant regulatory text found for this query."

        sections: list[str] = []
        for i, res in enumerate(results, start=1):
            citation = _build_citation(res)
            text = res.get("text", "").strip()
            sections.append(f"[{i}] {citation}\n{text}")

        return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_citation(result: dict[str, Any]) -> str:
    """Build a human-readable citation string from a result dict."""
    regulation = result.get("regulation", "")
    section_type = result.get("section_type", "").capitalize()
    section_number = result.get("section_number")
    section_title = result.get("section_title", "")
    chapter = result.get("chapter")

    parts = [f"According to {regulation}"]

    if chapter:
        parts.append(f"Chapter {chapter}")

    if section_type and section_number:
        parts.append(f"{section_type} {section_number}")
    elif section_type:
        parts.append(section_type)

    if section_title:
        # Avoid repeating "Article 5" in the title
        clean_title = section_title
        if section_number and section_number in clean_title:
            import re

            clean_title = re.sub(
                rf"\b{re.escape(section_type)}\s+{re.escape(section_number)}\b",
                "",
                clean_title,
                flags=re.IGNORECASE,
            ).strip(" —-:")
        if clean_title:
            parts[-1] = f"{parts[-1]} ({clean_title})"

    return " — ".join(parts) + ":"
