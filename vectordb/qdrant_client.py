"""Qdrant vector database wrapper.

Collection: eu_regulations
Payload indexes: regulation, section_type, section_number, celex
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

COLLECTION_NAME = "eu_regulations"
BATCH_SIZE = 100


class QdrantWrapper:
    """High-level Qdrant client for EU regulatory chunks."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        vector_size: int = 384,
    ) -> None:
        self.host = host
        self.port = port
        self.vector_size = vector_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient  # type: ignore

                self._client = QdrantClient(host=self.host, port=self.port)
                logger.info("Connected to Qdrant at %s:%d", self.host, self.port)
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is required. Install with: pip install qdrant-client"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if it does not exist.

        Args:
            recreate: If True, drop and recreate the collection.
        """
        from qdrant_client.http import models as qm  # type: ignore

        client = self._get_client()

        if recreate:
            try:
                client.delete_collection(COLLECTION_NAME)
                logger.info("Dropped existing collection '%s'", COLLECTION_NAME)
            except Exception:  # noqa: BLE001
                pass

        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in existing and not recreate:
            logger.debug("Collection '%s' already exists", COLLECTION_NAME)
            return

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(
                size=self.vector_size,
                distance=qm.Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '%s' (vector_size=%d)", COLLECTION_NAME, self.vector_size
        )

        # Create payload indexes for fast filtering
        for field_name in ("regulation", "section_type", "section_number", "celex"):
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
            logger.debug("Created payload index on '%s'", field_name)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks,  # list[Chunk] — avoid circular import by using duck typing
        vectors: list[list[float]],
    ) -> None:
        """Batch-upsert chunks with their corresponding vectors.

        Args:
            chunks:  List of :class:`~ingestion.chunker.Chunk` objects.
            vectors: Parallel list of float vectors.
        """
        from qdrant_client.http import models as qm  # type: ignore

        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have equal length"
            )

        client = self._get_client()

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]
            batch_vectors = vectors[batch_start : batch_start + BATCH_SIZE]

            points = []
            for chunk, vector in zip(batch_chunks, batch_vectors):
                payload = chunk.to_payload()
                # Use content hash as deterministic point ID (idempotent upsert)
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.content_hash))
                points.append(
                    qm.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
            logger.info(
                "Upserted batch %d–%d / %d",
                batch_start,
                batch_start + len(batch_chunks),
                len(chunks),
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        regulation: Optional[str] = None,
        section_type: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic search with optional metadata filtering.

        Args:
            query_vector: Embedding of the search query.
            regulation:   Filter by regulation ID (e.g. "DORA").
            section_type: Filter by section type (e.g. "article").
            top_k:        Maximum number of results to return.

        Returns:
            List of dicts with keys: id, score, payload.
        """
        from qdrant_client.http import models as qm  # type: ignore

        client = self._get_client()

        # Build filter conditions
        conditions = []
        if regulation:
            conditions.append(
                qm.FieldCondition(
                    key="regulation",
                    match=qm.MatchValue(value=regulation.upper()),
                )
            )
        if section_type:
            conditions.append(
                qm.FieldCondition(
                    key="section_type",
                    match=qm.MatchValue(value=section_type.lower()),
                )
            )

        query_filter = qm.Filter(must=conditions) if conditions else None

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def collection_info(self) -> dict[str, Any]:
        """Return basic collection statistics."""
        client = self._get_client()
        info = client.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def hash_exists(self, content_hash: str) -> bool:
        """Check if a chunk with this content hash already exists (idempotency check)."""
        from qdrant_client.http import models as qm  # type: ignore

        client = self._get_client()
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="content_hash",
                        match=qm.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
        )
        points, _ = results
        return len(points) > 0
