"""Embedding service — local (sentence-transformers) or OpenAI.

Configured via EMBEDDING_PROVIDER env var:
  - "local"  → all-MiniLM-L6-v2  (384 dims) via sentence-transformers
  - "openai" → text-embedding-3-small (1536 dims) via openai
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------


class EmbeddingService(Protocol):
    """Protocol satisfied by both local and OpenAI embedding backends."""

    @property
    def vector_size(self) -> int:
        """Dimension of produced vectors."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return a list of float vectors."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single string."""
        ...


# ---------------------------------------------------------------------------
# Local backend (sentence-transformers)
# ---------------------------------------------------------------------------


class LocalEmbeddingService:
    """Sentence-Transformers local embedding backend."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                logger.info("Loading sentence-transformer model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        return self._model

    @property
    def vector_size(self) -> int:
        return 384  # all-MiniLM-L6-v2

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        logger.debug("Embedding %d texts with %s", len(texts), self.model_name)
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


class OpenAIEmbeddingService:
    """OpenAI text-embedding-3-small backend."""

    OPENAI_MODEL = "text-embedding-3-small"
    BATCH_SIZE = 100

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self.api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for OpenAI embeddings. "
                    "Install it with: pip install openai"
                ) from exc
        return self._client

    @property
    def vector_size(self) -> int:
        return 1536  # text-embedding-3-small

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            logger.debug(
                "Embedding OpenAI batch %d–%d / %d",
                i,
                i + len(batch),
                len(texts),
            )
            response = client.embeddings.create(input=batch, model=self.OPENAI_MODEL)
            all_vectors.extend([item.embedding for item in response.data])

        return all_vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedding_service(
    provider: str | None = None,
    model: str = "all-MiniLM-L6-v2",
    openai_api_key: str = "",
) -> EmbeddingService:
    """Return the configured embedding service.

    Reads EMBEDDING_PROVIDER from config if *provider* is not passed explicitly.
    """
    if provider is None:
        try:
            from config import settings  # type: ignore

            provider = settings.embedding_provider
            model = settings.embedding_model
            openai_api_key = openai_api_key or settings.openai_api_key
        except Exception:  # noqa: BLE001
            provider = "local"

    provider = (provider or "local").lower()

    if provider == "openai":
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when using the OpenAI embedding provider."
            )
        logger.info("Using OpenAI embedding service (text-embedding-3-small)")
        return OpenAIEmbeddingService(api_key=openai_api_key)

    logger.info("Using local embedding service (%s)", model)
    return LocalEmbeddingService(model_name=model)
