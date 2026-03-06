"""Semantic chunker for EUR-Lex regulation sections.

Strategy:
- Articles ≤ 512 tokens → kept whole as a single chunk.
- Articles > 512 tokens → split at paragraph boundaries.
- Each chunk carries rich metadata for downstream filtering.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting (tiktoken)
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:
    logger.warning("tiktoken not installed — falling back to word-count approximation")

    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text.split())


MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single text chunk ready for embedding and vector storage."""

    text: str
    regulation: str
    celex: str
    section_type: str
    section_number: Optional[str]
    section_title: str
    chapter: Optional[str]
    chunk_index: int
    total_chunks: int
    topics: list[str] = field(default_factory=list)
    content_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()

    def to_payload(self) -> dict:
        """Return a flat dict suitable as Qdrant point payload."""
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class RegulationChunker:
    """Splits :class:`~ingestion.parsers.eurlex_parser.Section` objects into :class:`Chunk` objects."""

    def __init__(self, max_tokens: int = MAX_TOKENS) -> None:
        self.max_tokens = max_tokens

    def chunk_sections(self, sections, topics: list[str] | None = None) -> list[Chunk]:
        """Chunk all sections from a regulation.

        Args:
            sections: Iterable of :class:`Section` objects.
            topics:   Topic tags to attach to every chunk.

        Returns:
            List of :class:`Chunk` objects.
        """
        all_chunks: list[Chunk] = []
        for section in sections:
            chunks = self._chunk_section(section, topics or [])
            all_chunks.extend(chunks)
        logger.info("Produced %d chunks from %d sections", len(all_chunks), len(list))
        return all_chunks

    def _chunk_section(self, section, topics: list[str]) -> list[Chunk]:
        """Return one or more chunks for a single section."""
        text = section.text.strip()
        if not text:
            return []

        token_count = _count_tokens(text)

        if token_count <= self.max_tokens:
            raw_chunks = [text]
        else:
            raw_chunks = self._split_by_paragraphs(text)

        total = len(raw_chunks)
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    regulation=section.regulation_id,
                    celex=section.celex_number,
                    section_type=section.section_type,
                    section_number=section.section_number,
                    section_title=section.title,
                    chapter=section.parent_section,
                    chunk_index=idx,
                    total_chunks=total,
                    topics=list(topics),
                )
            )
        return chunks

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split *text* into chunks of ≤ max_tokens at paragraph boundaries.

        Paragraphs are separated by blank lines.  If a single paragraph
        exceeds the limit, it is further split at sentence boundaries.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para)

            if para_tokens > self.max_tokens:
                # Flush current buffer first
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                # Split the oversized paragraph at sentence boundaries
                chunks.extend(self._split_by_sentences(para))
                continue

            if current_tokens + para_tokens > self.max_tokens and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks if chunks else [text]

    def _split_by_sentences(self, text: str) -> list[str]:
        """Naïve sentence splitter used as last resort."""
        import re

        sentence_re = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_re.split(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = _count_tokens(sentence)
            if current_tokens + s_tokens > self.max_tokens and current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]
