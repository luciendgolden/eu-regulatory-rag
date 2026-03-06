"""RAG Chain — orchestrates embed → retrieve → prompt → LLM.

Public API
----------
>>> from rag.chain import RAGChain
>>> chain = RAGChain()
>>> result = chain.query("What are the ICT risk management requirements under DORA?")
>>> print(result["answer"])
>>> for citation in result["citations"]:
...     print(citation)

Streaming variant::

>>> for chunk in chain.query("...", stream=True):
...     print(chunk, end="", flush=True)
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Union

from rag.prompt import assemble_prompt, format_citation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAGChain
# ---------------------------------------------------------------------------


class RAGChain:
    """End-to-end RAG pipeline for EU regulatory queries.

    Lazy-initialises the retriever and LLM service on first use so the class
    can be imported without triggering network connections or heavy model loads.
    """

    def __init__(
        self,
        retriever=None,
        llm_service=None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_service

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_retriever(self):
        if self._retriever is None:
            from rag.retriever import RegulatoryRetriever  # noqa: PLC0415

            self._retriever = RegulatoryRetriever()
        return self._retriever

    def _get_llm(self):
        if self._llm is None:
            from rag.llm import get_llm_service  # noqa: PLC0415

            self._llm = get_llm_service()
        return self._llm

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        regulation: Optional[str] = None,
        section_type: Optional[str] = None,
        top_k: int = 5,
        stream: bool = False,
    ) -> Union[dict[str, Any], Iterator[str]]:
        """Run the full RAG pipeline for *question*.

        Args:
            question:     The user's natural-language question.
            regulation:   Optional filter (e.g. ``"DORA"`` or ``"NIS2"``).
            section_type: Optional filter (e.g. ``"article"``).
            top_k:        Number of context chunks to retrieve.
            stream:       If ``True``, return a generator that yields answer
                          chunks. The final yielded item is a special
                          ``"__citations__"`` sentinel followed by JSON-encoded
                          citations — callers that want citations in streaming
                          mode should filter for it.

        Returns:
            * **Non-streaming** — ``{"answer": str, "citations": list[dict]}``
            * **Streaming** — ``Iterator[str]`` of text chunks
        """
        retriever = self._get_retriever()

        logger.info(
            "RAG query: %r (top_k=%d, regulation=%s)",
            question[:80],
            top_k,
            regulation,
        )

        # 1. Retrieve relevant chunks
        results = retriever.retrieve(
            query=question,
            regulation=regulation,
            section_type=section_type,
            top_k=top_k,
        )

        # 2. Build citations list
        citations = _build_citations(results)

        # 3. Assemble prompt
        system_prompt, user_message = assemble_prompt(question, results)

        llm = self._get_llm()

        if stream:
            return self._stream_query(llm, system_prompt, user_message, citations)

        # 4. Call LLM (blocking)
        answer = llm.complete(system=system_prompt, user=user_message)
        logger.info("RAG answer generated (%d chars, %d citations)", len(answer), len(citations))

        return {"answer": answer, "citations": citations}

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    def _stream_query(
        self,
        llm,
        system_prompt: str,
        user_message: str,
        citations: list[dict[str, Any]],
    ) -> Iterator[str]:
        """Yield LLM chunks, then emit citations as a final sentinel."""
        import json

        for chunk in llm.stream(system=system_prompt, user=user_message):
            yield chunk

        # Yield citations as a structured sentinel for callers that need them
        yield f"\n__citations__:{json.dumps(citations)}"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract citation metadata from retrieval results."""
    citations = []
    for res in results:
        citations.append(
            {
                "regulation": res.get("regulation", ""),
                "article": _article_label(res),
                "score": round(res.get("score", 0.0), 4),
                "citation": format_citation(res),
            }
        )
    return citations


def _article_label(result: dict[str, Any]) -> str:
    """Return a compact article label like 'Article 5' or 'Recital 12'."""
    section_type = (result.get("section_type") or "").capitalize()
    section_number = result.get("section_number")
    if section_type and section_number is not None:
        return f"{section_type} {section_number}"
    return section_type or ""


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_chain: Optional[RAGChain] = None


def query(
    question: str,
    regulation: Optional[str] = None,
    section_type: Optional[str] = None,
    top_k: int = 5,
    stream: bool = False,
) -> Union[dict[str, Any], Iterator[str]]:
    """Convenience wrapper using a module-level singleton :class:`RAGChain`."""
    global _default_chain
    if _default_chain is None:
        _default_chain = RAGChain()
    return _default_chain.query(
        question,
        regulation=regulation,
        section_type=section_type,
        top_k=top_k,
        stream=stream,
    )
