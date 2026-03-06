"""RAG Chain — orchestrates embed → retrieve → prompt → LLM.

Public API
----------
>>> from rag.chain import RAGChain, get_chain, query
>>> chain = RAGChain()
>>> result = chain.query("What are the ICT risk management requirements under DORA?")
>>> print(result["answer"])
>>> for citation in result["citations"]:
...     print(citation)

Streaming variant::

>>> for chunk in chain.query("...", stream=True):
...     print(chunk, end="", flush=True)

Async API (for FastAPI)::

>>> result = await chain.aquery("What is NIS2?")
>>> async for chunk in chain.astream_query("What is NIS2?"):
...     print(chunk)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Iterator, Optional, Union

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
            from rag.retriever import RegulatoryRetriever

            self._retriever = RegulatoryRetriever()
        return self._retriever

    def _get_llm(self):
        if self._llm is None:
            from rag.llm import get_llm_service

            self._llm = get_llm_service()
        return self._llm

    # ------------------------------------------------------------------
    # Sync query (for CLI / scripts)
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

        results = retriever.retrieve(
            query=question,
            regulation=regulation,
            section_type=section_type,
            top_k=top_k,
        )

        citations = _build_citations(results)
        system_prompt, user_message = assemble_prompt(question, results)

        llm = self._get_llm()

        if stream:
            return self._stream_query(llm, system_prompt, user_message, citations)

        answer = llm.complete(system=system_prompt, user=user_message)
        logger.info("RAG answer generated (%d chars, %d citations)", len(answer), len(citations))

        return {"answer": answer, "citations": citations}

    def _stream_query(
        self,
        llm,
        system_prompt: str,
        user_message: str,
        citations: list[dict[str, Any]],
    ) -> Iterator[str]:
        """Yield LLM chunks, then emit citations as a final sentinel."""
        for chunk in llm.stream(system=system_prompt, user=user_message):
            yield chunk
        yield f"\n__citations__:{json.dumps(citations)}"

    # ------------------------------------------------------------------
    # Async query (for FastAPI)
    # ------------------------------------------------------------------

    async def aquery(
        self,
        question: str,
        regulation: Optional[str] = None,
        section_type: Optional[str] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Async variant of :meth:`query` for use in FastAPI endpoints.

        Returns ``{"answer": str, "sources": list[dict], "query_time_ms": int}``
        """
        t0 = time.monotonic()

        retriever = self._get_retriever()
        results = await asyncio.to_thread(
            retriever.retrieve,
            question,
            regulation=regulation,
            section_type=section_type,
            top_k=top_k,
        )

        citations = _build_citations(results)
        system_prompt, user_message = assemble_prompt(question, results)
        llm = self._get_llm()

        answer = await asyncio.to_thread(
            llm.complete, system=system_prompt, user=user_message,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        sources = [
            {
                "regulation": r["regulation"],
                "section_type": r["section_type"],
                "section_number": r.get("section_number"),
                "section_title": r.get("section_title"),
                "score": r["score"],
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "query_time_ms": elapsed_ms,
        }

    async def astream_query(
        self,
        question: str,
        regulation: Optional[str] = None,
        section_type: Optional[str] = None,
        top_k: int = 5,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async streaming variant — yields ``{"chunk": str}`` dicts,
        then a final ``{"done": True, "sources": [...], "query_time_ms": N}``.
        """
        t0 = time.monotonic()

        retriever = self._get_retriever()
        results = await asyncio.to_thread(
            retriever.retrieve,
            question,
            regulation=regulation,
            section_type=section_type,
            top_k=top_k,
        )

        citations = _build_citations(results)
        system_prompt, user_message = assemble_prompt(question, results)
        llm = self._get_llm()

        for chunk in llm.stream(system=system_prompt, user=user_message):
            yield {"chunk": chunk}

        sources = [
            {
                "regulation": r["regulation"],
                "section_type": r["section_type"],
                "section_number": r.get("section_number"),
                "section_title": r.get("section_title"),
                "score": r["score"],
            }
            for r in results
        ]

        yield {
            "done": True,
            "sources": sources,
            "query_time_ms": int((time.monotonic() - t0) * 1000),
        }


# ---------------------------------------------------------------------------
# Helpers
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
# Module-level convenience
# ---------------------------------------------------------------------------

_default_chain: Optional[RAGChain] = None


def get_chain() -> RAGChain:
    """Return the global :class:`RAGChain` instance (lazy init)."""
    global _default_chain
    if _default_chain is None:
        _default_chain = RAGChain()
    return _default_chain


def query(
    question: str,
    regulation: Optional[str] = None,
    section_type: Optional[str] = None,
    top_k: int = 5,
    stream: bool = False,
) -> Union[dict[str, Any], Iterator[str]]:
    """Convenience wrapper using a module-level singleton :class:`RAGChain`."""
    return get_chain().query(
        question,
        regulation=regulation,
        section_type=section_type,
        top_k=top_k,
        stream=stream,
    )


async def aquery(
    question: str,
    regulation: Optional[str] = None,
    section_type: Optional[str] = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Async convenience wrapper."""
    return await get_chain().aquery(
        question,
        regulation=regulation,
        section_type=section_type,
        top_k=top_k,
    )
