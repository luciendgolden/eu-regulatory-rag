"""RAG chain — orchestrates retriever + LLM to answer regulatory questions.

Public API
----------
>>> from rag.chain import get_chain, query
>>> result = await query("What are the ICT risk management requirements under DORA?")
>>> print(result["answer"])
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional

from rag.retriever import RegulatoryRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a regulatory compliance expert specialising in EU financial and
cybersecurity regulation (DORA, NIS2). Answer the user's question using ONLY the provided
regulatory context. Cite specific articles or recitals. If the context does not contain
enough information, say so clearly.

Regulatory context:
{context}"""

_USER_TEMPLATE = "Question: {question}"


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------


class RegulatoryChain:
    """End-to-end RAG chain: retrieve → prompt → generate."""

    def __init__(
        self,
        retriever: Optional[RegulatoryRetriever] = None,
    ) -> None:
        self._retriever = retriever or RegulatoryRetriever()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        question: str,
        regulation: Optional[str] = None,
        section_type: Optional[str] = None,
        top_k: int = 5,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Answer *question* and return a structured result dict.

        Returns::
            {
                "answer": str,
                "sources": list[dict],
                "query_time_ms": int,
            }
        """
        t0 = time.monotonic()

        results = await asyncio.to_thread(
            self._retriever.retrieve,
            question,
            regulation=regulation,
            section_type=section_type,
            top_k=top_k,
        )

        context = self._retriever.build_context(results)
        answer = await asyncio.to_thread(self._generate, question, context)
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

    async def stream_query(
        self,
        question: str,
        regulation: Optional[str] = None,
        section_type: Optional[str] = None,
        top_k: int = 5,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream LLM response chunks as an async generator.

        Yields dicts of shape ``{"chunk": str}`` followed by a final
        ``{"done": True, "sources": [...], "query_time_ms": N}`` event.
        """
        t0 = time.monotonic()

        results = await asyncio.to_thread(
            self._retriever.retrieve,
            question,
            regulation=regulation,
            section_type=section_type,
            top_k=top_k,
        )
        context = self._retriever.build_context(results)

        async for chunk in self._stream_generate(question, context):
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

        yield {"done": True, "sources": sources, "query_time_ms": int((time.monotonic() - t0) * 1000)}

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _generate(self, question: str, context: str) -> str:
        try:
            return self._openai_generate(question, context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI generation failed (%s); using stub answer.", exc)
            return self._stub_answer(question, context)

    async def _stream_generate(self, question: str, context: str) -> AsyncIterator[str]:
        try:
            async for chunk in self._openai_stream(question, context):
                yield chunk
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI streaming failed (%s); falling back to stub.", exc)
            for word in self._stub_answer(question, context).split():
                yield word + " "
                await asyncio.sleep(0)

    def _openai_generate(self, question: str, context: str) -> str:
        from config import settings

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": _USER_TEMPLATE.format(question=question)},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    async def _openai_stream(self, question: str, context: str) -> AsyncIterator[str]:
        from config import settings

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")

        from openai import AsyncOpenAI  # type: ignore

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": _USER_TEMPLATE.format(question=question)},
            ],
            temperature=0.2,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def _stub_answer(question: str, context: str) -> str:
        if "No relevant" in context:
            return (
                f"I could not find relevant regulatory text to answer: '{question}'. "
                "Please try a more specific query or ensure the regulations have been ingested."
            )
        return (
            f"[STUB] Based on the retrieved regulatory context, here is a synthesised "
            f"answer to '{question}'. In production this response is generated by an LLM "
            f"using the retrieved DORA/NIS2 articles as grounding."
        )


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions
# ---------------------------------------------------------------------------

_chain: Optional[RegulatoryChain] = None


def get_chain() -> RegulatoryChain:
    """Return the global :class:`RegulatoryChain` instance (lazy init)."""
    global _chain  # noqa: PLW0603
    if _chain is None:
        _chain = RegulatoryChain()
    return _chain


async def query(
    question: str,
    regulation: Optional[str] = None,
    section_type: Optional[str] = None,
    top_k: int = 5,
    stream: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper around the global chain's :meth:`~RegulatoryChain.query`."""
    return await get_chain().query(
        question=question,
        regulation=regulation,
        section_type=section_type,
        top_k=top_k,
        stream=stream,
    )
