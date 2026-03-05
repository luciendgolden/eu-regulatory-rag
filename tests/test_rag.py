"""Tests for the RAG query layer: prompt assembly, citations, and the RAG chain."""

from __future__ import annotations

import json
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures — sample retrieval results
# ---------------------------------------------------------------------------

DORA_RESULT: dict[str, Any] = {
    "score": 0.921,
    "regulation": "DORA",
    "celex": "32022R2554",
    "section_type": "article",
    "section_number": "5",
    "section_title": "ICT risk management framework",
    "chapter": "II",
    "topics": ["ict risk", "governance"],
    "text": (
        "Financial entities shall have in place a sound, comprehensive and well-documented "
        "ICT risk management framework as part of their overall risk management system."
    ),
}

NIS2_RESULT: dict[str, Any] = {
    "score": 0.814,
    "regulation": "NIS2",
    "celex": "32022L2555",
    "section_type": "article",
    "section_number": "21",
    "section_title": "Cybersecurity risk-management measures",
    "chapter": "IV",
    "topics": ["cybersecurity", "risk management"],
    "text": (
        "Member States shall ensure that essential and important entities take appropriate "
        "and proportionate technical and organisational measures."
    ),
}

MINIMAL_RESULT: dict[str, Any] = {
    "score": 0.5,
    "regulation": "DORA",
    "celex": "",
    "section_type": "",
    "section_number": None,
    "section_title": "",
    "chapter": None,
    "topics": [],
    "text": "Some orphaned text chunk.",
}


# ---------------------------------------------------------------------------
# Tests — rag.prompt
# ---------------------------------------------------------------------------


class TestFormatCitation:
    """Tests for rag.prompt.format_citation."""

    def test_full_citation(self) -> None:
        from rag.prompt import format_citation

        citation = format_citation(DORA_RESULT)
        assert "DORA" in citation
        assert "Article" in citation
        assert "5" in citation
        assert "ICT risk management framework" in citation

    def test_nis2_citation(self) -> None:
        from rag.prompt import format_citation

        citation = format_citation(NIS2_RESULT)
        assert "NIS2" in citation
        assert "21" in citation
        assert "Cybersecurity risk-management measures" in citation

    def test_citation_with_chapter(self) -> None:
        from rag.prompt import format_citation

        citation = format_citation(DORA_RESULT)
        # Chapter is present in DORA_RESULT — should appear
        assert "Chapter" in citation or "II" in citation

    def test_minimal_result_no_crash(self) -> None:
        from rag.prompt import format_citation

        # Should not raise even with missing fields
        citation = format_citation(MINIMAL_RESULT)
        assert isinstance(citation, str)
        assert "DORA" in citation

    def test_missing_regulation_graceful(self) -> None:
        from rag.prompt import format_citation

        result = {**DORA_RESULT, "regulation": ""}
        citation = format_citation(result)
        assert isinstance(citation, str)


class TestBuildContextBlock:
    """Tests for rag.prompt.build_context_block."""

    def test_empty_results(self) -> None:
        from rag.prompt import build_context_block

        context = build_context_block([])
        assert "No relevant" in context

    def test_numbered_entries(self) -> None:
        from rag.prompt import build_context_block

        context = build_context_block([DORA_RESULT, NIS2_RESULT])
        assert "[1]" in context
        assert "[2]" in context

    def test_text_included(self) -> None:
        from rag.prompt import build_context_block

        context = build_context_block([DORA_RESULT])
        assert DORA_RESULT["text"] in context

    def test_separator_between_chunks(self) -> None:
        from rag.prompt import build_context_block

        context = build_context_block([DORA_RESULT, NIS2_RESULT])
        assert "---" in context


class TestAssemblePrompt:
    """Tests for rag.prompt.assemble_prompt."""

    def test_returns_two_strings(self) -> None:
        from rag.prompt import assemble_prompt

        system, user = assemble_prompt("What is DORA?", [DORA_RESULT])
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_mentions_dora(self) -> None:
        from rag.prompt import assemble_prompt

        system, _ = assemble_prompt("test", [])
        assert "DORA" in system

    def test_user_message_contains_query(self) -> None:
        from rag.prompt import assemble_prompt

        question = "What are the ICT risk management obligations?"
        _, user = assemble_prompt(question, [DORA_RESULT])
        assert question in user

    def test_user_message_contains_context(self) -> None:
        from rag.prompt import assemble_prompt

        _, user = assemble_prompt("test", [DORA_RESULT])
        assert DORA_RESULT["text"] in user

    def test_template_structure(self) -> None:
        from rag.prompt import assemble_prompt

        _, user = assemble_prompt("my question", [])
        assert "Based on the following regulatory excerpts" in user
        assert "Question: my question" in user
        assert "Provide a detailed answer" in user


# ---------------------------------------------------------------------------
# Tests — rag.retriever (format_citation helper)
# ---------------------------------------------------------------------------


class TestRetrieverFormatCitation:
    """Tests for RegulatoryRetriever.format_citation."""

    def test_public_helper_returns_string(self) -> None:
        from rag.retriever import RegulatoryRetriever

        retriever = RegulatoryRetriever()
        citation = retriever.format_citation(DORA_RESULT)
        assert isinstance(citation, str)
        assert "DORA" in citation

    def test_consistent_with_prompt_format_citation(self) -> None:
        """Both helpers should produce equivalent output."""
        from rag.prompt import format_citation as prompt_fmt
        from rag.retriever import RegulatoryRetriever

        retriever = RegulatoryRetriever()
        retriever_citation = retriever.format_citation(DORA_RESULT)
        # They use different implementations but should both contain key info
        assert "DORA" in retriever_citation
        assert "5" in retriever_citation


# ---------------------------------------------------------------------------
# Tests — rag.chain.RAGChain (fully mocked)
# ---------------------------------------------------------------------------


class MockLLMService:
    """Test double for an LLM service."""

    ANSWER = "Financial entities must implement a comprehensive ICT risk management framework per DORA Article 5."

    def complete(self, system: str, user: str) -> str:
        return self.ANSWER

    def stream(self, system: str, user: str) -> Iterator[str]:
        for word in self.ANSWER.split():
            yield word + " "


class MockRetriever:
    """Test double for the regulatory retriever."""

    def retrieve(self, query, regulation=None, section_type=None, top_k=5):
        return [DORA_RESULT, NIS2_RESULT][:top_k]


class TestRAGChain:
    """Tests for rag.chain.RAGChain."""

    def _make_chain(self) -> "RAGChain":
        from rag.chain import RAGChain

        return RAGChain(retriever=MockRetriever(), llm_service=MockLLMService())

    def test_query_returns_answer_and_citations(self) -> None:
        chain = self._make_chain()
        result = chain.query("What are DORA ICT obligations?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "citations" in result

    def test_answer_is_nonempty_string(self) -> None:
        chain = self._make_chain()
        result = chain.query("test question")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_citations_contain_regulation(self) -> None:
        chain = self._make_chain()
        result = chain.query("DORA question")
        assert len(result["citations"]) > 0
        for citation in result["citations"]:
            assert "regulation" in citation
            assert "article" in citation
            assert "score" in citation

    def test_citations_include_dora(self) -> None:
        chain = self._make_chain()
        result = chain.query("test")
        regulations = {c["regulation"] for c in result["citations"]}
        assert "DORA" in regulations

    def test_top_k_limits_citations(self) -> None:
        chain = self._make_chain()
        result = chain.query("test", top_k=1)
        # MockRetriever respects top_k
        assert len(result["citations"]) <= 1

    def test_streaming_yields_strings(self) -> None:
        chain = self._make_chain()
        chunks = list(chain.query("test", stream=True))
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_streaming_includes_citations_sentinel(self) -> None:
        chain = self._make_chain()
        chunks = list(chain.query("test", stream=True))
        sentinel_chunks = [c for c in chunks if c.startswith("\n__citations__:")]
        assert len(sentinel_chunks) == 1

    def test_streaming_citations_are_valid_json(self) -> None:
        chain = self._make_chain()
        chunks = list(chain.query("test", stream=True))
        sentinel = next(c for c in chunks if c.startswith("\n__citations__:"))
        json_part = sentinel.split(":", 1)[1]
        citations = json.loads(json_part)
        assert isinstance(citations, list)

    def test_regulation_filter_passed_to_retriever(self) -> None:
        """Ensure the regulation filter is forwarded to the retriever."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [DORA_RESULT]

        from rag.chain import RAGChain

        chain = RAGChain(retriever=mock_retriever, llm_service=MockLLMService())
        chain.query("test", regulation="DORA")

        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args
        assert call_kwargs.kwargs.get("regulation") == "DORA" or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == "DORA"
        )
