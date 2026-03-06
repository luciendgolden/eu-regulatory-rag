"""Prompt assembly for EU regulatory RAG.

Builds the system prompt and user message from retrieved chunks,
injecting citation metadata and the user's question.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert EU regulatory compliance assistant specialising in the Digital \
Operational Resilience Act (DORA), the Network and Information Security Directive (NIS2), \
and related EU financial and cybersecurity regulations.

Your role is to provide precise, authoritative answers grounded exclusively in the \
regulatory text provided to you. When answering:

- Cite specific articles, chapters, and recitals by name and number.
- Use the citation format: "According to [Regulation] Article [N] ([Title])..."
- If the provided context does not contain sufficient information to answer the question,
  say so clearly — do not speculate or invent regulatory provisions.
- Structure complex answers with numbered or bulleted lists where appropriate.
- Maintain a professional, compliance-oriented tone throughout.
"""

# ---------------------------------------------------------------------------
# User message template
# ---------------------------------------------------------------------------

USER_TEMPLATE = (
    "Based on the following regulatory excerpts:\n\n"
    "{context}\n\n"
    "Question: {query}\n\n"
    "Provide a detailed answer with specific article references."
)


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------


def format_citation(result: dict[str, Any]) -> str:
    """Format a single retrieval result as a citation string.

    Example output:
        "According to DORA Article 5 (ICT risk management framework)..."
    """
    regulation = result.get("regulation", "")
    section_type = (result.get("section_type") or "").capitalize()
    section_number = result.get("section_number")
    section_title = result.get("section_title") or ""
    chapter = result.get("chapter")

    parts: list[str] = [f"According to {regulation}"]

    if chapter:
        parts.append(f"Chapter {chapter}")

    if section_type and section_number is not None:
        label = f"{section_type} {section_number}"
        if section_title:
            label = f"{label} ({section_title})"
        parts.append(label)
    elif section_type:
        label = section_type
        if section_title:
            label = f"{label} ({section_title})"
        parts.append(label)

    return " — ".join(parts)


def build_context_block(results: list[dict[str, Any]]) -> str:
    """Convert a list of retrieval results into a numbered context block.

    Each entry is prefixed with its citation, followed by the chunk text.
    """
    if not results:
        return "No relevant regulatory text found for this query."

    sections: list[str] = []
    for i, res in enumerate(results, start=1):
        citation = format_citation(res)
        text = (res.get("text") or "").strip()
        sections.append(f"[{i}] {citation}\n{text}")

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_prompt(
    query: str,
    results: list[dict[str, Any]],
) -> tuple[str, str]:
    """Assemble the (system_prompt, user_message) pair for the LLM.

    Args:
        query:   The user's natural-language question.
        results: Retrieved chunks from :meth:`RegulatoryRetriever.retrieve`.

    Returns:
        A ``(system, user)`` tuple ready to pass to any LLM service.
    """
    context = build_context_block(results)
    user_message = USER_TEMPLATE.format(context=context, query=query)
    return SYSTEM_PROMPT, user_message
