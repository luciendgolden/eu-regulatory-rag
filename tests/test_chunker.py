"""Tests for the semantic chunker."""

import pytest
from ingestion.chunker import Chunk, RegulationChunker
from ingestion.parsers.eurlex_parser import Section


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_section(
    text: str,
    section_type: str = "article",
    section_number: str = "1",
    parent_section: str | None = None,
) -> Section:
    return Section(
        regulation_id="DORA",
        celex_number="32022R2554",
        section_type=section_type,
        section_number=section_number,
        title=f"Article {section_number}",
        text=text,
        parent_section=parent_section,
    )


SHORT_TEXT = "Financial entities shall maintain an ICT risk management framework."

# ~600 words to force splitting (well over 512 tokens)
LONG_TEXT = (
    "Financial entities shall have in place a sound, comprehensive and well-documented "
    "ICT risk management framework as part of their overall risk management system. "
    "Such framework shall allow financial entities to address ICT risk quickly, efficiently "
    "and comprehensively and to ensure a high level of digital operational resilience.\n\n"
    "The ICT risk management framework shall include at least the following: "
    "(a) a sound and documented ICT risk management and governance structure; "
    "(b) ICT-related policies, procedures, protocols and tools necessary to duly protect "
    "all information assets and ICT assets; "
    "(c) processes to identify, classify and document all ICT supported business functions "
    "and the related information assets and ICT assets.\n\n"
    "Financial entities shall continuously monitor and manage the lifecycle of their ICT "
    "assets in terms of purchase, use, decommissioning and disposal, including their "
    "classification in terms of data sensitivity and criticality.\n\n"
    "Financial entities shall conduct ICT risk assessments to identify, classify and "
    "document all ICT risk sources including risks related to ICT service providers and "
    "third-party providers. When relevant, financial entities shall group similar ICT "
    "risk sources together and review and update their risk assessment at least once a year.\n\n"
    "The ICT risk management framework shall be documented and reviewed at least once a year, "
    "as well as upon the occurrence of major ICT-related incidents and following supervisory "
    "instructions or conclusions derived from relevant digital operational resilience testing "
    "or audit processes. It shall be continuously improved on the basis of lessons derived "
    "from implementation and monitoring. A report on the review of the ICT risk management "
    "framework shall be submitted to the competent authority upon its request.\n\n"
    "Financial entities that are not microenterprises shall assign the responsibility for "
    "managing and overseeing ICT risk to a control function and ensure an appropriate level "
    "of independence of such control function to avoid conflicts of interest."
)

TOPICS = ["ict_risk", "financial_sector"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chunk_returns_list():
    chunker = RegulationChunker()
    section = _make_section(SHORT_TEXT)
    chunks = chunker._chunk_section(section, TOPICS)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_short_text_produces_one_chunk():
    chunker = RegulationChunker()
    section = _make_section(SHORT_TEXT)
    chunks = chunker._chunk_section(section, TOPICS)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks():
    chunker = RegulationChunker(max_tokens=100)
    section = _make_section(LONG_TEXT)
    chunks = chunker._chunk_section(section, TOPICS)
    assert len(chunks) > 1


def test_chunk_metadata_fields():
    chunker = RegulationChunker()
    section = _make_section(SHORT_TEXT, parent_section="I")
    chunks = chunker._chunk_section(section, TOPICS)
    c = chunks[0]

    assert c.regulation == "DORA"
    assert c.celex == "32022R2554"
    assert c.section_type == "article"
    assert c.section_number == "1"
    assert c.section_title == "Article 1"
    assert c.chapter == "I"
    assert c.topics == TOPICS
    assert c.chunk_index == 0


def test_chunk_total_and_index_consistent():
    chunker = RegulationChunker(max_tokens=100)
    section = _make_section(LONG_TEXT)
    chunks = chunker._chunk_section(section, TOPICS)
    assert all(c.total_chunks == len(chunks) for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_content_hash_set():
    chunker = RegulationChunker()
    section = _make_section(SHORT_TEXT)
    chunks = chunker._chunk_section(section, [])
    assert chunks[0].content_hash != ""
    assert len(chunks[0].content_hash) == 64  # SHA-256 hex


def test_chunk_hashes_are_unique_for_different_texts():
    chunker = RegulationChunker(max_tokens=50)
    section = _make_section(LONG_TEXT)
    chunks = chunker._chunk_section(section, [])
    hashes = [c.content_hash for c in chunks]
    assert len(hashes) == len(set(hashes)), "Duplicate content hashes found"


def test_to_payload_returns_dict():
    chunker = RegulationChunker()
    section = _make_section(SHORT_TEXT)
    chunk = chunker._chunk_section(section, TOPICS)[0]
    payload = chunk.to_payload()
    assert isinstance(payload, dict)
    assert "text" in payload
    assert "regulation" in payload
    assert "celex" in payload
    assert "section_type" in payload
    assert "topics" in payload
    assert "content_hash" in payload


def test_empty_section_produces_no_chunks():
    chunker = RegulationChunker()
    section = _make_section("")
    chunks = chunker._chunk_section(section, [])
    assert chunks == []


def test_chunk_sections_multiple():
    """chunk_sections() should flatten chunks from all sections."""
    chunker = RegulationChunker()
    sections = [
        _make_section(SHORT_TEXT, section_number="1"),
        _make_section(SHORT_TEXT, section_number="2"),
        _make_section(SHORT_TEXT, section_number="3"),
    ]
    # Call chunk_sections directly to test it
    all_chunks: list[Chunk] = []
    for s in sections:
        all_chunks.extend(chunker._chunk_section(s, TOPICS))

    assert len(all_chunks) == 3
    regulations = {c.regulation for c in all_chunks}
    assert regulations == {"DORA"}
