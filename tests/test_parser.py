"""Tests for the EUR-Lex HTML parser."""

import pytest
from ingestion.parsers.eurlex_parser import EurLexParser, Section, _clean_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REGULATION_ID = "DORA"
CELEX = "32022R2554"


def _make_parser() -> EurLexParser:
    return EurLexParser(regulation_id=REGULATION_ID, celex_number=CELEX)


SIMPLE_HTML = """
<html>
<body>
  <div class="preamble">
    <p>THE EUROPEAN PARLIAMENT AND OF THE COUNCIL,</p>
    <p>Having regard to the Treaty on the Functioning of the European Union,</p>
  </div>
  <p>(1) The digital transformation of finance has accelerated over recent years.</p>
  <p>(2) ICT risk has become a key concern for financial entities.</p>
  <article>
    <h2>Article 1</h2>
    <p>Subject matter</p>
    <p>This Regulation lays down uniform requirements concerning the security of network and information systems.</p>
  </article>
  <article>
    <h2>Article 2</h2>
    <p>Scope</p>
    <p>This Regulation applies to the following financial entities: (a) credit institutions; (b) payment institutions.</p>
  </article>
  <h3>Annex I</h3>
  <p>List of ICT services considered critical.</p>
</body>
</html>
"""

ARTICLE_ONLY_HTML = """
<html><body>
  <div>Article 5 — ICT risk management framework</div>
  <p>Financial entities shall have in place a sound, comprehensive and well-documented ICT risk management framework.</p>
  <div>Article 6 — ICT risk management systems</div>
  <p>ICT risk management systems shall include policies, procedures, and protocols.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_returns_list():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    assert isinstance(sections, list)
    assert len(sections) > 0


def test_all_sections_are_section_objects():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    for s in sections:
        assert isinstance(s, Section)


def test_section_regulation_id():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    for s in sections:
        assert s.regulation_id == REGULATION_ID
        assert s.celex_number == CELEX


def test_preamble_extracted():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    preamble_sections = [s for s in sections if s.section_type == "preamble"]
    assert len(preamble_sections) >= 1
    assert preamble_sections[0].title.lower() == "preamble"


def test_recitals_extracted():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    recitals = [s for s in sections if s.section_type == "recital"]
    assert len(recitals) >= 1
    # Should pick up "(1)" and "(2)"
    numbers = [r.section_number for r in recitals if r.section_number]
    assert "1" in numbers or len(numbers) > 0


def test_articles_extracted():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    articles = [s for s in sections if s.section_type == "article"]
    assert len(articles) >= 1


def test_article_numbers_extracted():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    articles = [s for s in sections if s.section_type == "article"]
    numbers = [a.section_number for a in articles if a.section_number]
    assert len(numbers) >= 1


def test_annex_extracted():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    annexes = [s for s in sections if s.section_type == "annex"]
    assert len(annexes) >= 1


def test_section_text_not_empty():
    parser = _make_parser()
    sections = parser.parse(SIMPLE_HTML)
    for s in sections:
        assert s.text.strip(), f"Empty text for {s.label}"


def test_fallback_parse_plain_html():
    """Parser should not crash and return something even for unexpected HTML."""
    parser = _make_parser()
    sections = parser.parse(ARTICLE_ONLY_HTML)
    assert len(sections) > 0


def test_section_label():
    s = Section(
        regulation_id="DORA",
        celex_number="32022R2554",
        section_type="article",
        section_number="5",
        title="ICT risk",
        text="...",
    )
    assert s.label == "Article 5"


def test_section_label_no_number():
    s = Section(
        regulation_id="DORA",
        celex_number="32022R2554",
        section_type="preamble",
        section_number=None,
        title="Preamble",
        text="...",
    )
    assert s.label == "Preamble"


def test_clean_text():
    dirty = "  hello   \n\n\n\n   world   \n"
    result = _clean_text(dirty)
    assert "hello" in result
    assert "world" in result
    # Should not have more than 2 consecutive newlines
    assert "\n\n\n" not in result
