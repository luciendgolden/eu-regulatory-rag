"""EUR-Lex HTML parser — converts regulation HTML into structured sections.

Supported section types: preamble, recital, article, chapter, annex.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """A structured section extracted from a regulation document."""

    regulation_id: str
    celex_number: str
    section_type: str  # preamble | recital | article | chapter | annex
    section_number: Optional[str]  # e.g. "5", "II", "I"
    title: str
    text: str
    parent_section: Optional[str] = None  # e.g. parent chapter number

    # Convenience helpers
    @property
    def label(self) -> str:
        if self.section_number:
            return f"{self.section_type.capitalize()} {self.section_number}"
        return self.section_type.capitalize()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class EurLexParser:
    """Parse EUR-Lex HTML into a list of :class:`Section` objects."""

    # CSS / tag patterns used in EUR-Lex HTML (vary between documents)
    _ARTICLE_CLASSES = {"eli-subdivision", "article", "doc-ti"}
    _CHAPTER_CLASSES = {"title", "chapter"}
    _RECITAL_CLASSES = {"recital"}
    _PREAMBLE_CLASSES = {"preamble", "doc-preamble"}

    def __init__(self, regulation_id: str, celex_number: str) -> None:
        self.regulation_id = regulation_id
        self.celex_number = celex_number

    def parse(self, html: str) -> list[Section]:
        """Parse HTML and return all extracted sections."""
        soup = BeautifulSoup(html, "lxml")
        sections: list[Section] = []

        # Try structured EUR-Lex HTML first
        sections.extend(self._extract_preamble(soup))
        sections.extend(self._extract_recitals(soup))
        sections.extend(self._extract_chapters_and_articles(soup))
        sections.extend(self._extract_annexes(soup))

        if not sections:
            logger.warning(
                "No sections extracted for %s — falling back to plain-text split",
                self.celex_number,
            )
            sections = self._fallback_parse(soup)

        logger.info(
            "Parsed %d sections from %s (%s)",
            len(sections),
            self.regulation_id,
            self.celex_number,
        )
        return sections

    # ------------------------------------------------------------------
    # Section extractors
    # ------------------------------------------------------------------

    def _extract_preamble(self, soup: BeautifulSoup) -> list[Section]:
        sections = []
        # Look for explicit preamble element
        preamble_el = soup.find(
            lambda tag: isinstance(tag, Tag)
            and bool(tag.get("class"))
            and any(
                cls in self._PREAMBLE_CLASSES
                for cls in (tag.get("class") or [])
            )
        )
        if preamble_el:
            text = _clean_text(preamble_el.get_text(separator="\n"))
            if text:
                sections.append(
                    Section(
                        regulation_id=self.regulation_id,
                        celex_number=self.celex_number,
                        section_type="preamble",
                        section_number=None,
                        title="Preamble",
                        text=text,
                    )
                )
        else:
            # Try to find preamble by heading text
            for tag in soup.find_all(["h1", "h2", "p"]):
                if "preamble" in (tag.get_text() or "").lower():
                    text = _clean_text(tag.get_text(separator="\n"))
                    if text:
                        sections.append(
                            Section(
                                regulation_id=self.regulation_id,
                                celex_number=self.celex_number,
                                section_type="preamble",
                                section_number=None,
                                title="Preamble",
                                text=text,
                            )
                        )
                    break
        return sections

    def _extract_recitals(self, soup: BeautifulSoup) -> list[Section]:
        sections = []
        # Pattern: <p> or <div> with recital numbering like "(1)", "(2)" …
        recital_re = re.compile(r"^\s*\((\d+)\)\s*")

        for tag in soup.find_all(["p", "div"]):
            classes = tag.get("class") or []
            is_recital_class = any(c in self._RECITAL_CLASSES for c in classes)

            text = tag.get_text(separator=" ", strip=True)
            match = recital_re.match(text)

            if is_recital_class or match:
                num = match.group(1) if match else None
                clean = recital_re.sub("", text).strip()
                if len(clean) > 30:  # skip noise
                    sections.append(
                        Section(
                            regulation_id=self.regulation_id,
                            celex_number=self.celex_number,
                            section_type="recital",
                            section_number=num,
                            title=f"Recital {num}" if num else "Recital",
                            text=_clean_text(clean),
                        )
                    )
        return sections

    def _extract_chapters_and_articles(self, soup: BeautifulSoup) -> list[Section]:
        """Extract chapters and articles, linking articles to their chapter."""
        sections: list[Section] = []
        current_chapter: Optional[str] = None

        # EUR-Lex uses a mix of <article>, <div class="eli-subdivision">, headings
        article_re = re.compile(r"Article\s+(\d+[a-z]?)", re.IGNORECASE)
        chapter_re = re.compile(
            r"(?:Chapter|Title|Section)\s+([\dIVXivx]+)", re.IGNORECASE
        )

        all_elements = soup.find_all(
            ["article", "div", "section", "h2", "h3", "h4"]
        )

        for el in all_elements:
            if not isinstance(el, Tag):
                continue
            classes = el.get("class") or []
            text_preview = (el.get_text(separator=" ", strip=True))[:200]

            # ---- Chapter / Title heading ----
            ch_match = chapter_re.match(text_preview)
            if ch_match and el.name in ("h2", "h3", "h4"):
                current_chapter = ch_match.group(1)
                title = _clean_text(text_preview)
                full_text = _clean_text(el.get_text(separator="\n"))
                sections.append(
                    Section(
                        regulation_id=self.regulation_id,
                        celex_number=self.celex_number,
                        section_type="chapter",
                        section_number=current_chapter,
                        title=title,
                        text=full_text,
                    )
                )
                continue

            # ---- Article ----
            art_match = article_re.match(text_preview)
            is_article_class = (
                el.name == "article"
                or any(c in self._ARTICLE_CLASSES for c in classes)
            )

            if art_match or is_article_class:
                art_num = art_match.group(1) if art_match else None
                full_text = _clean_text(el.get_text(separator="\n"))

                if len(full_text) < 20:
                    continue  # skip empty/noise elements

                # Try to extract title from first child heading
                title = ""
                heading = el.find(["h1", "h2", "h3", "h4", "p", "strong"])
                if heading:
                    title = _clean_text(heading.get_text())
                if not title:
                    title = f"Article {art_num}" if art_num else "Article"

                sections.append(
                    Section(
                        regulation_id=self.regulation_id,
                        celex_number=self.celex_number,
                        section_type="article",
                        section_number=art_num,
                        title=title,
                        text=full_text,
                        parent_section=current_chapter,
                    )
                )

        return sections

    def _extract_annexes(self, soup: BeautifulSoup) -> list[Section]:
        sections = []
        annex_re = re.compile(r"Annex\s*([\dIVXivx]*)", re.IGNORECASE)

        for tag in soup.find_all(["div", "section", "h2", "h3"]):
            text_preview = tag.get_text(separator=" ", strip=True)[:200]
            match = annex_re.match(text_preview)
            if match:
                num = match.group(1) or None
                # For heading elements, collect text from following siblings too
                if tag.name in ("h2", "h3", "h4"):
                    parts = [tag.get_text(separator="\n")]
                    sibling = tag.find_next_sibling()
                    while sibling and sibling.name not in ("h2", "h3", "h4"):
                        parts.append(sibling.get_text(separator="\n"))
                        sibling = sibling.find_next_sibling()
                    full_text = _clean_text("\n".join(parts))
                else:
                    full_text = _clean_text(tag.get_text(separator="\n"))
                # Accept even short annexes (heading-only is valid)
                if full_text:
                    sections.append(
                        Section(
                            regulation_id=self.regulation_id,
                            celex_number=self.celex_number,
                            section_type="annex",
                            section_number=num,
                            title=f"Annex {num}" if num else "Annex",
                            text=full_text,
                        )
                    )
        return sections

    def _fallback_parse(self, soup: BeautifulSoup) -> list[Section]:
        """Last-resort: split visible text by 'Article N' headings."""
        full_text = soup.get_text(separator="\n")
        article_re = re.compile(r"(Article\s+\d+[a-z]?)", re.IGNORECASE)
        parts = article_re.split(full_text)
        sections = []

        it = iter(parts)
        preamble_text = next(it, "").strip()
        if preamble_text:
            sections.append(
                Section(
                    regulation_id=self.regulation_id,
                    celex_number=self.celex_number,
                    section_type="preamble",
                    section_number=None,
                    title="Preamble",
                    text=_clean_text(preamble_text),
                )
            )

        while True:
            heading = next(it, None)
            body = next(it, None)
            if heading is None:
                break
            art_match = re.search(r"\d+[a-z]?", heading, re.IGNORECASE)
            art_num = art_match.group(0) if art_match else None
            text = _clean_text((body or "").strip())
            if text:
                sections.append(
                    Section(
                        regulation_id=self.regulation_id,
                        celex_number=self.celex_number,
                        section_type="article",
                        section_number=art_num,
                        title=heading.strip(),
                        text=text,
                    )
                )
        return sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Normalise whitespace in extracted text."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
