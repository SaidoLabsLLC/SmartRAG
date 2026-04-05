"""Section splitter for long documents.

Splits documents that exceed a word-count threshold into a parent article
with child section articles, preserving structure and generating proper
cross-references via frontmatter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from smartrag.types import SplitDocument, SplitResult


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Return the number of whitespace-delimited tokens in *text*."""
    return len(text.split())


def _slugify_heading(heading: str) -> str:
    """Convert a markdown heading string into a kebab-case slug.

    Strips leading ``#`` characters, lowercases, replaces non-alphanumeric
    runs with hyphens, and truncates to 30 characters on a word boundary.
    """
    # Remove leading # and whitespace
    cleaned = re.sub(r"^#+\s*", "", heading).strip()
    # Lowercase and replace non-alphanum with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", cleaned.lower()).strip("-")
    # Truncate to 30 chars on a hyphen boundary
    if len(slug) > 30:
        truncated = slug[:30]
        # Cut at last hyphen so we don't break mid-word
        last_hyphen = truncated.rfind("-")
        if last_hyphen > 0:
            truncated = truncated[:last_hyphen]
        slug = truncated
    return slug


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    """Raw parsed section before assembly into SplitDocuments."""
    heading: str  # original heading text (empty for preamble)
    body: str     # section body content (without the heading line itself)
    level: int    # 2 for ##, 3 for ###, 0 for preamble / no-header


# ---------------------------------------------------------------------------
# SectionSplitter
# ---------------------------------------------------------------------------

class SectionSplitter:
    """Splits long markdown documents into parent + child articles.

    Parameters
    ----------
    threshold:
        Word-count threshold.  Documents with word counts **strictly**
        greater than this value are eligible for splitting.
    """

    def __init__(self, threshold: int = 2000) -> None:
        self.threshold = threshold

    # -- public API ---------------------------------------------------------

    def should_split(self, text: str) -> bool:
        """Return ``True`` when *text* exceeds the word-count threshold."""
        return count_words(text) > self.threshold

    def split(
        self,
        text: str,
        source_slug: str,
        metadata: dict,
    ) -> SplitResult:
        """Split *text* into parent + children, or return as a single doc.

        Parameters
        ----------
        text:
            Full markdown body of the document.
        source_slug:
            The slug assigned to this document prior to splitting.
        metadata:
            Arbitrary metadata dict to merge into every produced
            document's frontmatter.
        """
        if not self.should_split(text):
            return self._as_single(text, source_slug, metadata)

        # Try H2 first, then H3
        used_headers = False
        sections = self._parse_sections(text, level=2)
        content_sections = self._extract_content_sections(sections)
        if len(content_sections) > 1:
            used_headers = True

        if not used_headers:
            sections = self._parse_sections(text, level=3)
            content_sections = self._extract_content_sections(sections)
            if len(content_sections) > 1:
                used_headers = True

        # If still only one logical section (or no headers at all), try
        # paragraph-boundary splitting.
        if not used_headers:
            sections = self._split_by_paragraphs(text)
            content_sections = sections  # paragraph splits have no preamble

        # After all strategies: if we still have only one chunk, don't split.
        if len(content_sections) <= 1:
            return self._as_single(text, source_slug, metadata)

        # Merge short content sections (<50 words) into their successor.
        # The preamble is never subject to merging — it always becomes
        # part of the parent body.
        content_sections = self._merge_short_sections(content_sections)

        # Re-check after merging — may have collapsed back to one section
        if len(content_sections) <= 1:
            return self._as_single(text, source_slug, metadata)

        # Extract preamble from the header-based parse only.
        # Paragraph splits have no structural preamble.
        preamble = self._extract_preamble(sections) if used_headers else ""

        return self._assemble_from_parts(
            preamble, content_sections, source_slug, metadata
        )

    # -- parsing helpers ----------------------------------------------------

    def _parse_sections(
        self, text: str, level: int
    ) -> list[_Section]:
        """Split *text* on headings of *level* (2 or 3).

        Content before the first heading becomes the preamble section
        (level=0).  Nested ``###`` under ``##`` is kept with the parent
        ``##`` section — we only split on the requested level.
        """
        # Pattern matches a line starting with exactly `level` '#' chars
        pattern = re.compile(
            rf"^({'#' * level})\s+(.+)$", re.MULTILINE
        )

        parts: list[_Section] = []
        last_end = 0
        last_heading = ""
        last_level = 0

        for match in pattern.finditer(text):
            chunk = text[last_end : match.start()].strip()
            if last_end == 0:
                # Everything before the first heading is the preamble
                if chunk:
                    parts.append(_Section(heading="", body=chunk, level=0))
            else:
                parts.append(
                    _Section(heading=last_heading, body=chunk, level=last_level)
                )
            last_heading = match.group(0)  # full heading line
            last_level = level
            last_end = match.end()

        # Trailing content after last heading
        tail = text[last_end:].strip()
        if last_end == 0:
            # No headings found at all — return the whole doc as one section
            if tail:
                parts.append(_Section(heading="", body=tail, level=0))
        else:
            parts.append(
                _Section(heading=last_heading, body=tail, level=last_level)
            )

        return parts

    def _split_by_paragraphs(self, text: str) -> list[_Section]:
        """Split *text* at double-newline boundaries near *threshold* words."""
        paragraphs = re.split(r"\n{2,}", text)
        if len(paragraphs) <= 1:
            return [_Section(heading="", body=text.strip(), level=0)]

        sections: list[_Section] = []
        current_parts: list[str] = []
        current_words = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_words = count_words(para)

            # If adding this paragraph would exceed threshold and we already
            # have accumulated content, flush the current chunk.
            if current_words + para_words > self.threshold and current_parts:
                sections.append(
                    _Section(
                        heading="",
                        body="\n\n".join(current_parts),
                        level=0,
                    )
                )
                current_parts = []
                current_words = 0

            current_parts.append(para)
            current_words += para_words

        # Flush remainder
        if current_parts:
            sections.append(
                _Section(
                    heading="",
                    body="\n\n".join(current_parts),
                    level=0,
                )
            )

        return sections

    @staticmethod
    def _merge_short_sections(sections: list[_Section]) -> list[_Section]:
        """Merge sections with fewer than 50 words into the next section.

        A short section's heading and body are prepended to the *next*
        section's body, while the next section keeps its own heading.
        If the short section is the last one, it is appended to the
        previous section's body instead.
        """
        if not sections:
            return sections

        merged: list[_Section] = []
        carry: _Section | None = None

        for section in sections:
            if carry is not None:
                # Prepend carried content into this section's body,
                # but preserve this section's heading.
                extra_parts: list[str] = []
                if carry.heading:
                    extra_parts.append(carry.heading)
                if carry.body:
                    extra_parts.append(carry.body)
                extra = "\n\n".join(extra_parts)

                new_body = (
                    f"{extra}\n\n{section.body}" if section.body else extra
                ).strip()

                section = _Section(
                    heading=section.heading,
                    body=new_body,
                    level=section.level,
                )
                carry = None

            # Check word count of the section (heading words + body words)
            total_text = f"{section.heading} {section.body}".strip()
            if count_words(total_text) < 50:
                carry = section
                continue

            merged.append(section)

        # If the last section was short, append it to the previous one
        if carry is not None:
            if merged:
                prev = merged[-1]
                extra_parts_tail: list[str] = []
                if carry.heading:
                    extra_parts_tail.append(carry.heading)
                if carry.body:
                    extra_parts_tail.append(carry.body)
                extra_tail = "\n\n".join(extra_parts_tail)
                new_body = (
                    f"{prev.body}\n\n{extra_tail}" if prev.body else extra_tail
                ).strip()
                merged[-1] = _Section(
                    heading=prev.heading,
                    body=new_body,
                    level=prev.level,
                )
            else:
                merged.append(carry)

        return merged

    # -- section extraction helpers ----------------------------------------

    @staticmethod
    def _extract_preamble(sections: list[_Section]) -> str:
        """Return the preamble text (level-0 first section), or ``""``."""
        if sections and sections[0].level == 0:
            return sections[0].body
        return ""

    @staticmethod
    def _extract_content_sections(sections: list[_Section]) -> list[_Section]:
        """Return sections that are actual content (skip leading preamble)."""
        if sections and sections[0].level == 0:
            return sections[1:]
        return list(sections)

    # -- assembly -----------------------------------------------------------

    def _assemble_from_parts(
        self,
        preamble: str,
        content_sections: list[_Section],
        source_slug: str,
        metadata: dict,
    ) -> SplitResult:
        """Build parent + child ``SplitDocument`` objects."""
        children: list[SplitDocument] = []
        section_map: list[dict[str, str]] = []
        child_slugs: list[str] = []
        links: list[str] = []

        for idx, section in enumerate(content_sections):
            # Derive child slug
            if section.heading:
                heading_slug = _slugify_heading(section.heading)
                child_slug = f"{source_slug}-{heading_slug}" if heading_slug else f"{source_slug}-section-{idx}"
            else:
                child_slug = f"{source_slug}-section-{idx}"

            # Derive title from heading or fallback
            if section.heading:
                title = re.sub(r"^#+\s*", "", section.heading).strip()
            else:
                # Use first line or a generic title
                first_line = section.body.split("\n", 1)[0].strip()
                title = first_line[:60] if first_line else f"Section {idx + 1}"

            # Build child body: include heading in the body for context
            child_body_parts: list[str] = []
            if section.heading:
                child_body_parts.append(section.heading)
            if section.body:
                child_body_parts.append(section.body)
            child_body = "\n\n".join(child_body_parts).strip()

            child_frontmatter = {
                **metadata,
                "title": title,
                "parent": source_slug,
                "split_from": source_slug,
                "section_index": idx,
                "categories": metadata.get("categories", []),
                "concepts": metadata.get("concepts", []),
            }

            children.append(
                SplitDocument(
                    slug=child_slug,
                    body=child_body,
                    frontmatter=child_frontmatter,
                )
            )

            section_map.append(
                {"slug": child_slug, "title": title, "synopsis": ""}
            )
            child_slugs.append(child_slug)
            links.append(f"- [[{child_slug}]]")

        # Build parent body
        parent_body_parts: list[str] = []
        if preamble:
            parent_body_parts.append(preamble)
        parent_body_parts.append("## Sections\n")
        parent_body_parts.append("\n".join(links))
        parent_body = "\n\n".join(parent_body_parts).strip()

        parent_frontmatter = {
            **metadata,
            "section_map": section_map,
            "children": child_slugs,
        }

        parent = SplitDocument(
            slug=source_slug,
            body=parent_body,
            frontmatter=parent_frontmatter,
        )

        return SplitResult(
            is_split=True,
            parent=parent,
            children=children,
        )

    @staticmethod
    def _as_single(
        text: str,
        source_slug: str,
        metadata: dict,
    ) -> SplitResult:
        """Wrap *text* as a non-split single document."""
        doc = SplitDocument(
            slug=source_slug,
            body=text.strip(),
            frontmatter={**metadata},
        )
        return SplitResult(is_split=False, single=doc)
