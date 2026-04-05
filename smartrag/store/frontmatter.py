"""YAML frontmatter parser and writer for SmartRAG markdown documents.

Handles reading, writing, and updating YAML frontmatter blocks delimited
by ``---`` at the top of markdown files.  All YAML operations use
``yaml.safe_load`` / ``yaml.safe_dump`` exclusively.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical key ordering for deterministic output
# ---------------------------------------------------------------------------

_CANONICAL_ORDER: list[str] = [
    "title",
    "summary",
    "type",
    "categories",
    "concepts",
    "fingerprint",
    "backlinks",
    "parent",
    "children",
    "section_map",
    "split_from",
    "section_index",
    "created",
    "updated",
    "source",
    "code_structure",
]

# Keys whose values should always be rendered in YAML block (nested) style.
_BLOCK_STYLE_KEYS: frozenset[str] = frozenset({"section_map"})

# Regex that matches the ``---`` delimited frontmatter block at the very
# start of a document.  The block must begin on the first line.
_FM_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)---[ \t]*\r?\n?",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split *content* into a (metadata, body) pair.

    Parameters
    ----------
    content:
        Full markdown string, possibly prefixed with a YAML frontmatter
        block delimited by ``---``.

    Returns
    -------
    tuple[dict, str]
        ``(metadata_dict, body_string)``.  If no frontmatter is present
        the metadata dict is empty and body is the full *content*.
        Malformed YAML produces a logged warning and an empty dict.
    """
    if not content or not content.startswith("---"):
        return {}, content or ""

    match = _FM_RE.match(content)
    if match is None:
        # Opening ``---`` found but no closing delimiter.
        logger.warning("Frontmatter opening delimiter found but no closing '---'")
        return {}, content

    yaml_text = match.group(1)
    # Strip exactly one leading newline from body so that the blank line
    # added by write_frontmatter round-trips cleanly.
    body = content[match.end() :]
    if body.startswith("\n"):
        body = body[1:]

    # Handle empty frontmatter block (only whitespace between delimiters).
    if not yaml_text.strip():
        return {}, body

    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        logger.warning("Malformed YAML in frontmatter: %s", exc)
        return {}, body

    if not isinstance(parsed, dict):
        logger.warning(
            "Frontmatter YAML parsed to %s instead of dict; ignoring",
            type(parsed).__name__,
        )
        return {}, body

    return parsed, body


def write_frontmatter(metadata: dict[str, Any], body: str) -> str:
    """Combine *metadata* and *body* into a markdown string with YAML frontmatter.

    Keys are emitted in the canonical order defined by ``_CANONICAL_ORDER``.
    Keys not in the canonical list are appended alphabetically after the
    canonical ones.  ``None`` and empty-collection values are omitted to
    keep the output clean.

    Short lists (categories, concepts, fingerprint, backlinks, children)
    use YAML flow style (``[a, b, c]``).  The ``section_map`` key uses
    block style for readability.

    A single blank line separates the closing ``---`` from the body.
    """
    if not metadata:
        return f"---\n---\n\n{body}"

    ordered = _sort_metadata(metadata)
    yaml_text = _dump_yaml(ordered)

    # Guarantee exactly one blank line between closing --- and body.
    body = body.lstrip("\n")
    return f"---\n{yaml_text}---\n\n{body}"


def update_frontmatter(content: str, updates: dict[str, Any]) -> str:
    """Parse *content*, shallow-merge *updates* into the metadata, and rewrite.

    Parameters
    ----------
    content:
        Full markdown string (with or without existing frontmatter).
    updates:
        Key/value pairs to merge into the existing metadata.  Existing
        keys are overwritten; new keys are added.

    Returns
    -------
    str
        The reconstructed markdown with updated frontmatter.
    """
    metadata, body = parse_frontmatter(content)
    metadata.update(updates)
    return write_frontmatter(metadata, body)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return *metadata* ordered by canonical key order.

    Keys present in ``_CANONICAL_ORDER`` come first (in that order).
    Remaining keys are appended in sorted (alphabetical) order.
    """
    order_map = {k: i for i, k in enumerate(_CANONICAL_ORDER)}
    max_idx = len(_CANONICAL_ORDER)

    def _key(item: tuple[str, Any]) -> tuple[int, str]:
        return (order_map.get(item[0], max_idx), item[0])

    return dict(sorted(metadata.items(), key=_key))


def _dump_yaml(ordered: dict[str, Any]) -> str:
    """Serialize *ordered* dict to a YAML string with mixed flow/block style.

    Short lists get flow style (compact inline ``[a, b]``).  The
    ``section_map`` key and other deeply nested structures use block style.
    """
    lines: list[str] = []

    for key, value in ordered.items():
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue

        if key in _BLOCK_STYLE_KEYS:
            # Block style for complex nested structures.
            chunk = yaml.safe_dump(
                {key: value},
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            lines.append(chunk.rstrip("\n"))
        elif isinstance(value, list) and _is_simple_list(value):
            # Flow-style for short, simple lists.
            chunk = yaml.safe_dump(
                {key: value},
                default_flow_style=None,
                sort_keys=False,
                allow_unicode=True,
            )
            # safe_dump with default_flow_style=None renders simple lists
            # in block style; force flow style manually.
            flow_repr = _flow_list(value)
            lines.append(f"{key}: {flow_repr}")
        else:
            chunk = yaml.safe_dump(
                {key: value},
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            lines.append(chunk.rstrip("\n"))

    return "\n".join(lines) + "\n" if lines else ""


def _is_simple_list(value: list[Any]) -> bool:
    """Return True if *value* is a flat list of scalars (str, int, float, bool)."""
    return all(isinstance(v, (str, int, float, bool)) for v in value)


def _flow_list(items: list[Any]) -> str:
    """Render a flat list in YAML flow style: ``[a, b, c]``.

    Strings that require quoting (contain special YAML characters) are
    single-quoted.
    """
    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            parts.append(_yaml_scalar(item))
        elif isinstance(item, bool):
            parts.append("true" if item else "false")
        else:
            parts.append(str(item))
    return "[" + ", ".join(parts) + "]"


def _yaml_scalar(value: str) -> str:
    """Return a YAML-safe representation of a scalar string.

    Plain strings are unquoted.  Strings that could be misinterpreted by
    YAML (contain colons, hashes, brackets, etc.) are single-quoted.
    """
    if not value:
        return "''"
    # Characters that force quoting in YAML.
    needs_quoting = set(":#{}[]|>&*!?@`,\n\r\t")
    if any(c in needs_quoting for c in value):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    # Values that look like YAML booleans, nulls, or numbers.
    lower = value.lower()
    if lower in {
        "true",
        "false",
        "yes",
        "no",
        "on",
        "off",
        "null",
        "~",
    }:
        return f"'{value}'"
    # Check if it looks like a number.
    try:
        float(value)
        return f"'{value}'"
    except ValueError:
        pass
    return value
