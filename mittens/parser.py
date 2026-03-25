"""Markdown + YAML frontmatter parser.

Single responsibility: read a markdown file, extract YAML frontmatter
and named sections. Does NOT interpret the content — that's the
registry's job.

Uses a custom YAML loading approach to handle Mittens' markdown-heavy
frontmatter (backticks, asterisks in exit criteria strings, etc.)
that standard YAML parsers choke on.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from mittens.types import ParsedDoc

# Matches ## or ### headings
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

# Matches frontmatter delimiters
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_doc(path: str | Path) -> ParsedDoc:
    """Parse a markdown file with YAML frontmatter into a ParsedDoc."""
    path = Path(path)
    text = path.read_text()

    frontmatter_data, body = _split_frontmatter(text)
    sections = _extract_sections(body)

    return ParsedDoc(
        path=str(path),
        frontmatter=frontmatter_data,
        body=body,
        sections=sections,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown file into frontmatter dict and body string.

    Handles YAML that contains markdown-isms (backticks, asterisks)
    by pre-processing problematic values.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw_yaml = match.group(1)
    body = text[match.end():]

    # Pre-process: quote unquoted string values that contain YAML-hostile chars
    # This handles lines like:  - **bold text** or - `code` stuff
    processed = _sanitize_yaml(raw_yaml)

    try:
        data = yaml.safe_load(processed)
    except yaml.YAMLError:
        # Fallback: try line-by-line quoting of problematic list items
        data = _fallback_parse(raw_yaml)

    return data if isinstance(data, dict) else {}, body


def _sanitize_yaml(raw: str) -> str:
    """Make Mittens' markdown-heavy YAML frontmatter safe for parsing.

    Handles two problems:
    1. List items starting with *, `, etc. (YAML aliases/special chars)
    2. Nested sub-bullets under string list items (invalid YAML)
       e.g., exit_criteria items with markdown-style sub-bullets

    Strategy: detect string list items and quote them; flatten sub-bullets
    by appending them to their parent item.
    """
    lines = raw.split("\n")
    result: list[str] = []
    # Track indentation of list contexts to detect sub-bullets
    list_indent_stack: list[int] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith("- "):
            value = stripped[2:].strip()

            # Check if this looks like a nested YAML mapping (id:, name:, etc.)
            is_mapping = (
                ":" in value
                and not value.startswith('"')
                and not value.startswith("`")
                and not value.startswith("*")
                and any(
                    value.startswith(k)
                    for k in ("id:", "name:", "description:", "talents:",
                              "inputs:", "outputs:", "exit_criteria:",
                              "required_skills:", "owner_talent:",
                              "consulting_talents:", "escalates_to:",
                              "type:", "required:", "default:")
                )
            )

            if is_mapping:
                result.append(line)
                list_indent_stack.append(indent)
                i += 1
                continue

            # It's a string list item — collect any sub-bullets
            collected = value
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.lstrip()
                next_indent = len(next_line) - len(next_stripped)
                if next_stripped.startswith("- ") and next_indent > indent:
                    # Sub-bullet — flatten into parent
                    sub_value = next_stripped[2:].strip()
                    collected += " " + sub_value
                    j += 1
                elif not next_stripped:
                    # Blank line — might end the block
                    break
                else:
                    break

            # Strip inline YAML comments (# preceded by whitespace)
            comment_match = re.search(r'\s+#\s', collected)
            if comment_match:
                collected = collected[:comment_match.start()].rstrip()

            # Quote if it contains problematic characters
            needs_quoting = any(
                c in collected for c in ("*", "`", "{", "}", "&", "!")
            ) or collected.startswith(("'", '"', ">", "|"))

            prefix = " " * indent
            if needs_quoting:
                escaped = collected.replace("\\", "\\\\").replace('"', '\\"')
                result.append(f'{prefix}- "{escaped}"')
            else:
                result.append(f"{prefix}- {collected}")
            i = j
            continue

        result.append(line)
        i += 1

    return "\n".join(result)


def _fallback_parse(raw: str) -> dict[str, Any]:
    """Last-resort parser: aggressively quote all string list items."""
    lines = raw.split("\n")
    result = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("- ") and not stripped.startswith("- id:"):
            value = stripped[2:].strip()
            # Skip if it's a mapping (contains : followed by space)
            if ":" in value and not value.startswith('"'):
                # Could be a nested mapping — leave as-is
                result.append(line)
            elif value and not value.startswith('"') and not value.startswith("'"):
                indent = line[: len(line) - len(stripped)]
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                result.append(f'{indent}- "{escaped}"')
            else:
                result.append(line)
        else:
            result.append(line)
    processed = "\n".join(result)
    try:
        data = yaml.safe_load(processed)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def _extract_sections(body: str) -> dict[str, str]:
    """Split markdown body into {heading: content} dict.

    Handles ## and ### headings. Content runs until the next heading
    of same or higher level.
    """
    sections: dict[str, str] = {}
    matches = list(_HEADING_RE.finditer(body))

    for i, match in enumerate(matches):
        heading_level = len(match.group(1))
        heading_text = match.group(2).strip()
        start = match.end()

        # Content runs until next heading of same or higher level, or EOF
        end = len(body)
        for next_match in matches[i + 1 :]:
            next_level = len(next_match.group(1))
            if next_level <= heading_level:
                end = next_match.start()
                break

        content = body[start:end].strip()
        sections[heading_text] = content

    return sections
