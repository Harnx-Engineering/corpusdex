"""Heading-aware Markdown chunking.

A chunk is one Markdown section: the text under an H2 or H3 heading, carrying a
breadcrumb (``H1 > H2 > H3``) as its ``heading_path``. Sections too small to
stand alone are merged into a neighbour, and sections too large for a retrieval
window are hard-split with an overlap so a fact straddling a split boundary is
still retrievable from at least one piece.

Token counts are approximated as ``len(text) / 4``: good enough for sizing and
free of a tokenizer dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

TOKEN_CHARS = 4
MAX_CHUNK_TOKENS = 800
MIN_CHUNK_TOKENS = 50
OVERLAP_RATIO = 0.15

MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * TOKEN_CHARS
MIN_CHUNK_CHARS = MIN_CHUNK_TOKENS * TOKEN_CHARS
OVERLAP_CHARS = int(MAX_CHUNK_CHARS * OVERLAP_RATIO)

BREADCRUMB_SEP = " > "

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL
)
_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*\s*$")


def approx_tokens(text: str) -> int:
    """Approximate the token count of ``text`` (4 characters per token)."""
    return (len(text) + TOKEN_CHARS - 1) // TOKEN_CHARS


@dataclass(frozen=True)
class Chunk:
    heading_path: str
    body: str


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    chunks: tuple[Chunk, ...]
    decided_on: str | None = None
    superseded_by: str | None = None
    tags: str | None = None
    frontmatter: dict[str, object] = field(default_factory=dict)


def split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split leading YAML frontmatter from the Markdown body.

    Returns an empty mapping (and the untouched text) when there is no
    frontmatter or the YAML is unparseable.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(loaded, dict):
        return {}, text
    return loaded, text[match.end() :]


def _clean_scalar(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "~"}:
        return None
    return text


def _clean_tags(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(",", " ").split()]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(p).strip() for p in value]
    else:
        parts = [str(value).strip()]
    parts = [p for p in parts if p]
    return ",".join(parts) if parts else None


def _iter_lines(text: str) -> list[tuple[str, int | None, str]]:
    """Yield ``(line, heading_level, heading_text)`` with fenced blocks masked."""
    out: list[tuple[str, int | None, str]] = []
    fence: str | None = None
    for line in text.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence is None and fence_match is not None:
            fence = fence_match.group(1)[0]
            out.append((line, None, ""))
            continue
        if fence is not None:
            if fence_match is not None and fence_match.group(1)[0] == fence:
                fence = None
            out.append((line, None, ""))
            continue
        heading = _HEADING_RE.match(line)
        if heading is not None:
            out.append((line, len(heading.group(1)), heading.group(2).strip()))
        else:
            out.append((line, None, ""))
    return out


def _overlap_tail(lines: list[str]) -> list[str]:
    """Return the trailing lines of ``lines`` fitting the overlap budget.

    Always leaves at least one line behind so splitting makes forward progress.
    """
    tail: list[str] = []
    total = 0
    for line in reversed(lines[1:]):
        total += len(line) + 1
        if total > OVERLAP_CHARS:
            break
        tail.append(line)
    tail.reverse()
    return tail


def _split_long_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if len(line) <= MAX_CHUNK_CHARS:
            out.append(line)
            continue
        for start in range(0, len(line), MAX_CHUNK_CHARS):
            out.append(line[start : start + MAX_CHUNK_CHARS])
    return out


def hard_split(body: str) -> list[str]:
    """Split an oversized section body into overlapping pieces."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in _split_long_lines(body.splitlines()):
        cost = len(line) + 1
        if current and current_len + cost > MAX_CHUNK_CHARS:
            pieces.append("\n".join(current))
            current = _overlap_tail(current)
            current_len = sum(len(item) + 1 for item in current)
        current.append(line)
        current_len += cost
    if current:
        pieces.append("\n".join(current))
    return pieces


def _sections(body: str, title: str) -> list[Chunk]:
    """Split the body on H2/H3 headings, tracking the H1/H2/H3 breadcrumb."""
    sections: list[Chunk] = []
    h1 = ""
    h2 = ""
    current_heading = title
    current: list[str] = []

    def flush() -> None:
        text = "\n".join(current).strip()
        if text:
            sections.append(Chunk(heading_path=current_heading, body=text))
        current.clear()

    for line, level, heading in _iter_lines(body):
        if level == 1:
            h1 = heading
            h2 = ""
            current.append(line)
            continue
        if level in (2, 3):
            flush()
            if level == 2:
                h2 = heading
                parts = [h1, h2]
            else:
                parts = [h1, h2, heading]
            current_heading = BREADCRUMB_SEP.join(p for p in parts if p) or title
            current.append(line)
            continue
        current.append(line)
    flush()
    return sections


def _merge_small(sections: list[Chunk]) -> list[Chunk]:
    """Fold sections below the minimum size into a neighbour.

    Backward merge is preferred; a leading small section merges forward. Either
    way the surviving chunk keeps the neighbour's heading path, because the
    neighbour supplies the bulk of the text.
    """
    merged: list[Chunk] = []
    pending: list[str] = []
    for section in sections:
        body = "\n\n".join([*pending, section.body]) if pending else section.body
        pending = []
        if len(body.strip()) < MIN_CHUNK_CHARS:
            if merged:
                previous = merged[-1]
                merged[-1] = Chunk(
                    heading_path=previous.heading_path,
                    body=f"{previous.body}\n\n{body}",
                )
            else:
                pending = [body]
            continue
        merged.append(Chunk(heading_path=section.heading_path, body=body))
    if pending:
        leftover = "\n\n".join(pending)
        if merged:
            previous = merged[-1]
            merged[-1] = Chunk(
                heading_path=previous.heading_path,
                body=f"{previous.body}\n\n{leftover}",
            )
        else:
            merged.append(Chunk(heading_path=sections[0].heading_path, body=leftover))
    return merged


def chunk_markdown(text: str, *, fallback_title: str) -> ParsedDocument:
    """Parse frontmatter and chunk a Markdown document."""
    frontmatter, body = split_frontmatter(text)

    title = _clean_scalar(frontmatter.get("title")) or _clean_scalar(frontmatter.get("name"))
    if title is None:
        for _line, level, heading in _iter_lines(body):
            if level == 1 and heading:
                title = heading
                break
    title = title or fallback_title

    sections = _merge_small(_sections(body, title))
    chunks: list[Chunk] = []
    for section in sections:
        for piece in hard_split(section.body):
            if piece.strip():
                chunks.append(Chunk(heading_path=section.heading_path, body=piece))

    return ParsedDocument(
        title=title,
        chunks=tuple(chunks),
        decided_on=_clean_scalar(frontmatter.get("decided_on")),
        superseded_by=_clean_scalar(frontmatter.get("superseded_by")),
        tags=_clean_tags(frontmatter.get("tags")),
        frontmatter=frontmatter,
    )
