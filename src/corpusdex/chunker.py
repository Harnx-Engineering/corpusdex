"""Heading-aware Markdown chunking.

A chunk is one Markdown section: the text under an H2 or H3 heading, carrying a
breadcrumb (``H1 > H2 > H3``) as its ``heading_path``. Sections too small to
stand alone are merged into a neighbour, and sections too large for a retrieval
window are hard-split with an overlap so a fact straddling a split boundary is
still retrievable from at least one piece. The one exception is text holding no
word boundary to start an overlap at, such as a long hash or base64 blob: there
every candidate overlap would be a piece of a single token, and carrying one
would mint vocabulary into the index that appears in no document.

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

#: Shortest overlap worth carrying. The guarantee is about a straddling FACT,
#: and a fragment shorter than a sentence cannot hold one, so below this the
#: overlap only costs storage and mints vocabulary into the FTS index without
#: buying any retrievability.
MIN_OVERLAP_CHARS = TOKEN_CHARS * 20

#: How far a BACKWARD cut may move to land on whitespace instead of inside a
#: word. Prose puts a space every few characters, so this is never the binding
#: constraint on real text; it bounds the damage on input that is NOT prose,
#: where an unbroken run of this length means a hash, a URL or base64 and a
#: hard cut is the more honest answer than a fragment of arbitrary size.
#:
#: It deliberately does NOT bound the forward search in
#: :func:`_tail_from_word_start`. There, moving further only shortens the
#: overlap, and ``MIN_OVERLAP_CHARS`` already refuses a fragment that has
#: become too short to be worth carrying. Bounding it twice made a usable
#: 199-character suffix unreachable because its only space sat outside an
#: unrelated window.
WORD_BOUNDARY_WINDOW = 200

#: The longest fragment a single oversized line is broken into.
#:
#: Sized so a full overlap, the newline joining it to the next line, and one
#: fragment fit one chunk exactly: ``OVERLAP_CHARS + 1 + LINE_SPLIT_CHARS``
#: equals ``MAX_CHUNK_CHARS``. ``hard_split`` measures the JOINED text, so this
#: is the text's own length. It used to charge one newline per line rather than
#: one per join, running one ahead of the text, and this constant subtracted 2
#: to compensate, so the longest piece was 3199 characters (issue #85).
#: Splitting at ``MAX_CHUNK_CHARS`` instead left no room,
#: so the overlap was added on top of an already-full chunk and pieces ran
#: over the retrieval window they are supposed to fit.
#:
#: Floored at 1. It is a DERIVED constant, and raising ``OVERLAP_RATIO`` far
#: enough drives it to zero or below, at which point the split loop asks for a
#: cut of 0, appends an empty fragment and leaves the remainder unchanged:
#: it spins forever rather than failing. A tuning knob one file away should
#: not be able to hang the indexer.
LINE_SPLIT_CHARS = max(1, MAX_CHUNK_CHARS - OVERLAP_CHARS - 1)

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


def _tail_from_word_start(text: str, size: int) -> str:
    """Return at most ``size`` trailing characters of ``text``, beginning at a
    word start.

    Taking ``text[-size:]`` directly would open the fragment with half a word,
    which reaches the FTS index as a token that appears in no document
    (issue #70 minted ``utright`` this way).

    The scan looks for a whitespace character followed by a non-whitespace one,
    so the fragment begins at a real word and never at a run of spaces: a line
    padded with trailing whitespace would otherwise yield an "overlap" made
    entirely of spaces. Returns "" when no such boundary exists, since every
    candidate fragment would then be a piece of one unbroken token.

    The caller decides whether what comes back is long enough to be worth
    carrying; this only guarantees where it starts.
    """
    if size <= 0 or not text:
        return ""
    # Scanning from ``start - 1`` rather than ``start``: when the character
    # just before the window is whitespace, ``text[start:]`` is already a
    # full-size fragment opening at a word, and starting one later would skip
    # the single best answer and settle for a shorter one.
    start = max(0, len(text) - size - 1)
    for index in range(start, len(text) - 1):
        if text[index].isspace() and not text[index + 1].isspace():
            return text[index + 1 :]
    return ""


def _overlap_tail(lines: list[str], budget: int = OVERLAP_CHARS) -> list[str]:
    """Return the trailing text of ``lines`` fitting the overlap budget.

    The overlap is taken from the JOINED text rather than a whole number of
    lines. Working line by line meant a piece whose whole content was one
    oversized line had nothing to give: ``lines[0]`` was excluded to guarantee
    forward progress, and with a single line there was nothing else left. That
    is not a rare shape, it is what every fragment of a hard-split long line
    looks like, so the boundaries created by splitting a long line were exactly
    the ones that carried no overlap (issue #70).

    Progress is guaranteed by the word-start scan rather than by arithmetic:
    a fragment must begin after a whitespace character, so at least that
    character is left behind and the overlap is always a PROPER suffix. The
    budget is capped one below the length as well, but that cap is a cheap
    explicit bound, not the thing doing the work; mutating it away changes no
    behaviour.

    Returns [] rather than a short fragment when what fits is below
    ``MIN_OVERLAP_CHARS``. The floor is checked on the RESULT and only there.
    Checking the remaining budget instead let a fragment far below the floor
    through, because the scan shortens the fragment after the budget has
    already been approved: a budget of exactly the floor yielded a
    one-character overlap.
    """
    text = "\n".join(lines)
    piece = _tail_from_word_start(text, min(budget, len(text) - 1))
    if len(piece) < MIN_OVERLAP_CHARS:
        return []
    return piece.split("\n")


def _split_long_lines(lines: list[str]) -> list[str]:
    """Break over-long lines, cutting between words where possible.

    The threshold is ``LINE_SPLIT_CHARS``, which is BELOW the chunk ceiling: a
    line is broken once it is too long to sit in a chunk alongside a full
    overlap, not merely once it is too long for a chunk by itself.

    The cut is a partition: concatenating the fragments of a line reproduces it
    exactly, so nothing is dropped or duplicated by this step.
    """
    out: list[str] = []
    for line in lines:
        rest = line
        while len(rest) > LINE_SPLIT_CHARS:
            cut = _word_split_point(rest, LINE_SPLIT_CHARS)
            out.append(rest[:cut])
            rest = rest[cut:]
        out.append(rest)
    return out


def _word_split_point(text: str, limit: int) -> int:
    """Return where to cut ``text``, at or before ``limit``.

    Searches backwards for whitespace so the cut falls between words, and
    keeps the whitespace on the left side so the split stays a partition.
    Falls back to ``limit`` when there is no boundary within the window.

    Total for any input: a limit at or past the end of ``text``, or at or
    below zero, returns a cut that leaves the string intact rather than
    indexing out of bounds. Callers only ever pass a limit inside the string,
    but a helper that raises on an unremarkable argument is a trap for the
    next caller.
    """
    limit = min(limit, len(text))
    if limit <= 0:
        return 0
    floor = max(1, limit - WORD_BOUNDARY_WINDOW)
    for index in range(limit, floor - 1, -1):
        if text[index - 1].isspace():
            return index
    return limit


def _fence_states(lines: list[str]) -> list[str | None]:
    """For each line, the opening fence line in force BEFORE it, or None.

    The same open/close rule as :func:`_iter_lines`: a fence closes only on a
    delimiter of its own kind, so a ``~~~`` line inside a backtick block is
    content.
    """
    states: list[str | None] = []
    opener: str | None = None
    for line in lines:
        states.append(opener)
        match = _FENCE_RE.match(line)
        if match is None:
            continue
        if opener is None:
            opener = line
        elif match.group(1)[0] == _FENCE_RE.match(opener).group(1)[0]:
            opener = None
    return states


def _joined_len(lines: list[str]) -> int:
    return sum(len(line) for line in lines) + max(0, len(lines) - 1)


def _carry_into_next_piece(
    entries: list[tuple[str, str | None]], next_state: str | None
) -> list[tuple[str, str | None]]:
    """Return the start of the next piece: the overlap, re-opened if needed.

    A piece that begins inside a fenced block would read that block's closing
    delimiter as an OPENING one, and everything after it would render as code
    (issue #85). That happens whether the piece begins with an overlap or with
    the next line, so the fence line in force is re-emitted first, with its
    info string, keeping the overlap contiguous.

    The re-emitted line takes its room out of the overlap budget rather than
    being added on top. Added on top, a full overlap plus the fence plus one
    maximal fragment is a piece over the ceiling.
    """
    lines = [line for line, _ in entries]
    tail = _overlap_tail(lines)
    state = _tail_state(entries, tail, next_state)
    if state is not None:
        tail = _overlap_tail(lines, OVERLAP_CHARS - len(state) - 1)
        state = _tail_state(entries, tail, next_state)
    carried = [(line, None) for line in tail]
    if tail:
        # The first carried line keeps the state of the source line it is a
        # suffix of; the rest are whole lines and keep their own.
        offset = len(entries) - len(tail)
        carried = [(line, entries[offset + i][1]) for i, line in enumerate(tail)]
    if state is not None:
        carried.insert(0, (state, None))
    return carried


def _tail_state(
    entries: list[tuple[str, str | None]], tail: list[str], next_state: str | None
) -> str | None:
    """The fence in force where ``tail`` begins, or where the next line does."""
    if not tail:
        return next_state
    return entries[len(entries) - len(tail)][1]


def hard_split(body: str) -> list[str]:
    """Split an oversized section body into overlapping pieces.

    Lengths are those of the JOINED text: each line after the first costs its
    length plus the one newline that joins it.
    """
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    source = body.splitlines()
    states = _fence_states(source)
    lines: list[tuple[str, str | None]] = []
    for line, state in zip(source, states, strict=True):
        # Every fragment of one source line sits under that line's fence; only
        # a whole line can open or close one.
        lines.extend((fragment, state) for fragment in _split_long_lines([line]))
    pieces: list[str] = []
    current: list[tuple[str, str | None]] = []
    for line, state in lines:
        if current and _joined_len([c for c, _ in current]) + 1 + len(line) > MAX_CHUNK_CHARS:
            pieces.append("\n".join(c for c, _ in current))
            current = _carry_into_next_piece(current, state)
        current.append((line, state))
    if current:
        pieces.append("\n".join(c for c, _ in current))
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
