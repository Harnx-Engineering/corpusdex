from __future__ import annotations

from corpusdex.chunker import (
    MIN_CHUNK_CHARS,
    chunk_markdown,
    hard_split,
    split_frontmatter,
)

DECISION_DOC = """---
name: sample-decision
decided_on: 2026-01-15
superseded_by: null
tags: [alpha, beta]
---

# Sample decision

## Decision

We decided to do the thing. This section carries enough body text to clear
the minimum chunk size threshold used by the chunker, so it is not folded
into a neighbouring section during the small-section merge pass, which keeps
this test deterministic across changes to that constant elsewhere in the
module. Extra padding continues here to stay safely above the floor.

## Why

Because reasons, explained in a paragraph that is also long enough to
survive the minimum-size merge step without being folded into the section
above it, keeping the two chunks distinct and independently checkable for
heading-path breadcrumb correctness after the split.
"""

NO_FRONTMATTER_DOC = """# Plain document

## First section

Some content here that is long enough on its own to survive the minimum
chunk size merge pass without needing to borrow text from a neighbour, so
this test can assert on an exact heading path without surprises.

## Second section

### Nested subsection

A nested H3 section under an H2, long enough to stand alone above the
minimum chunk size threshold so the breadcrumb assertion for three levels
of heading is exercised deterministically by this test.
"""


def test_split_frontmatter_present():
    import datetime

    frontmatter, body = split_frontmatter(DECISION_DOC)
    assert frontmatter["name"] == "sample-decision"
    # YAML parses an unquoted ISO date scalar as a date object; chunk_markdown
    # (not split_frontmatter) is responsible for stringifying it.
    assert frontmatter["decided_on"] == datetime.date(2026, 1, 15)
    assert body.startswith("\n# Sample decision")


def test_split_frontmatter_absent():
    frontmatter, body = split_frontmatter(NO_FRONTMATTER_DOC)
    assert frontmatter == {}
    assert body == NO_FRONTMATTER_DOC


def test_split_frontmatter_malformed_is_ignored():
    text = "---\nkey: [1, 2\nkey2: 3\n---\n\n# Title\n"
    frontmatter, body = split_frontmatter(text)
    assert frontmatter == {}
    assert body == text


def test_chunk_markdown_splits_on_h2_headings():
    parsed = chunk_markdown(DECISION_DOC, fallback_title="fallback")
    assert len(parsed.chunks) == 2
    assert parsed.chunks[0].heading_path == "Sample decision > Decision"
    assert parsed.chunks[1].heading_path == "Sample decision > Why"
    assert "We decided to do the thing" in parsed.chunks[0].body
    assert "Because reasons" in parsed.chunks[1].body


def test_chunk_markdown_tracks_h1_h2_h3_breadcrumb():
    parsed = chunk_markdown(NO_FRONTMATTER_DOC, fallback_title="fallback")
    heading_paths = [c.heading_path for c in parsed.chunks]
    assert "Plain document > First section" in heading_paths
    assert "Plain document > Second section > Nested subsection" in heading_paths


def test_chunk_markdown_frontmatter_becomes_metadata():
    parsed = chunk_markdown(DECISION_DOC, fallback_title="fallback")
    assert parsed.title == "sample-decision"
    assert parsed.decided_on == "2026-01-15"
    assert parsed.superseded_by is None
    assert parsed.tags == "alpha,beta"


def test_chunk_markdown_no_frontmatter_has_no_decision_metadata():
    parsed = chunk_markdown(NO_FRONTMATTER_DOC, fallback_title="fallback")
    assert parsed.title == "Plain document"
    assert parsed.decided_on is None
    assert parsed.superseded_by is None
    assert parsed.tags is None


def test_chunk_markdown_falls_back_to_h1_title_when_no_frontmatter_title():
    text = "# Real Title\n\n## Section\n\n" + ("body text " * 20)
    parsed = chunk_markdown(text, fallback_title="fallback-title")
    assert parsed.title == "Real Title"


def test_chunk_markdown_uses_fallback_title_when_nothing_else_present():
    text = "## Section only\n\n" + ("body text " * 20)
    parsed = chunk_markdown(text, fallback_title="fallback-title")
    assert parsed.title == "fallback-title"


def test_small_sections_merge_into_neighbour():
    text = (
        "# Doc\n\n"
        "## Tiny\n\ntoo short\n\n"
        "## Real section\n\n" + ("substantial body content here. " * 20)
    )
    parsed = chunk_markdown(text, fallback_title="fallback")
    # The tiny section must not survive as its own standalone chunk.
    bodies = [c.body for c in parsed.chunks]
    assert not any(b.strip() == "too short" for b in bodies)
    joined = "\n".join(bodies)
    assert "too short" in joined
    assert "substantial body content" in joined


def test_hard_split_oversized_section_produces_overlapping_pieces():
    long_body = "\n".join(f"line {i} of a very long section body" for i in range(400))
    pieces = hard_split(long_body)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) > 0
    # Consecutive pieces share at least one overlapping line so a fact
    # straddling the split boundary is retrievable from either piece.
    first_lines = set(pieces[0].splitlines())
    second_lines = set(pieces[1].splitlines())
    assert first_lines & second_lines


def test_hard_split_leaves_small_body_untouched():
    small_body = "just one short section"
    assert hard_split(small_body) == [small_body]


def test_fenced_code_block_headings_are_not_split():
    text = (
        "# Doc\n\n"
        "## Section\n\n"
        "Some intro text that is long enough to clear the minimum chunk "
        "size on its own without merging into a neighbour section here.\n\n"
        "```markdown\n"
        "# not a real heading\n"
        "## also not a heading\n"
        "```\n\n"
        "More trailing content after the fence closes out this section body."
    )
    parsed = chunk_markdown(text, fallback_title="fallback")
    heading_paths = {c.heading_path for c in parsed.chunks}
    assert heading_paths == {"Doc > Section"}
    assert len(parsed.chunks) == 1
    assert "# not a real heading" in parsed.chunks[0].body


def test_min_chunk_chars_constant_is_positive():
    # Sanity check the constant this test file's fixtures are sized against.
    assert MIN_CHUNK_CHARS > 0
