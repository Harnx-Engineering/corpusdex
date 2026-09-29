from __future__ import annotations

from corpusdex.chunker import (
    LINE_SPLIT_CHARS,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    MIN_OVERLAP_CHARS,
    OVERLAP_CHARS,
    WORD_BOUNDARY_WINDOW,
    _overlap_tail,
    _split_long_lines,
    _tail_from_word_start,
    _word_split_point,
    chunk_markdown,
    hard_split,
    split_frontmatter,
)


def _longest_shared_join(left: str, right: str) -> int:
    """Length of the longest suffix of ``left`` that is a prefix of ``right``."""
    for size in range(min(len(left), len(right)), 0, -1):
        if right[:size] == left[-size:]:
            return size
    return 0


#: One dense bullet per line, which is how this corpus is actually written and
#: the input class the pre-existing overlap tests never exercised: their lines
#: are ~35 characters, comfortably inside the overlap budget, so the defect in
#: issue #70 could not appear in them. Sized against the real corpus, whose
#: p99 line is 916 characters and whose longest is 4212, with 171 lines past
#: the 480-character overlap budget.
DENSE_BULLET = (
    "- the chunker documents a guarantee that a fact straddling a split boundary "
    "stays retrievable from at least one piece, and the budget check charged the "
    "cost of a line before deciding whether to take it and then gave up rather "
    "than trimming, so a single bullet longer than the whole overlap budget "
    "carried no overlap at all instead of a shortened one, which on a corpus "
    "written as one dense bullet per line is the normal case and not an edge "
    "case, and it hides text the markdown plainly contains from every lexical "
    "query that happens to cross that particular boundary, leaving the caller "
    "unable to tell a fact that was never recorded from one recorded and made "
    "unretrievable by the index derived from it"
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


def test_a_line_too_long_to_carry_whole_contributes_its_tail():
    """The core of issue #70. The budget was charged before the line was taken
    and the loop broke rather than trimming, so ONE trailing line over budget
    yielded an empty overlap, not a shortened one."""
    carried = _overlap_tail(["a preceding line", DENSE_BULLET])
    assert carried, "a line over budget must contribute its tail, not nothing"
    assert sum(len(piece) for piece in carried) <= OVERLAP_CHARS
    assert DENSE_BULLET.endswith(carried[-1]), "the overlap must be a real suffix"


def test_the_overlap_fragment_opens_at_a_word_start():
    """A fragment cut mid-word reaches FTS as a token present in no document;
    issue #70 minted `utright` from `outright` exactly this way."""
    carried = _overlap_tail(["a preceding line", DENSE_BULLET])
    fragment = carried[-1]
    prefix = DENSE_BULLET[: len(DENSE_BULLET) - len(fragment)]
    assert prefix.endswith((" ", "\t")), "the character before the fragment must be a break"
    assert not fragment[0].isspace()


def test_an_overlap_too_small_to_hold_a_fact_is_refused_rather_than_minted():
    """The mirror direction of the fix: trimming must not degrade into carrying
    a few characters. A fragment shorter than a sentence cannot hold a
    straddling fact, so it is vocabulary and storage bought for nothing.

    The floor is checked on what is actually CARRIED, not on the budget. The
    budget here is the full 480, and the only word start inside it leaves ten
    characters, so a budget-side check would pass this and mint the fragment
    anyway."""
    line = "x" * 500 + " " + "y" * 10
    assert _overlap_tail([line]) == []


def test_a_fragment_at_the_floor_is_carried():
    """The other side of the same boundary, so the floor cannot be satisfied by
    refusing everything."""
    line = "x" * 500 + " " + "y" * MIN_OVERLAP_CHARS
    carried = _overlap_tail([line])
    assert carried == ["y" * MIN_OVERLAP_CHARS]


def test_a_line_with_no_word_boundary_carries_no_overlap():
    """Not a regression: an unbroken 600-character run is one token, so every
    candidate fragment would be a piece of a word. Refusing is the honest
    answer, and this test pins it so a later change cannot quietly start
    minting fragments of hashes and URLs."""
    assert _overlap_tail(["head", "x" * 600]) == []


def test_overlap_never_consumes_the_whole_of_what_preceded_it():
    """Forward progress, which the joined-text form has to earn differently:
    excluding the first line used to guarantee it, and now the guarantee is
    that the overlap is a PROPER suffix of what preceded it."""
    for lines in ([DENSE_BULLET], [DENSE_BULLET, DENSE_BULLET]):
        text = "\n".join(lines)
        carried = "\n".join(_overlap_tail(lines))
        # Non-empty is asserted first. Allowing "" here would let the old
        # single-line behaviour, which returned nothing at all, satisfy a test
        # named for progress.
        assert carried, "prose longer than the floor must carry an overlap"
        assert carried != text, "an overlap equal to its source makes no progress"
        assert text.endswith(carried)
    # Too short to reach the floor: refusing is the correct answer, not a
    # violation of the progress rule.
    assert _overlap_tail(["a", "b", "c"]) == []


def test_every_hard_split_boundary_of_a_dense_document_carries_overlap():
    """The guarantee in the module docstring, asserted directly and on the
    input class that breaks it. The pre-existing test used 35-character lines,
    where the defect cannot appear."""
    body = "\n".join(f"- bullet {i}: {DENSE_BULLET}" for i in range(20))
    pieces = hard_split(body)
    assert len(pieces) > 2, "the fixture must actually split several times"
    for left, right in zip(pieces, pieces[1:], strict=False):
        shared = _longest_shared_join(left, right)
        assert shared >= MIN_OVERLAP_CHARS, (
            f"boundary carries only {shared} characters of overlap"
        )


def test_splitting_a_long_line_does_not_cut_inside_a_word():
    """A real corpus file carried a 4212-character line whose cut landed
    inside `outright`, so one of its two occurrences was not retrievable and a
    junk token entered the vocabulary in its place."""
    line = "alpha beta gamma delta " * 400
    fragments = _split_long_lines([line])
    assert len(fragments) > 1
    for left, right in zip(fragments, fragments[1:], strict=False):
        assert left.endswith((" ", "\t")) or right[0].isspace(), (
            f"cut landed mid-word: {left[-12:]!r} | {right[:12]!r}"
        )


def test_splitting_a_long_line_is_a_partition():
    """Nothing is dropped or duplicated by the cut, so the word-boundary search
    cannot become a way to lose text."""
    line = "alpha beta gamma delta " * 400
    assert "".join(_split_long_lines([line])) == line


def test_an_unbroken_run_falls_back_to_a_hard_cut():
    """The word-boundary search is bounded. Base64 or a long hash has no
    boundary to find, and the fragment must still be produced rather than the
    search walking back to the start of the line."""
    line = "z" * (LINE_SPLIT_CHARS * 2)
    fragments = _split_long_lines([line])
    assert len(fragments) == 2
    assert len(fragments[0]) == LINE_SPLIT_CHARS
    assert "".join(fragments) == line


def test_the_word_boundary_search_gives_up_only_a_bounded_prefix():
    """A cut may move to reach whitespace, but not far enough to shrink a chunk
    materially: a distant boundary would trade mid-word tokens for chunks of
    arbitrary size."""
    head = "q" * (LINE_SPLIT_CHARS - WORD_BOUNDARY_WINDOW - 50)
    line = head + " " + "r" * LINE_SPLIT_CHARS
    fragments = _split_long_lines([line])
    assert len(fragments[0]) >= LINE_SPLIT_CHARS - WORD_BOUNDARY_WINDOW


def test_a_boundary_created_by_splitting_one_long_line_carries_overlap():
    """The case the first attempt at this fix missed entirely. When a section
    is ONE oversized line, every boundary in it comes from fragmenting that
    line, and each fragment is a single line: taking overlap only from whole
    lines other than the first therefore had nothing to give at exactly the
    boundaries the issue is about. The 4212-character line above is this shape."""
    # Every word distinct, deliberately. A repetitive fixture ("x " repeated)
    # makes the two pieces share a 476-character suffix and prefix BY
    # COINCIDENCE even with the overlap removed entirely, so the assertion
    # below would hold against code that carries no overlap at all. The
    # measurement has to be impossible to satisfy by accident.
    line = (
        " ".join(f"w{i:04d}" for i in range(700))
        + " alpha beta "
        + " ".join(f"z{i:04d}" for i in range(60))
    )
    pieces = hard_split(line)
    assert len(pieces) > 1, "the fixture must actually split"
    # Asserting the OVERLAP, not merely that the phrase survived. Checking only
    # for "alpha beta" passes with overlap removed entirely, because moving the
    # cut to a word boundary already happens to leave that phrase intact: the
    # test would then be measuring the other half of this change.
    for left, right in zip(pieces, pieces[1:], strict=False):
        assert _longest_shared_join(left, right) >= MIN_OVERLAP_CHARS, (
            "a boundary inside one long line must carry overlap like any other"
        )
    assert any("alpha beta" in piece for piece in pieces)


def test_no_piece_exceeds_the_chunk_ceiling():
    """Overlap is prepended to a piece and then lines are added on top, so the
    ceiling only holds if a fragment leaves room for a full overlap. Splitting
    long lines at MAX_CHUNK_CHARS left none, and pieces ran over."""
    body = "\n".join(["a" * 2500, "word " * 119 + "tail", "z" * 3199 + " " + "q" * 100])
    for piece in hard_split(body):
        assert len(piece) <= MAX_CHUNK_CHARS, f"piece of {len(piece)} exceeds the ceiling"
    dense = "\n".join(f"- bullet {i}: {DENSE_BULLET}" for i in range(20))
    for piece in hard_split(dense):
        assert len(piece) <= MAX_CHUNK_CHARS


def test_a_usable_suffix_is_not_refused_because_its_only_space_is_distant():
    """The forward scan must not be bounded by WORD_BOUNDARY_WINDOW. Bounding
    it made a 199-character suffix, comfortably above the floor, unreachable
    because the space that opens it sat outside a window that exists for the
    unrelated backward search."""
    carried = _overlap_tail(["a" * 400 + " " + "b" * 199])
    assert carried == ["b" * 199]


def test_a_whitespace_only_tail_carries_no_overlap():
    """A fragment made of spaces opens at no word and carries no fact."""
    assert _overlap_tail([" " * 600]) == []
    assert _overlap_tail(["q" * 200 + " " * 480]) == []


def test_the_split_helpers_are_total_on_degenerate_input():
    """A helper that raises on an unremarkable argument is a trap for the next
    caller, even when today's call sites happen to stay inside the safe range."""
    assert _word_split_point("", 1) == 0
    assert _word_split_point("abc", 0) == 0
    assert _word_split_point("abc", -5) == 0
    assert _word_split_point("abc", 99) == 3
    assert _tail_from_word_start("", 10) == ""
    assert _tail_from_word_start("abc", 0) == ""
    assert _tail_from_word_start("abc", -1) == ""


def test_the_boundary_immediately_before_the_window_is_considered():
    """The scan starts one character before the budget window, not at it.

    When the character just outside the window is whitespace, the text from
    the window's first character is already a full-size fragment opening at a
    word: the single best answer available. Starting the scan at the window
    itself skips it and settles for the next boundary inside, which here does
    not exist, so the overlap collapses to nothing rather than shrinking."""
    line = "abc " + "z" * OVERLAP_CHARS
    assert _overlap_tail([line]) == ["z" * OVERLAP_CHARS]


def test_the_split_limit_cannot_be_tuned_down_to_a_spin():
    """`LINE_SPLIT_CHARS` is derived from `OVERLAP_RATIO`, which lives a few
    lines away and looks like a free tuning knob. A ratio near 1.0 drives the
    limit to zero or below, and a limit of zero makes the split loop append an
    empty fragment and leave the remainder untouched: it hangs rather than
    failing. The floor keeps every cut at one character or more."""
    assert LINE_SPLIT_CHARS >= 1
    for limit in (1, 2, 5):
        assert _word_split_point("abcdefghij", limit) >= 1, "a cut must advance"
    assert _word_split_point("abcdefghij", 0) == 0


def _lines_inside_a_fence(piece: str) -> set[str]:
    """The lines a renderer reads as code when ``piece`` is read on its own."""
    inside: set[str] = set()
    opener = None
    for line in piece.splitlines():
        mark = next((m for m in ("```", "~~~") if line.lstrip().startswith(m)), None)
        if mark is not None and (opener is None or opener == mark):
            opener = None if opener else mark
        elif opener is not None:
            inside.add(line)
    return inside


def test_a_piece_can_fill_the_ceiling_exactly():
    """Lengths are those of the joined text. Charging a newline per line
    rather than per join ran one ahead of the text, so no piece ever reached
    MAX_CHUNK_CHARS and the longest was 3199 (issue #85)."""
    assert OVERLAP_CHARS + 1 + LINE_SPLIT_CHARS == MAX_CHUNK_CHARS
    body = "\n".join(["a" * 1599, "b" * 1600, "c" * 1600])
    pieces = hard_split(body)
    assert pieces[0] == "a" * 1599 + "\n" + "b" * 1600
    assert len(pieces[0]) == MAX_CHUNK_CHARS


def test_a_piece_starting_inside_a_fence_reopens_it():
    """A piece that begins inside a fenced block would read the block's
    closing delimiter as an opening one, and the prose after it as code."""
    code = [f"line {i} = compute({i}) + other_value_{i}" for i in range(200)]
    prose = "Closing prose after the block."
    body = "Intro.\n\n```python\n" + "\n".join(code) + "\n```\n\n" + prose
    pieces = hard_split(body)
    assert len(pieces) > 2
    for piece in pieces:
        assert len(piece) <= MAX_CHUNK_CHARS
        inside = _lines_inside_a_fence(piece)
        assert prose not in inside, "prose after the block must not render as code"
        whole = [line for line in piece.splitlines() if line in code]
        assert set(whole) <= inside, "code lines must render as code"
    for piece in pieces[1:]:
        assert piece.startswith("```python\n"), "the info string is kept"


def test_a_tilde_line_inside_a_backtick_block_does_not_close_it():
    code = "\n".join(["~~~ not a closer"] + [f"x_{i} = {i} * 31337" for i in range(300)])
    body = "```\n" + code + "\n```\nafter"
    for piece in hard_split(body)[1:]:
        assert piece.startswith("```\n"), piece[:30]


def test_a_reopened_fence_keeps_maximal_fragments_under_the_ceiling():
    """The re-emitted fence takes its room from the overlap budget. Added on
    top instead, a full overlap, the fence and one maximal fragment made a
    piece of 3403 characters."""
    opener = "```" + "i" * 200
    # Word fragments, so each piece carries a real overlap for the fence to
    # compete with; a run with no spaces yields no overlap at all.
    fragment = ("word " * 1000)[: LINE_SPLIT_CHARS - 1] + "w"
    body = opener + "\n" + "\n".join(fragment for _ in range(4)) + "\n```"
    pieces = hard_split(body)
    assert sum(piece.count(fragment) for piece in pieces) == 4
    for piece in pieces:
        assert len(piece) <= MAX_CHUNK_CHARS
    for piece in pieces[1:]:
        assert piece.startswith(opener + "\n")
