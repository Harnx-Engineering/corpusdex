from __future__ import annotations

import time

import pytest

from corpusdex import db, search


def _insert_doc_with_chunk(
    conn,
    *,
    path: str,
    body: str,
    heading_path: str = "Title > Section",
    decided_on: str | None = None,
    superseded_by: str | None = None,
    superseded_by_doc_id: int | None = None,
    mtime: float | None = None,
) -> int:
    """Insert one document with one chunk; return the CHUNK id.

    ``superseded_by`` writes the raw frontmatter string, which is what the
    result payload displays. ``superseded_by_doc_id`` writes the resolved
    ``doc_links`` edge, which is what the ranking penalty reads. They are
    separate parameters on purpose: keeping them separable is what lets a
    test pin the case where a record claims supersedence that resolved to
    nothing (issue #21).
    """
    if mtime is None:
        mtime = time.time()
    with conn:
        cur = conn.execute(
            "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
            "VALUES ('repo', ?, 'Title', 'doc', ?, ?)",
            (path, mtime, f"hash-{path}"),
        )
        doc_id = cur.lastrowid
        chunk_cur = conn.execute(
            "INSERT INTO chunks (ref, doc_id, heading_path, body, decided_on, superseded_by) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                db.chunk_ref(path, heading_path, 0),
                doc_id,
                heading_path,
                body,
                decided_on,
                superseded_by,
            ),
        )
        if superseded_by_doc_id is not None:
            conn.execute(
                "INSERT INTO doc_links (src_doc_id, dst_doc_id, relation) VALUES (?, ?, ?)",
                (doc_id, superseded_by_doc_id, "superseded_by"),
            )
    return chunk_cur.lastrowid


def _doc_id_of(conn, path: str) -> int:
    return conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()["id"]


def _insert_vector(conn, chunk_id: int, value: float = 0.1) -> None:
    import sqlite_vec

    with conn:
        conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32([value] * db.EMBED_DIM)),
        )


# ---------------------------------------------------------------------------
# RRF fusion math
# ---------------------------------------------------------------------------


def test_fuse_reciprocal_rank_math():
    lexical = [10, 20, 30]
    vector = [20, 30, 40]
    fused = search._fuse(lexical, vector)

    k = search.RRF_K
    assert fused[10] == pytest.approx(1 / (k + 1))
    assert fused[20] == pytest.approx(1 / (k + 2) + 1 / (k + 1))
    assert fused[30] == pytest.approx(1 / (k + 3) + 1 / (k + 2))
    assert fused[40] == pytest.approx(1 / (k + 3))


def test_fuse_empty_lists_produce_no_scores():
    assert search._fuse([], []) == {}


def test_fuse_single_list_is_just_reciprocal_rank():
    fused = search._fuse([5, 6, 7])
    k = search.RRF_K
    assert fused == {
        5: pytest.approx(1 / (k + 1)),
        6: pytest.approx(1 / (k + 2)),
        7: pytest.approx(1 / (k + 3)),
    }


# ---------------------------------------------------------------------------
# Recency boost + supersedence penalty
# ---------------------------------------------------------------------------


def test_recency_factor_is_one_for_today():
    import datetime as dt

    # UTC, matching _recency_factor's own "today" reference point: using the
    # local date here would be flaky near a UTC day boundary in timezones
    # ahead or behind UTC.
    today = dt.datetime.now(dt.UTC).date().isoformat()
    assert search._recency_factor(today, mtime=0.0) == pytest.approx(1.0)


def test_recency_factor_decays_with_age():
    import datetime as dt

    today_utc = dt.datetime.now(dt.UTC).date()
    old = (today_utc - dt.timedelta(days=int(search.RECENCY_HALFLIFE_DAYS))).isoformat()
    factor = search._recency_factor(old, mtime=0.0)
    assert factor == pytest.approx(0.5, rel=1e-2)


def test_recency_factor_falls_back_to_mtime_when_no_decided_on():
    import datetime as dt

    today_mtime = dt.datetime.now(dt.UTC).timestamp()
    factor = search._recency_factor(None, mtime=today_mtime)
    assert factor == pytest.approx(1.0, rel=1e-3)


def test_apply_boosts_downranks_superseded_chunk(lexical_conn):
    same_time = time.time()
    active_id = _insert_doc_with_chunk(
        lexical_conn, path="active.md", body="the active fact", mtime=same_time
    )
    superseded_id = _insert_doc_with_chunk(
        lexical_conn,
        path="superseded.md",
        body="the superseded fact",
        superseded_by="active.md#active-fact",
        superseded_by_doc_id=_doc_id_of(lexical_conn, "active.md"),
        mtime=same_time,
    )

    # Equal base RRF scores: only supersedence should differentiate them.
    boosted = search._apply_boosts(lexical_conn, {active_id: 1.0, superseded_id: 1.0})

    active_score = boosted[active_id][0]
    superseded_score = boosted[superseded_id][0]
    assert superseded_score == pytest.approx(active_score * search.SUPERSEDED_PENALTY)
    assert superseded_score < active_score


def test_search_ranks_superseded_chunk_below_its_replacement(lexical_conn):
    same_time = time.time()
    _insert_doc_with_chunk(
        lexical_conn,
        path="active.md",
        body="cache invalidation rules applies here",
        mtime=same_time,
    )
    _insert_doc_with_chunk(
        lexical_conn,
        path="superseded.md",
        body="cache invalidation rules applies here too",
        superseded_by="active.md",
        superseded_by_doc_id=_doc_id_of(lexical_conn, "active.md"),
        mtime=same_time,
    )

    response = search.search(lexical_conn, vec_ok=False, query="cache invalidation rules")
    assert len(response.hits) == 2
    assert response.hits[0].path == "active.md"
    assert response.hits[1].path == "superseded.md"
    assert response.hits[0].score > response.hits[1].score


def test_supersedence_penalty_needs_the_edge_not_the_frontmatter_string(lexical_conn):
    """A superseded_by that resolved to nothing must not bury the document.

    This is issue #21. The penalty used to key on the raw frontmatter string,
    while the successor shown alongside the hit comes from the resolved
    ``doc_links`` edge, so a value naming no document dropped the record by 70
    percent while displaying no successor and raising no error. The two now
    read the same edge, so they cannot disagree.
    """
    same_time = time.time()
    active_id = _insert_doc_with_chunk(
        lexical_conn, path="active.md", body="the active fact", mtime=same_time
    )
    dangling_id = _insert_doc_with_chunk(
        lexical_conn,
        path="dangling.md",
        body="the dangling fact",
        # Names a document that is not in the corpus, so no edge was written.
        superseded_by="a-document-that-does-not-exist.md",
        mtime=same_time,
    )

    boosted = search._apply_boosts(lexical_conn, {active_id: 1.0, dangling_id: 1.0})

    assert boosted[dangling_id][0] == pytest.approx(boosted[active_id][0])


def test_a_resolved_edge_penalises_even_with_no_frontmatter_string(lexical_conn):
    """The mirror direction: the edge alone is sufficient to penalise.

    Asserted separately from the test above because a mutation that made the
    penalty read ``superseded_by AND the edge`` would satisfy that one and
    still be wrong here.
    """
    same_time = time.time()
    active_id = _insert_doc_with_chunk(
        lexical_conn, path="active.md", body="the active fact", mtime=same_time
    )
    superseded_id = _insert_doc_with_chunk(
        lexical_conn,
        path="superseded.md",
        body="the superseded fact",
        superseded_by=None,
        superseded_by_doc_id=_doc_id_of(lexical_conn, "active.md"),
        mtime=same_time,
    )

    boosted = search._apply_boosts(lexical_conn, {active_id: 1.0, superseded_id: 1.0})

    assert boosted[superseded_id][0] == pytest.approx(
        boosted[active_id][0] * search.SUPERSEDED_PENALTY
    )


def test_an_unrelated_outgoing_link_does_not_count_as_supersedence(lexical_conn):
    """Only the ``superseded_by`` relation penalises, not any edge at all.

    The penalty is an EXISTS over ``doc_links`` filtered by relation. Without
    the filter every document that links to anything would be treated as
    replaced, which is most of the corpus.
    """
    same_time = time.time()
    active_id = _insert_doc_with_chunk(
        lexical_conn, path="active.md", body="the active fact", mtime=same_time
    )
    linking_id = _insert_doc_with_chunk(
        lexical_conn, path="linking.md", body="the linking fact", mtime=same_time
    )
    with lexical_conn:
        lexical_conn.execute(
            "INSERT INTO doc_links (src_doc_id, dst_doc_id, relation) VALUES (?, ?, ?)",
            (
                _doc_id_of(lexical_conn, "linking.md"),
                _doc_id_of(lexical_conn, "active.md"),
                "links_to",
            ),
        )

    boosted = search._apply_boosts(lexical_conn, {active_id: 1.0, linking_id: 1.0})

    assert boosted[linking_id][0] == pytest.approx(boosted[active_id][0])


def test_being_the_target_of_supersedence_does_not_penalise_the_successor(lexical_conn):
    """The replacement must not be demoted by the edge that points AT it.

    The EXISTS matches on ``src_doc_id``. Keying it on ``dst_doc_id`` instead
    would invert the feature exactly, promoting the retired record over the
    one that replaced it, and every assertion above would still pass because
    in those the pair differs on both ends at once.
    """
    same_time = time.time()
    active_id = _insert_doc_with_chunk(
        lexical_conn, path="active.md", body="the active fact", mtime=same_time
    )
    neutral_id = _insert_doc_with_chunk(
        lexical_conn, path="neutral.md", body="the neutral fact", mtime=same_time
    )
    _insert_doc_with_chunk(
        lexical_conn,
        path="superseded.md",
        body="the superseded fact",
        superseded_by="active.md",
        superseded_by_doc_id=_doc_id_of(lexical_conn, "active.md"),
        mtime=same_time,
    )

    boosted = search._apply_boosts(lexical_conn, {active_id: 1.0, neutral_id: 1.0})

    assert boosted[active_id][0] == pytest.approx(boosted[neutral_id][0])


# ---------------------------------------------------------------------------
# Degraded mode, and the mode label that reports it
# ---------------------------------------------------------------------------


def test_search_degrades_when_vec_ok_is_false(lexical_conn):
    _insert_doc_with_chunk(lexical_conn, path="a.md", body="findable content about widgets")

    response = search.search(lexical_conn, vec_ok=False, query="widgets")

    assert response.degraded is True
    assert response.degraded_reason == "sqlite-vec extension not loaded"
    assert response.mode == db.MODE_LEXICAL
    assert len(response.hits) == 1
    assert response.hits[0].path == "a.md"


def test_search_degrades_when_vec_ok_but_no_vec_table(lexical_conn, stub_embedder):
    # vec_ok=True but the connection's schema was created with vec=False, so
    # db.has_vec_table(conn) is False: the vector path must be skipped and
    # the response must say degraded, not silently error.
    _insert_doc_with_chunk(lexical_conn, path="a.md", body="findable content about sprockets")

    response = search.search(lexical_conn, vec_ok=True, query="sprockets", embedder=stub_embedder)

    assert response.degraded is True
    assert response.degraded_reason == "vector table not present"
    assert response.mode == db.MODE_LEXICAL
    assert len(response.hits) == 1


def test_search_degrades_when_zero_chunks_embedded(vec_conn):
    # The vector table exists (backend previously loaded, e.g. the index was
    # built while Ollama was down) but nothing has been embedded into it yet.
    # Table existence alone must not read as "not degraded": coverage is what
    # decides it, so this must degrade with a backfill-pointing reason.
    _insert_doc_with_chunk(vec_conn, path="a.md", body="findable content about thingamajigs")

    response = search.search(vec_conn, vec_ok=True, query="thingamajigs")

    assert response.degraded is True
    assert "0 chunks are embedded" in response.degraded_reason
    assert response.mode == db.MODE_LEXICAL
    assert len(response.hits) == 1


def test_search_degrades_when_embedder_raises(vec_conn, failing_embedder):
    # At least one chunk is embedded (so the zero-coverage branch is not what
    # trips this), but the query-time embed call itself fails: this is the
    # backend-down-mid-query case.
    chunk_id = _insert_doc_with_chunk(vec_conn, path="a.md", body="findable content about gadgets")
    _insert_vector(vec_conn, chunk_id)

    response = search.search(vec_conn, vec_ok=True, query="gadgets", embedder=failing_embedder)

    assert response.degraded is True
    assert response.degraded_reason.startswith("embedding backend unavailable")
    assert response.mode == db.MODE_LEXICAL
    assert len(response.hits) == 1
    assert response.hits[0].path == "a.md"


def test_search_not_degraded_when_backend_and_vectors_available(vec_conn, stub_embedder):
    chunk_id = _insert_doc_with_chunk(vec_conn, path="a.md", body="findable content about widgets")
    _insert_vector(vec_conn, chunk_id)

    response = search.search(vec_conn, vec_ok=True, query="widgets", embedder=stub_embedder)

    assert response.degraded is False
    assert response.degraded_reason is None
    assert response.mode == db.MODE_LEXICAL_VECTOR
    assert len(response.hits) == 1


def test_search_empty_query_returns_no_hits(lexical_conn):
    _insert_doc_with_chunk(lexical_conn, path="a.md", body="anything at all")
    response = search.search(lexical_conn, vec_ok=False, query="   ")
    assert response.hits == []
    # No channel ran, so no channel can be named. MODE_LEXICAL here would be
    # the same false claim as the degraded case: a label asserting that a
    # particular channel produced the page when none did.
    assert response.channels_used == frozenset()
    assert response.mode == db.MODE_NONE


def test_narrowed_channels_are_not_reported_as_hybrid(vec_conn, stub_embedder):
    # The second instance of issue #13, in the opposite direction and never
    # filed: with the vector channel simply not requested, nothing has failed,
    # so `degraded` is correctly False. Deriving the label from `degraded`
    # therefore announced `hybrid` for a page that only the lexical channel
    # produced. Availability and participation are different questions.
    chunk_id = _insert_doc_with_chunk(vec_conn, path="a.md", body="findable content about widgets")
    _insert_vector(vec_conn, chunk_id)

    response = search.search(
        vec_conn,
        vec_ok=True,
        query="widgets",
        embedder=stub_embedder,
        channels={search.CHANNEL_LEXICAL},
    )

    assert response.degraded is False
    assert response.channels_used == frozenset({search.CHANNEL_LEXICAL})
    assert response.mode == db.MODE_LEXICAL


def test_mode_is_derived_from_channels_used_and_cannot_contradict_it(
    vec_conn, stub_embedder, failing_embedder
):
    # The invariant, pinned across every state reachable here rather than at
    # one point: whatever `channels_used` says, `mode` is its summary. A label
    # computed independently is free to drift from it, which is how the two
    # fields disagreed in the first place.
    chunk_id = _insert_doc_with_chunk(vec_conn, path="a.md", body="findable content about widgets")
    _insert_vector(vec_conn, chunk_id)

    def mode_of(**kwargs):
        response = search.search(vec_conn, vec_ok=True, query="widgets", **kwargs)
        return response.mode, response.channels_used

    observed = [
        mode_of(embedder=stub_embedder),
        mode_of(embedder=failing_embedder),
        mode_of(embedder=stub_embedder, channels={search.CHANNEL_LEXICAL}),
        mode_of(embedder=stub_embedder, channels={search.CHANNEL_VECTOR}),
    ]
    # Rendering, not a lookup table: the assertion is that the label is the
    # channel set spelled out, which holds for the narrow sets a caller can
    # ask for as much as for the two production shapes. A table would have to
    # be extended for every new subset, and the entry nobody added is exactly
    # where the old two-value form went wrong.
    for mode, used in observed:
        assert mode == "+".join(c for c in search.CHANNEL_ORDER if c in used) or (
            mode == db.MODE_NONE and not used
        )
    modes = [mode for mode, _ in observed]
    assert modes == [
        db.MODE_LEXICAL_VECTOR,
        db.MODE_LEXICAL,
        db.MODE_LEXICAL,
        "vector",
    ]


def test_fts_match_query_escapes_punctuation(lexical_conn):
    _insert_doc_with_chunk(
        lexical_conn, path="a.md", body="cache invalidation rules: do not reintroduce"
    )
    # Punctuation and quote characters must not break the MATCH expression.
    response = search.search(lexical_conn, vec_ok=False, query='cache-invalidation "rules"?!')
    assert len(response.hits) == 1


# ---------------------------------------------------------------------------
# get_chunk / recent
# ---------------------------------------------------------------------------


def test_get_chunk_returns_full_body(lexical_conn):
    _insert_doc_with_chunk(lexical_conn, path="a.md", body="the full body text")
    hit = search.get_chunk(lexical_conn, db.chunk_ref("a.md", "Title > Section", 0))
    assert hit is not None
    assert hit.body == "the full body text"
    assert hit.citation == "a.md#Title > Section"
    assert hit.ref == db.chunk_ref("a.md", "Title > Section", 0)


def test_get_chunk_missing_ref_returns_none(lexical_conn):
    assert search.get_chunk(lexical_conn, "cdeadbeefdeadbeef") is None


def test_get_chunk_does_not_accept_a_rowid(lexical_conn):
    """A rowid must not resolve, even when it names a real row.

    This is the whole point of issue #23: an agent that kept an id from an
    older search must get a miss, not the row that now happens to occupy
    that id.
    """
    chunk_id = _insert_doc_with_chunk(lexical_conn, path="a.md", body="the full body text")
    assert search.get_chunk(lexical_conn, str(chunk_id)) is None


def test_chunk_ref_survives_a_rowid_reassignment(lexical_conn):
    """The ref keeps naming its section after the row is deleted and rebuilt.

    Simulates what reindex does to a changed document: the chunk rows are
    deleted and reinserted, so every rowid moves. The ref is unchanged and
    still resolves to the same section, now carrying the updated body.
    """
    original_id = _insert_doc_with_chunk(lexical_conn, path="a.md", body="original body")
    ref = db.chunk_ref("a.md", "Title > Section", 0)
    assert search.get_chunk(lexical_conn, ref).body == "original body"

    with lexical_conn:
        doc_id = lexical_conn.execute("SELECT doc_id FROM chunks WHERE ref = ?", (ref,)).fetchone()[
            "doc_id"
        ]
        lexical_conn.execute("DELETE FROM chunks WHERE ref = ?", (ref,))
        # A different document's chunk lands first and takes the freed rowid.
        lexical_conn.execute(
            "INSERT INTO chunks (ref, doc_id, heading_path, body) VALUES (?, ?, ?, ?)",
            (db.chunk_ref("a.md", "Title > Other", 0), doc_id, "Title > Other", "unrelated"),
        )
        lexical_conn.execute(
            "INSERT INTO chunks (ref, doc_id, heading_path, body) VALUES (?, ?, ?, ?)",
            (ref, doc_id, "Title > Section", "edited body"),
        )

    rebuilt = search.get_chunk(lexical_conn, ref)
    assert rebuilt is not None
    assert rebuilt.body == "edited body"
    assert rebuilt.heading_path == "Title > Section"
    # The rowid moved, which is exactly why it cannot be the durable handle.
    assert rebuilt.chunk_id != original_id


def test_repeated_heading_paths_in_one_document_get_distinct_refs():
    """Two identically titled sections must not collide on one ref.

    The ordinal disambiguates them, and it counts occurrences of the same
    heading path rather than position in the document, so adding an unrelated
    section does not renumber either of these.
    """
    first = db.chunk_ref("a.md", "Title > Notes", 0)
    second = db.chunk_ref("a.md", "Title > Notes", 1)
    assert first != second
    assert db.chunk_ref("a.md", "Title > Notes", 0) == first


def test_refs_are_scoped_to_their_document():
    """The same heading path in two documents must produce different refs."""
    assert db.chunk_ref("a.md", "Title > Section", 0) != db.chunk_ref("b.md", "Title > Section", 0)


def test_chunk_ref_never_parses_as_an_integer():
    """The prefix is load-bearing: a ref must not be mistakable for a rowid."""
    ref = db.chunk_ref("a.md", "Title > Section", 0)
    assert ref.startswith(db.CHUNK_REF_PREFIX)
    with pytest.raises(ValueError):
        int(ref)


def test_recent_orders_by_document_mtime_desc(lexical_conn):
    _insert_doc_with_chunk(lexical_conn, path="older.md", body="old content", mtime=100.0)
    _insert_doc_with_chunk(lexical_conn, path="newer.md", body="new content", mtime=200.0)

    hits = search.recent(lexical_conn, limit=10)
    assert [h.path for h in hits] == ["newer.md", "older.md"]


# ---------------------------------------------------------------------------
# One chunk per document
# ---------------------------------------------------------------------------


def _insert_doc_with_chunks(conn, *, path: str, bodies: list[str]) -> int:
    """Insert one document carrying several chunks, all under distinct headings."""
    with conn:
        cur = conn.execute(
            "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
            "VALUES ('repo', ?, 'Title', 'doc', ?, ?)",
            (path, time.time(), f"hash-{path}"),
        )
        doc_id = cur.lastrowid
        for ordinal, body in enumerate(bodies):
            heading = f"Title > Section {ordinal}"
            conn.execute(
                "INSERT INTO chunks (ref, doc_id, heading_path, body) VALUES (?, ?, ?, ?)",
                (db.chunk_ref(path, heading, ordinal), doc_id, heading, body),
            )
    return doc_id


def test_search_returns_at_most_one_chunk_per_document(lexical_conn):
    """A single document must not occupy every slot with its own sections.

    Relevance is judged, cited and read per document, so a page of near-identical
    sections from one document spends the caller's slot budget without adding an
    answer. Measured before this rule existed: a 10-slot page carried a mean of
    5.93 distinct documents across the judged set, and one query returned ten
    chunks of a single document.
    """
    crowder = _insert_doc_with_chunks(
        conn=lexical_conn,
        path="crowder.md",
        bodies=["quorum quorum quorum alpha"] * 4,
    )
    other = _insert_doc_with_chunks(
        conn=lexical_conn,
        path="other.md",
        bodies=["quorum beta"],
    )

    response = search.search(lexical_conn, False, "quorum", limit=2)

    doc_ids = [hit.doc_id for hit in response.hits]
    assert len(doc_ids) == len(set(doc_ids)), "a document appeared twice in one page"
    assert set(doc_ids) == {crowder, other}, (
        "the crowding document should keep only its best chunk, leaving room for "
        "the other document rather than filling both slots"
    )


def test_search_keeps_the_best_scoring_chunk_of_a_document(lexical_conn):
    """Deduplication must not change which chunk represents a document.

    The list is already in final ranked order when duplicates are dropped, so the
    survivor is the highest scoring chunk. A survivor chosen by insertion order
    instead would silently return a weaker section than the ranking selected.
    """
    _insert_doc_with_chunks(
        conn=lexical_conn,
        path="doc.md",
        bodies=["gamma unrelated filler text here", "gamma gamma gamma gamma"],
    )

    response = search.search(lexical_conn, False, "gamma", limit=5)

    assert len(response.hits) == 1
    assert response.hits[0].body == "gamma gamma gamma gamma"


# ---------------------------------------------------------------------------
# The graph channel votes at a discount, because it is derived (issue #17)
#
# It is seeded from the fused lexical+vector head, so a full peer vote counts
# the base ranking's own opinion twice and lets the derived channel outrank
# its own source. Measured: seven queries whose answer the base already had at
# rank 1 were pushed down, every one of them with a recall delta of zero.
# ---------------------------------------------------------------------------


def test_fuse_applies_a_per_list_weight():
    fused = search._fuse([1], [2], weights=(1.0, 0.25))
    assert fused[1] == pytest.approx(1.0 / (search.RRF_K + 1))
    assert fused[2] == pytest.approx(0.25 / (search.RRF_K + 1))


def test_fuse_is_unweighted_by_default():
    """The default must stay one vote per list.

    `base_fused`, the ranking the graph channel is seeded from, calls _fuse
    without weights and must not be discounted by this change.
    """
    assert search._fuse([1], [2]) == search._fuse([1], [2], weights=(1.0, 1.0))


def test_fuse_rejects_a_weight_count_that_does_not_match():
    """Silently padding or truncating would discount the wrong channel.

    A mismatch means the caller's idea of the channel order has drifted from
    the argument order, and guessing which list lost its weight is exactly the
    kind of quiet misranking this whole constant exists to prevent.
    """
    with pytest.raises(ValueError, match="2 ranked lists but 3 weights"):
        search._fuse([1], [2], weights=(1.0, 1.0, 1.0))


def test_a_weighted_graph_vote_cannot_tie_the_lexical_top_hit(lexical_conn):
    """The concrete displacement the weight prevents.

    A document found ONLY by the graph channel, at graph rank 1, scores
    ``w/(RRF_K+1)``. The lexical top hit scores ``1/(RRF_K+1)``. At a full
    vote those are EQUAL and the winner is decided by the tiebreak, so a
    document the query does not match can take rank 1 from one that does.
    """
    top = search._fuse(["lexical-top"], weights=(1.0,))["lexical-top"]
    graph_only = search._fuse(["graph-only"], weights=(search.GRAPH_VOTE_WEIGHT,))[
        "graph-only"
    ]
    assert graph_only < top
    assert graph_only == pytest.approx(top * search.GRAPH_VOTE_WEIGHT)


def test_search_gives_the_graph_list_the_discounted_weight(lexical_conn, monkeypatch):
    """Pins the wiring: the discount must reach the GRAPH list specifically.

    Asserted at the seam rather than through scores because a weight applied
    to the wrong list, or dropped on the way, produces a ranking that is
    merely different rather than obviously wrong, and no score assertion
    distinguishes those cases from a corpus change.
    """
    _insert_doc_with_chunk(lexical_conn, path="a.md", body="cache invalidation rules")
    seen: list[tuple[float, ...] | None] = []
    real_fuse = search._fuse

    def spy(*ranked_lists, weights=None):
        seen.append(weights)
        return real_fuse(*ranked_lists, weights=weights)

    monkeypatch.setattr(search, "_fuse", spy)
    search.search(lexical_conn, vec_ok=False, query="cache invalidation rules")

    # The first call is base_fused (unweighted, it is the thing being seeded
    # from); the final call is the one the ranking is built on.
    assert seen[0] is None
    assert seen[-1] == (1.0, 1.0, search.GRAPH_VOTE_WEIGHT)


def test_a_model_of_another_width_degrades_instead_of_raising(vec_conn, stub_embedder):
    """Switching to a narrower model without reindexing must degrade, not
    crash.

    The embedder learns its width from the model now, so nothing upstream
    rejects the mismatched query vector; vec0 answers it with
    sqlite3.OperationalError, which no caller catches and which is not in
    cli._CLEAN_ERRORS, so the CLI printed a traceback. The pre-#15 code was
    accidentally safe here because the pinned width made the embedder raise
    EmbeddingUnavailable first.
    """
    from conftest import StubEmbedder

    chunk_id = _insert_doc_with_chunk(vec_conn, path="a.md", body="findable content about widgets")
    _insert_vector(vec_conn, chunk_id)
    assert db.vec_table_dim(vec_conn) == db.EMBED_DIM

    narrow = StubEmbedder(dim=384, model="narrow-model")
    response = search.search(vec_conn, vec_ok=True, query="widgets", embedder=narrow)

    assert response.degraded is True
    assert str(db.EMBED_DIM) in response.degraded_reason
    assert "384" in response.degraded_reason
    assert "reindex" in response.degraded_reason
    # Degraded is not empty: the lexical channel still answers the query.
    assert len(response.hits) == 1
    assert response.channels_used == frozenset({search.CHANNEL_LEXICAL})
