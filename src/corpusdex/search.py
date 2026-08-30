"""Hybrid lexical + vector + graph retrieval over the brain index.

Three ranked candidate lists are built independently, FTS5 BM25 for lexical
relevance, sqlite-vec KNN for semantic similarity, and Personalized PageRank
over the document link graph for associative reach, then fused by reciprocal
rank fusion: ``score = sum(1 / (RRF_K + rank))`` across whichever lists a
chunk appears in. A recency boost (from a decision's ``decided_on``, falling
back to the document's mtime) multiplies the fused score up, and a superseded
chunk's final score is multiplied down by :data:`SUPERSEDED_PENALTY` so
replaced facts still surface but rank below their replacement.

The graph channel is seeded from the documents behind the top fused
lexical+vector candidates, so it can only ever reinforce or extend what the
first two channels already found (see :mod:`corpusdex.graph`). On an index
with no links, or for a query whose seeds have no graph presence, it
contributes an empty list and the fusion is exactly what it was before.

Each hit also carries an assembled context: its document's supersedence chain
and 1-hop graph neighbours as compact references, hard-capped so a response
stays small.

Every hit is identified by :attr:`SearchHit.ref`, a stable handle derived from
document position (:func:`corpusdex.db.chunk_ref`). That is the value a
caller keeps and hands back to :func:`get_chunk`. The rowid in
:attr:`SearchHit.chunk_id` is internal: any reindex reassigns it, so a held
rowid resolves to unrelated content rather than failing. Refs also give the
ranking a content-derived tiebreak, which is what makes a search reproducible
across a rebuild of an unchanged corpus.

If the vector extension is not loaded or the local embedder cannot serve a
query embedding, the vector channel drops out and the response says so via
:attr:`SearchResponse.degraded`. The lexical and graph channels both keep
working, so degraded results are lexical plus graph, not lexical alone, and
:attr:`SearchResponse.mode` reports
:data:`corpusdex.db.MODE_LEXICAL_GRAPH` rather than claiming the results
were lexical-only.

``mode`` is RENDERED from :attr:`SearchResponse.channels_used`, so the two
can never contradict each other. It used to be derived from ``degraded``,
which made it wrong in two separate ways: results labelled ``lexical-only``
had in fact been re-ranked by the graph channel, and a caller narrowing
``channels`` to lexical alone was told ``hybrid``, because no backend had
failed. Neither case is a degradation, and only one of them involves the
embedder at all, which is why availability was the wrong thing to summarise.
``degraded`` keeps its own narrower meaning: the vector channel was asked for
and could not run.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from . import db, graph
from .embedder import EmbeddingUnavailable, OllamaEmbedder, default_embedder

RRF_K = 60

#: The graph channel's share of one RRF vote.
#:
#: It is not a peer channel. It is SEEDED from the fused lexical+vector head
#: (see :func:`search`), so its ranking partly re-expresses the base ranking's
#: own opinion rather than adding independent evidence. At a full vote that
#: opinion is counted twice and the derived channel can outvote its own
#: source: measured on the 36-query judged set, seven queries whose answer the
#: base ranking already had at RANK 1 were pushed down, one of them from rank
#: 1 to rank 5, and every one of those had a recall delta of exactly zero. The
#: channel was not finding anything, only reshuffling.
#:
#: 0.5 is the edge of the recall plateau, not the best number on a curve. A
#: weight sweep against a frozen index (contribution over ``lexical+vector``):
#:
#: ====== ========= =======
#: weight  recall     mrr
#: ====== ========= =======
#: 1.00    +0.060    -0.032
#: 0.75    +0.060    -0.009
#: 0.50    +0.060    +0.019
#: 0.35    +0.051    +0.015
#: 0.25    +0.051    +0.026
#: 0.10    +0.014    +0.018
#: ====== ========= =======
#:
#: Recall is FLAT down to 0.5 and decays below it, so 0.5 is the smallest
#: weight that gives up no recall at all, and it is where MRR first turns
#: positive. Chosen by that structural criterion rather than by the best MRR
#: cell, which would have picked 0.25 and paid recall for it.
#:
#: Dropping the channel to pure expansion (voting only for documents the base
#: ranking missed) was measured too and is much worse: recall +0.005. The
#: channel's value really is in promoting documents the base already found but
#: ranked below the cut, so it must keep voting for them, just not loudly
#: enough to outrank them.
GRAPH_VOTE_WEIGHT = 0.5
CANDIDATE_POOL = 50

CHANNEL_LEXICAL = "lexical"
CHANNEL_VECTOR = "vector"
CHANNEL_GRAPH = "graph"
#: Production default: every channel on. The ablation harness (see
#: :mod:`corpusdex.evaluate`) narrows this to measure what each channel
#: contributes, and narrowing it is the only supported way to do that, so the
#: measurement runs the same ranking code the real search runs.
ALL_CHANNELS = frozenset({CHANNEL_LEXICAL, CHANNEL_VECTOR, CHANNEL_GRAPH})
RECENCY_WEIGHT = 0.25
RECENCY_HALFLIFE_DAYS = 180.0
SUPERSEDED_PENALTY = 0.3
SNIPPET_CHARS = 240

_HIT_COLUMNS = (
    "chunks.id AS id, chunks.ref AS ref, "
    "chunks.doc_id AS doc_id, chunks.heading_path AS heading_path, "
    "chunks.body AS body, "
    "chunks.decided_on AS decided_on, chunks.superseded_by AS superseded_by, "
    "chunks.tags AS tags, documents.repo AS repo, documents.path AS path, "
    "documents.title AS title, documents.doc_type AS doc_type, documents.mtime AS mtime, "
    # Derived from the SAME edge the assembled supersedence chain walks
    # (corpusdex.graph._supersedence_chain), so the ranking penalty and the
    # displayed successor cannot disagree. Computing the penalty from the raw
    # frontmatter STRING instead let a value that resolves to nothing bury a
    # record by 70 percent while showing no successor and raising no error
    # (issue #21). The string stays on the hit for display; it is no longer
    # what the ranking reads.
    "EXISTS (SELECT 1 FROM doc_links WHERE doc_links.src_doc_id = chunks.doc_id "
    "AND doc_links.relation = 'superseded_by') AS is_superseded"
)
_HIT_FROM = "FROM chunks JOIN documents ON documents.id = chunks.doc_id"


@dataclass(frozen=True)
class SearchHit:
    #: The durable handle for this chunk, and the only one a caller should
    #: keep. See :func:`corpusdex.db.chunk_ref`; ``chunk_id`` below is an
    #: internal rowid that is reassigned by any reindex.
    ref: str
    chunk_id: int
    repo: str
    path: str
    title: str
    heading_path: str
    doc_type: str
    body: str
    snippet: str
    score: float
    decided_on: str | None
    superseded_by: str | None
    tags: str | None
    doc_id: int = 0
    assembled: tuple[graph.ContextRef, ...] = ()

    @property
    def citation(self) -> str:
        # ``path`` is already workspace-relative and therefore repo-prefixed
        # (see corpusdex.indexer.discover_corpus), so it alone reads as
        # "repo/path"; prepending ``repo`` again would duplicate it.
        return f"{self.path}#{self.heading_path}"


@dataclass(frozen=True)
class SearchResponse:
    query: str
    mode: str
    degraded: bool
    degraded_reason: str | None
    hits: list[SearchHit]
    #: Channels that actually contributed to this ranking. A channel the
    #: caller switched off is absent here but is *not* reported as degraded:
    #: "you asked for lexical only" and "the vector backend died" produce the
    #: same empty candidate list and must not produce the same status, or an
    #: ablation run silently reads as a broken run (and vice versa).
    channels_used: frozenset[str] = ALL_CHANNELS


def _snippet(body: str, limit: int = SNIPPET_CHARS) -> str:
    collapsed = " ".join(body.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."


def _row_to_hit(
    row: sqlite3.Row, score: float, assembled: tuple[graph.ContextRef, ...] = ()
) -> SearchHit:
    return SearchHit(
        ref=row["ref"],
        chunk_id=row["id"],
        doc_id=row["doc_id"],
        assembled=assembled,
        repo=row["repo"],
        path=row["path"],
        title=row["title"],
        heading_path=row["heading_path"],
        doc_type=row["doc_type"],
        body=row["body"],
        snippet=_snippet(row["body"]),
        score=round(score, 6),
        decided_on=row["decided_on"],
        superseded_by=row["superseded_by"],
        tags=row["tags"],
    )


def _fts_match_query(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression from free-text ``query``.

    Tokens are quoted individually and OR-ed so punctuation in the raw query
    (colons, hyphens, quotes) can never be interpreted as FTS5 query syntax.
    """
    tokens = [tok for tok in "".join(c if c.isalnum() else " " for c in query).split() if tok]
    if not tokens:
        return None
    return " OR ".join(f'"{tok}"' for tok in tokens)


def _lexical_ranked_ids(conn: sqlite3.Connection, query: str, limit: int) -> list[int]:
    match = _fts_match_query(query)
    if match is None:
        return []
    rows = conn.execute(
        "SELECT chunks_fts.rowid AS id FROM chunks_fts "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (match, limit),
    ).fetchall()
    return [row["id"] for row in rows]


def _vector_ranked_ids(
    conn: sqlite3.Connection, query_vector: list[float], limit: int
) -> list[int]:
    import sqlite_vec

    rows = conn.execute(
        "SELECT chunk_id AS id FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (sqlite_vec.serialize_float32(query_vector), limit),
    ).fetchall()
    return [row["id"] for row in rows]


def _fuse(*ranked_lists: list[int], weights: Sequence[float] | None = None) -> dict[int, float]:
    """Reciprocal rank fusion: ``sum(w / (RRF_K + rank))`` over 1-indexed ranks.

    ``weights`` defaults to one vote per list. It exists so a channel that is
    not independent evidence can be discounted without a second copy of the
    RRF formula living at the call site, where the two would be free to drift.
    """
    if weights is None:
        weights = (1.0,) * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(f"got {len(ranked_lists)} ranked lists but {len(weights)} weights")
    scores: dict[int, float] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank)
    return scores


def _recency_factor(decided_on: str | None, mtime: float) -> float:
    """Exponential decay in ``[0, 1]``, 1.0 for "today", halving every ``RECENCY_HALFLIFE_DAYS``.

    Both the "today" reference point and the mtime fallback are computed in
    UTC, matching how ``mtime`` itself is interpreted elsewhere (e.g.
    :func:`_row_to_hit`'s callers via ``datetime.fromtimestamp(mtime, tz=UTC)``),
    so recency is never off by a timezone offset near a day boundary.
    """
    reference: date | None = None
    if decided_on:
        try:
            reference = date.fromisoformat(decided_on[:10])
        except ValueError:
            reference = None
    if reference is None:
        reference = datetime.fromtimestamp(mtime, tz=UTC).date()
    today = datetime.now(UTC).date()
    age_days = max(0, (today - reference).days)
    return math.pow(2.0, -age_days / RECENCY_HALFLIFE_DAYS)


def _fetch_refs(conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, str]:
    """Map rowids to their stable refs, for use as a deterministic tiebreak.

    Ties in the base RRF ranking are the common case, not an edge case: a
    chunk that only the lexical channel returned at rank 1 scores exactly
    what a chunk only the vector channel returned at rank 1 scores. Breaking
    those ties on rowid made the ordering depend on insert history, so a full
    rebuild of an unchanged corpus could reorder the graph channel's seeds and
    change the final results. Breaking them on ref depends only on content
    position, which is what makes an evaluation run reproducible (issue #17).
    """
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"SELECT id, ref FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
        chunk_ids,
    ).fetchall()
    return {row["id"]: row["ref"] for row in rows}


def _fetch_rows(conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, sqlite3.Row]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"SELECT {_HIT_COLUMNS} {_HIT_FROM} WHERE chunks.id IN ({placeholders})",  # noqa: S608
        chunk_ids,
    ).fetchall()
    return {row["id"]: row for row in rows}


def _apply_boosts(
    conn: sqlite3.Connection, scores: dict[int, float]
) -> dict[int, tuple[float, sqlite3.Row]]:
    """Apply the recency boost and supersedence penalty to fused RRF scores."""
    rows = _fetch_rows(conn, list(scores.keys()))
    boosted: dict[int, tuple[float, sqlite3.Row]] = {}
    for chunk_id, base_score in scores.items():
        row = rows.get(chunk_id)
        if row is None:
            continue
        recency = _recency_factor(row["decided_on"], row["mtime"])
        final = base_score * (1.0 + RECENCY_WEIGHT * recency)
        if row["is_superseded"]:
            final *= SUPERSEDED_PENALTY
        boosted[chunk_id] = (final, row)
    return boosted


#: Pipeline order, so the rendered mode reads the way the channels are
#: described everywhere else. Alphabetical order would put graph first and
#: suggest it leads, when it is seeded from the other two.
CHANNEL_ORDER = (CHANNEL_LEXICAL, CHANNEL_VECTOR, CHANNEL_GRAPH)


def _mode_for(contributed: set[str]) -> str:
    """Render the channels that contributed as the response's ``mode``.

    Total by construction: every subset renders, so there is no state this
    can decline to describe and no state it can describe wrongly. The empty
    case gets its own word rather than falling through to a channel name,
    because a page produced by nothing is not a page produced by lexical
    search that happened to find nothing.
    """
    ordered = [channel for channel in CHANNEL_ORDER if channel in contributed]
    return "+".join(ordered) if ordered else db.MODE_NONE


def search(
    conn: sqlite3.Connection,
    vec_ok: bool,
    query: str,
    *,
    limit: int = 10,
    embedder: OllamaEmbedder | None = None,
    channels: frozenset[str] | None = None,
) -> SearchResponse:
    """Run hybrid search for ``query`` against the open index ``conn``.

    ``vec_ok`` should be the value returned by :func:`corpusdex.db.open_index`
    for this connection; when False (or the embedder cannot serve a query
    embedding) the vector channel is skipped and the lexical and graph
    channels alone produce the ranking.

    ``channels`` restricts which retrieval channels run, defaulting to all
    three. It exists for the ablation in :mod:`corpusdex.evaluate` and is
    the reason that harness does not need its own copy of the fusion: an
    ablation that reimplements ranking measures the copy, not the product.
    Switching a channel off is never reported as degradation; see
    :attr:`SearchResponse.channels_used`.
    """
    active = ALL_CHANNELS if channels is None else frozenset(channels)
    unknown = active - ALL_CHANNELS
    if unknown:
        raise ValueError(f"unknown retrieval channel(s): {', '.join(sorted(unknown))}")

    query = query.strip()
    lexical_ids = (
        _lexical_ranked_ids(conn, query, CANDIDATE_POOL)
        if query and CHANNEL_LEXICAL in active
        else []
    )

    vector_ids: list[int] = []
    degraded = False
    degraded_reason: str | None = None
    if CHANNEL_VECTOR not in active:
        pass
    elif not vec_ok:
        degraded = True
        degraded_reason = "sqlite-vec extension not loaded"
    elif query:
        if not db.has_vec_table(conn):
            degraded = True
            degraded_reason = "vector table not present"
        else:
            # Coverage, not mere table existence, decides degraded: a live
            # backend with zero embedded chunks (e.g. the index was built
            # while Ollama was down and has not yet been backfilled) cannot
            # contribute a vector ranking, so it is degraded too.
            total_vectors = conn.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()["n"]
            if total_vectors == 0:
                degraded = True
                degraded_reason = "0 chunks are embedded; run `brain reindex` to backfill vectors"
            else:
                active_embedder = embedder or default_embedder()
                try:
                    query_vector = active_embedder.embed([query])[0]
                except EmbeddingUnavailable as exc:
                    degraded = True
                    degraded_reason = f"embedding backend unavailable: {exc}"
                else:
                    # The embedder learns its width from the model, so a model
                    # switched since the last reindex produces a query vector
                    # the stored table cannot accept. vec0 answers that with
                    # sqlite3.OperationalError, which no caller catches and
                    # which reaches the CLI as a traceback. Compare first and
                    # degrade the way an unreachable backend already does: the
                    # lexical and graph channels still answer, and the fix is
                    # a reindex rather than an error the user must decode.
                    table_dim = db.vec_table_dim(conn)
                    if table_dim is not None and len(query_vector) != table_dim:
                        degraded = True
                        degraded_reason = (
                            f"index holds {table_dim}-dimension vectors but the "
                            f"active embedding model produced {len(query_vector)}; "
                            "run `brain reindex` to re-embed at the new width"
                        )
                    else:
                        vector_ids = _vector_ranked_ids(conn, query_vector, CANDIDATE_POOL)

    # The graph channel is seeded from the documents behind the strongest
    # lexical+vector candidates, so it extends the existing ranking rather
    # than introducing an independent one. The whole base ranking is handed
    # over, not just the seed slice: its order also decides which chunks of a
    # promoted document the channel votes for.
    base_fused = _fuse(lexical_ids, vector_ids)
    base_refs = _fetch_refs(conn, list(base_fused))
    base_ranked = sorted(
        base_fused, key=lambda cid: (-base_fused[cid], base_refs.get(cid, ""), cid)
    )
    graph_ids: list[int] = []
    if CHANNEL_GRAPH in active:
        graph_ids = graph.ppr_ranked_chunk_ids(conn, base_ranked, CANDIDATE_POOL)

    # Weighted, because the graph list was derived from the other two: see
    # GRAPH_VOTE_WEIGHT. base_fused above stays unweighted, since that is the
    # ranking being seeded FROM rather than voted on.
    fused = _fuse(
        lexical_ids,
        vector_ids,
        graph_ids,
        weights=(1.0, 1.0, GRAPH_VOTE_WEIGHT),
    )
    boosted = _apply_boosts(conn, fused)
    # Descending score, ties broken by the stable ref rather than left to dict
    # insertion order, so the same index and query always produce the same
    # ordering.
    ordered = sorted(boosted.items(), key=lambda item: (-item[1][0], item[1][1]["ref"]))
    # One chunk per document. Relevance is judged, cited and read per document,
    # but the ranking is over chunks, so without this a single well-matching
    # document can occupy most of a result page with near-identical sections and
    # crowd out every other answer. Measured on the 30-query judged set: a
    # 10-slot page carried a mean of 5.93 DISTINCT documents, and one query
    # returned 10 chunks of a single document. Keeping the best-scoring chunk
    # preserves the ranking (the list is already in final order) while making
    # the slot budget mean what a caller assumes it means.
    seen_docs: set[int] = set()
    deduped = []
    for chunk_id, payload in ordered:
        doc_id = payload[1]["doc_id"]
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        deduped.append((chunk_id, payload))
    ranked = deduped[:limit]
    hits = [
        _row_to_hit(row, score, graph.assemble_context(conn, row["doc_id"]))
        for _chunk_id, (score, row) in ranked
    ]

    contributed = set(active)
    if CHANNEL_VECTOR in contributed and not vector_ids:
        contributed.discard(CHANNEL_VECTOR)
    if CHANNEL_GRAPH in contributed and not graph_ids:
        contributed.discard(CHANNEL_GRAPH)
    if CHANNEL_LEXICAL in contributed and not lexical_ids:
        contributed.discard(CHANNEL_LEXICAL)
    return SearchResponse(
        query=query,
        mode=_mode_for(contributed),
        degraded=degraded,
        degraded_reason=degraded_reason,
        hits=hits,
        channels_used=frozenset(contributed),
    )


def get_chunk(conn: sqlite3.Connection, ref: str) -> SearchHit | None:
    """Fetch a single chunk by its stable ref, full body included.

    Takes a ref, never a rowid. A rowid held across a reindex resolves to
    whatever row now occupies that id, so the ``get`` surface would answer a
    stale reference with confidently wrong content instead of a miss; a ref
    that no longer names a live section simply returns None (issue #23).
    """
    row = conn.execute(
        f"SELECT {_HIT_COLUMNS} {_HIT_FROM} WHERE chunks.ref = ?",  # noqa: S608
        (ref,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_hit(row, 0.0, graph.assemble_context(conn, row["doc_id"]))


def recent(conn: sqlite3.Connection, limit: int = 10) -> list[SearchHit]:
    """Most recently touched chunks by document mtime, for the ``recent`` command.

    Assembles context like the other surfaces. Leaving it empty here made
    "not computed" indistinguishable from "this document has no relations".
    """
    rows = conn.execute(
        f"SELECT {_HIT_COLUMNS} {_HIT_FROM} ORDER BY documents.mtime DESC LIMIT ?",  # noqa: S608
        (limit,),
    ).fetchall()
    return [_row_to_hit(row, 0.0, graph.assemble_context(conn, row["doc_id"])) for row in rows]
