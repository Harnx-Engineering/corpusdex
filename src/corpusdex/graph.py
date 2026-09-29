"""Doc-level link graph: Personalized PageRank and recall-time context assembly.

The graph is built at index time from links the corpus already writes (see
:mod:`corpusdex.links`) and stored as resolved document-to-document edges in
``doc_links``. Two things read it:

* :func:`ppr_ranked_chunk_ids`, which seeds Personalized PageRank with the
  documents behind the fused lexical+vector hits and returns a chunk-level
  ranking for :mod:`corpusdex.search` to fuse as a third RRF channel. This
  is the cheap, LLM-free approximation of HippoRAG-style associative recall:
  a document that many strong hits point at rises even when its own text
  never matched the query.
* :func:`assemble_context`, which returns a small, capped set of references
  around one hit: its supersedence chain in both directions plus its 1-hop
  neighbours. Small and curated on purpose (decisions/0005, Zep finding): the
  point is a few high-value pointers, not a second pile of text.

Edges are directed as written, and every edge is also walked backwards at
:data:`REVERSE_EDGE_WEIGHT`. A citation is a stronger signal forwards (the
cited document is the one being relied on) than backwards, but relevance does
flow both ways: a hit on a superseded decision should be able to surface the
decision that replaced it. Weighting the reverse direction down keeps that
path open without discarding the direction information a fully symmetric
graph would throw away.

Every entry point degrades to "contributes nothing" rather than raising: an
index with no links, or a query whose seeds are absent from the graph, yields
an empty ranking and leaves the other channels untouched.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .links import RELATION_SUPERSEDED_BY

DAMPING = 0.85
TOLERANCE = 1e-8
MAX_ITERATIONS = 100
REVERSE_EDGE_WEIGHT = 0.5
PPR_SEED_POOL = 10
PPR_SEED_DOCS = 10
MAX_CHUNKS_PER_DOC = 3
MAX_ASSEMBLED_ITEMS = 8

REL_SUPERSEDED_BY = RELATION_SUPERSEDED_BY
REL_SUPERSEDES = "supersedes"
REL_LINKS_TO = "links_to"
REL_LINKED_FROM = "linked_from"


@dataclass(frozen=True)
class ContextRef:
    """One compact pointer to a related document, for assembled context."""

    doc_id: int
    title: str
    repo: str
    path: str
    relation: str


def load_adjacency(conn: sqlite3.Connection) -> dict[int, dict[int, float]]:
    """Load the weighted out-edge adjacency, reverse edges included.

    Returns an empty mapping when the index holds no links at all, which is
    what makes the whole PPR channel a no-op on a link-free corpus.
    """
    adjacency: dict[int, dict[int, float]] = {}
    rows = conn.execute("SELECT src_doc_id, dst_doc_id FROM doc_links").fetchall()
    for row in rows:
        src = row["src_doc_id"]
        dst = row["dst_doc_id"]
        forward = adjacency.setdefault(src, {})
        forward[dst] = max(forward.get(dst, 0.0), 1.0)
        backward = adjacency.setdefault(dst, {})
        backward[src] = max(backward.get(src, 0.0), REVERSE_EDGE_WEIGHT)
    return adjacency


def personalized_pagerank(
    adjacency: dict[int, dict[int, float]],
    seeds: list[int],
    *,
    damping: float = DAMPING,
    tolerance: float = TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
) -> dict[int, float]:
    """Power-iterate Personalized PageRank over ``adjacency`` from ``seeds``.

    Returns ``{}`` when there is no graph, no seed, or no seed present in the
    graph. Iteration order is sorted throughout so the result is bit-for-bit
    reproducible for a given graph and seed set.

    Dangling mass (a node with no out-edges) is returned to the personalization
    vector rather than spread uniformly, which is what keeps the walk anchored
    to the seeds instead of drifting toward whatever the graph's largest
    component happens to be.
    """
    if not adjacency:
        return {}
    nodes = sorted(adjacency)
    present = [node for node in sorted(set(seeds)) if node in adjacency]
    if not present:
        return {}

    seed_mass = 1.0 / len(present)
    personalization = {node: 0.0 for node in nodes}
    for node in present:
        personalization[node] = seed_mass

    rank = dict(personalization)
    normalized: dict[int, list[tuple[int, float]]] = {}
    for node in nodes:
        out = adjacency.get(node) or {}
        total = sum(out.values())
        if total > 0.0:
            normalized[node] = [(dst, weight / total) for dst, weight in sorted(out.items())]

    for _ in range(max_iterations):
        updated = {node: 0.0 for node in nodes}
        dangling = 0.0
        for node in nodes:
            mass = rank[node]
            edges = normalized.get(node)
            if not edges:
                dangling += mass
                continue
            for dst, share in edges:
                updated[dst] += mass * share
        delta = 0.0
        for node in nodes:
            value = (
                damping * updated[node]
                + damping * dangling * personalization[node]
                + (1.0 - damping) * personalization[node]
            )
            delta += abs(value - rank[node])
            updated[node] = value
        rank = updated
        if delta < tolerance:
            break
    return rank


def _chunk_ids_for_docs(conn: sqlite3.Connection, doc_ids: list[int]) -> dict[int, list[int]]:
    if not doc_ids:
        return {}
    placeholders = ",".join("?" for _ in doc_ids)
    rows = conn.execute(
        f"SELECT id, doc_id FROM chunks WHERE doc_id IN ({placeholders}) "  # noqa: S608
        "ORDER BY doc_id, id",
        doc_ids,
    ).fetchall()
    grouped: dict[int, list[int]] = {}
    for row in rows:
        grouped.setdefault(row["doc_id"], []).append(row["id"])
    return grouped


def doc_ids_for_chunks(conn: sqlite3.Connection, chunk_ids: list[int]) -> list[int]:
    """Map chunk ids to their document ids, preserving first-seen chunk order."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"SELECT id, doc_id FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
        chunk_ids,
    ).fetchall()
    by_chunk = {row["id"]: row["doc_id"] for row in rows}
    ordered: list[int] = []
    seen: set[int] = set()
    for chunk_id in chunk_ids:
        doc_id = by_chunk.get(chunk_id)
        if doc_id is not None and doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def ppr_ranked_chunk_ids(
    conn: sqlite3.Connection, candidate_chunk_ids: list[int], limit: int
) -> list[int]:
    """Rank chunks by their document's Personalized PageRank score.

    ``candidate_chunk_ids`` is the fused lexical+vector ranking, best first.
    It is read twice, for two different purposes that must not be conflated:
    only its head (``PPR_SEED_POOL`` chunks) seeds the walk, while the whole
    list decides which of a promoted document's chunks this channel votes for.
    Seeding from the whole list instead measurably displaces strong lexical
    hits rather than extending them, because ten unrelated seed documents pull
    the walk toward whatever they collectively neighbour.

    Returns at most ``limit`` chunk ids, and an empty list whenever the graph
    cannot contribute, so the caller can fuse it unconditionally.
    """
    if limit <= 0:
        return []
    seed_docs = doc_ids_for_chunks(conn, candidate_chunk_ids[:PPR_SEED_POOL])[:PPR_SEED_DOCS]
    if not seed_docs:
        return []
    adjacency = load_adjacency(conn)
    rank = personalized_pagerank(adjacency, seed_docs)
    if not rank:
        return []
    scored = [(doc_id, score) for doc_id, score in rank.items() if score > 0.0]
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[1], item[0]))
    ordered_docs = [doc_id for doc_id, _score in scored]
    chunks_by_doc = _chunk_ids_for_docs(conn, ordered_docs)
    # PageRank scores documents, not chunks. Emitting every chunk of a
    # high-scoring document would hand it a whole consecutive block of top
    # ranks and crowd genuinely distinct documents out of the fused head, so
    # each document contributes at most MAX_CHUNKS_PER_DOC.
    #
    # Which chunks matters as much as how many: a document's leading chunks
    # are usually its title and preamble, so voting for those blindly would
    # spend the channel's weight on text that matched nothing. Chunks the
    # first two channels already ranked come first, in that ranking's order,
    # and leading chunks only pad the remainder for documents the query never
    # touched directly (which is exactly the multi-hop case this channel is
    # for).
    candidate_order = {chunk_id: i for i, chunk_id in enumerate(candidate_chunk_ids)}
    ranked: list[int] = []
    for doc_id in ordered_docs:
        chunks = chunks_by_doc.get(doc_id, [])
        matched = sorted(
            (c for c in chunks if c in candidate_order), key=lambda c: candidate_order[c]
        )
        padding = [c for c in chunks if c not in candidate_order]
        for chunk_id in (matched + padding)[:MAX_CHUNKS_PER_DOC]:
            ranked.append(chunk_id)
            if len(ranked) >= limit:
                return ranked
    return ranked


def _neighbours(
    conn: sqlite3.Connection, doc_id: int, relation: str, *, reverse: bool
) -> list[int]:
    if reverse:
        sql = "SELECT src_doc_id AS other FROM doc_links WHERE dst_doc_id = ? AND relation = ?"
    else:
        sql = "SELECT dst_doc_id AS other FROM doc_links WHERE src_doc_id = ? AND relation = ?"
    rows = conn.execute(sql + " ORDER BY other", (doc_id, relation)).fetchall()
    return [row["other"] for row in rows]


def _supersedence_chain(conn: sqlite3.Connection, doc_id: int) -> list[tuple[int, str]]:
    """Walk ``superseded_by`` transitively in both directions from ``doc_id``.

    Forward hops are what supersedes this document; backward hops are what
    this document supersedes. Visited ids are tracked across both directions,
    so a cyclic chain terminates instead of looping and a document reachable
    each way is reported once, under the forward relation.
    """
    found: list[tuple[int, str]] = []
    visited = {doc_id}
    for reverse, relation in ((False, REL_SUPERSEDED_BY), (True, REL_SUPERSEDES)):
        frontier = [doc_id]
        while frontier:
            nxt: list[int] = []
            for current in frontier:
                for other in _neighbours(conn, current, REL_SUPERSEDED_BY, reverse=reverse):
                    if other in visited:
                        continue
                    visited.add(other)
                    found.append((other, relation))
                    nxt.append(other)
            frontier = nxt
    return found


def _doc_rows(conn: sqlite3.Connection, doc_ids: list[int]) -> dict[int, sqlite3.Row]:
    if not doc_ids:
        return {}
    placeholders = ",".join("?" for _ in doc_ids)
    rows = conn.execute(
        f"SELECT id, repo, path, title FROM documents WHERE id IN ({placeholders})",  # noqa: S608
        doc_ids,
    ).fetchall()
    return {row["id"]: row for row in rows}


def assemble_context(
    conn: sqlite3.Connection, doc_id: int, *, cap: int = MAX_ASSEMBLED_ITEMS
) -> tuple[ContextRef, ...]:
    """Return up to ``cap`` compact references around ``doc_id``.

    Ordered by value: the full supersedence chain first (it changes whether a
    fact is still true), then 1-hop links. Returns an empty tuple when the
    document has no graph presence, and never raises on a link-free index.
    """
    if cap <= 0:
        return ()
    candidates: list[tuple[int, str]] = list(_supersedence_chain(conn, doc_id))
    seen = {other for other, _relation in candidates}
    seen.add(doc_id)
    for reverse, relation in ((False, REL_LINKS_TO), (True, REL_LINKED_FROM)):
        for other in _neighbours(conn, doc_id, REL_LINKS_TO, reverse=reverse):
            if other in seen:
                continue
            seen.add(other)
            candidates.append((other, relation))
    if not candidates:
        return ()
    capped = candidates[:cap]
    rows = _doc_rows(conn, [other for other, _relation in capped])
    refs: list[ContextRef] = []
    for other, relation in capped:
        row = rows.get(other)
        if row is None:
            continue
        refs.append(
            ContextRef(
                doc_id=other,
                title=row["title"],
                repo=row["repo"],
                path=row["path"],
                relation=relation,
            )
        )
    return tuple(refs)
