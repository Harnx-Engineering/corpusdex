"""Coverage for the document link graph, PPR channel, and context assembly.

The pure graph maths is exercised on hand-built adjacency; everything else
runs against a real index built by ``indexer.reindex`` over a temporary
workspace, so link resolution, the table rebuild, and the search wiring are
tested end to end rather than against a mock.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import StubEmbedder, write_registry

from corpusdex import db, graph, indexer, links
from corpusdex import search as search_mod

FILLER = "notes about the watering schedule for the season. " * 6


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build(tmp_path: Path, files: dict[str, str]) -> tuple[Path, Path]:
    """Write ``files`` under a one-repo workspace and index it."""
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    for rel, text in files.items():
        _write(workspace / rel, text)
    write_registry(workspace, ["repo-a"])
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    return workspace, db_path


def _edges(conn) -> set[tuple[str, str, str]]:
    rows = conn.execute(
        "SELECT s.path AS src, t.path AS dst, l.relation AS relation FROM doc_links l "
        "JOIN documents s ON s.id = l.src_doc_id JOIN documents t ON t.id = l.dst_doc_id"
    ).fetchall()
    return {(row["src"], row["dst"], row["relation"]) for row in rows}


def _doc_id(conn, path: str) -> int:
    return conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()["id"]


# --- pure graph maths -------------------------------------------------------


def test_personalized_pagerank_is_deterministic_on_a_fixed_graph():
    adjacency = {1: {2: 1.0}, 2: {3: 1.0}, 3: {1: 1.0}, 4: {1: 1.0}}
    first = graph.personalized_pagerank(adjacency, [1])
    second = graph.personalized_pagerank(adjacency, [1])
    assert first == second
    assert set(first) == {1, 2, 3, 4}
    assert all(score >= 0.0 for score in first.values())


def test_personalized_pagerank_concentrates_mass_near_the_seed():
    # A path 1 -> 2 -> 3 seeded at 1: rank must decay with distance.
    adjacency = {1: {2: 1.0}, 2: {3: 1.0}, 3: {}}
    rank = graph.personalized_pagerank(adjacency, [1])
    assert rank[1] > rank[2] > rank[3]


def test_personalized_pagerank_returns_nothing_without_a_graph():
    assert graph.personalized_pagerank({}, [1]) == {}


def test_personalized_pagerank_returns_nothing_when_no_seed_is_in_the_graph():
    adjacency = {1: {2: 1.0}, 2: {1: 1.0}}
    assert graph.personalized_pagerank(adjacency, [99]) == {}


def test_personalized_pagerank_returns_nothing_without_seeds():
    assert graph.personalized_pagerank({1: {2: 1.0}, 2: {}}, []) == {}


def test_reverse_edges_are_walkable_at_reduced_weight():
    # Only 1 -> 2 is written, but seeding at 2 must still reach 1.
    adjacency = {1: {2: 1.0}, 2: {1: graph.REVERSE_EDGE_WEIGHT}}
    rank = graph.personalized_pagerank(adjacency, [2])
    assert rank[1] > 0.0


def test_a_hub_outranks_a_leaf_from_the_same_seeds():
    # Three seeds all point at 100; 200 is pointed at by one of them.
    adjacency: dict[int, dict[int, float]] = {}
    for seed in (1, 2, 3):
        adjacency.setdefault(seed, {})[100] = 1.0
        adjacency.setdefault(100, {})[seed] = graph.REVERSE_EDGE_WEIGHT
    adjacency.setdefault(1, {})[200] = 1.0
    adjacency.setdefault(200, {})[1] = graph.REVERSE_EDGE_WEIGHT
    rank = graph.personalized_pagerank(adjacency, [1, 2, 3])
    assert rank[100] > rank[200]


# --- graph construction at index time --------------------------------------


def test_reindex_builds_resolved_edges_and_drops_unresolved_targets(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/alpha.md": (
                f"# Alpha\n\n## Body\n\n{FILLER} see [[beta]] and [[nowhere]].\n"
            ),
            "repo-a/docs/beta.md": f"# Beta\n\n## Body\n\n{FILLER}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {("repo-a/docs/alpha.md", "repo-a/docs/beta.md", "links_to")}
        # The raw target is kept, but it never becomes an edge.
        targets = {row["target"] for row in conn.execute("SELECT target FROM doc_link_targets")}
        assert "nowhere" in targets
    finally:
        conn.close()


def test_reindex_resolves_bare_paths_and_markdown_links(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/alpha.md": (
                f"# Alpha\n\n## Body\n\n{FILLER} see docs/beta.md and [g](docs/gamma.md).\n"
            ),
            "repo-a/docs/beta.md": f"# Beta\n\n## Body\n\n{FILLER}\n",
            "repo-a/docs/gamma.md": f"# Gamma\n\n## Body\n\n{FILLER}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {
            ("repo-a/docs/alpha.md", "repo-a/docs/beta.md", "links_to"),
            ("repo-a/docs/alpha.md", "repo-a/docs/gamma.md", "links_to"),
        }
    finally:
        conn.close()


def test_superseded_by_frontmatter_becomes_a_typed_edge(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/old.md": f"---\nsuperseded_by: new\n---\n\n# Old\n\n## Body\n\n{FILLER}\n",
            "repo-a/docs/new.md": f"# New\n\n## Body\n\n{FILLER}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {("repo-a/docs/old.md", "repo-a/docs/new.md", "superseded_by")}
    finally:
        conn.close()


def test_self_links_are_not_edges(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {"repo-a/docs/alpha.md": f"# Alpha\n\n## Body\n\n{FILLER} see [[alpha]].\n"},
    )
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == set()
    finally:
        conn.close()


def test_graph_is_rebuilt_when_a_document_changes(tmp_path: Path):
    # A changed document is deleted and reinserted under a new id, so edges
    # pointing at it must be rebuilt rather than left dangling.
    # zeta.md exists purely so beta is not the highest document id: SQLite
    # would otherwise hand the reinserted row its old id back and the test
    # would not actually exercise id churn.
    workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/alpha.md": f"# Alpha\n\n## Body\n\n{FILLER} see [[beta]].\n",
            "repo-a/docs/beta.md": f"# Beta\n\n## Body\n\n{FILLER}\n",
            "repo-a/docs/zeta.md": f"# Zeta\n\n## Body\n\n{FILLER}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        before = _doc_id(conn, "repo-a/docs/beta.md")
    finally:
        conn.close()

    _write(workspace / "repo-a/docs/beta.md", f"# Beta\n\n## Body\n\n{FILLER} revised text.\n")
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.changed == 1

    conn, _vec = db.open_index(db_path)
    try:
        after = _doc_id(conn, "repo-a/docs/beta.md")
        assert after != before
        assert _edges(conn) == {("repo-a/docs/alpha.md", "repo-a/docs/beta.md", "links_to")}
        assert graph.load_adjacency(conn)[_doc_id(conn, "repo-a/docs/alpha.md")] == {after: 1.0}
    finally:
        conn.close()


def test_removing_a_document_removes_its_edges(tmp_path: Path):
    workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/alpha.md": f"# Alpha\n\n## Body\n\n{FILLER} see [[beta]].\n",
            "repo-a/docs/beta.md": f"# Beta\n\n## Body\n\n{FILLER}\n",
        },
    )
    (workspace / "repo-a/docs/beta.md").unlink()
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())

    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == set()
    finally:
        conn.close()


def test_reindex_reports_link_counts(tmp_path: Path):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(
        workspace / "repo-a/docs/alpha.md",
        f"# Alpha\n\n## Body\n\n{FILLER} see [[beta]] and [[nowhere]].\n",
    )
    _write(workspace / "repo-a/docs/beta.md", f"# Beta\n\n## Body\n\n{FILLER}\n")
    write_registry(workspace, ["repo-a"])
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 1
    assert stats.link_targets_unresolved == 1
    assert stats.link_targets_unlinkable == 0


# --- relative-path resolution (issue #6) ------------------------------------


def test_dot_and_dotdot_targets_resolve_against_the_linking_document(tmp_path: Path):
    # Path arithmetic, not a name lookup: `../context/card.md` from inside
    # decisions/ can only mean the sibling directory's file.
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/decisions/one.md": (
                f"# One\n\n## Body\n\n{FILLER} see [ctx](../context/card.md) and [two](./two.md).\n"
            ),
            "repo-a/docs/decisions/two.md": f"# Two\n\n## Body\n\n{FILLER}\n",
            "repo-a/docs/context/card.md": f"# Card\n\n## Body\n\n{FILLER}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {
            ("repo-a/docs/decisions/one.md", "repo-a/docs/context/card.md", "links_to"),
            ("repo-a/docs/decisions/one.md", "repo-a/docs/decisions/two.md", "links_to"),
        }
    finally:
        conn.close()


def test_a_relative_target_does_not_fall_back_to_a_name_match(tmp_path: Path):
    # `./card.md` says "the card.md next to me", and there is none. A card.md
    # DOES exist elsewhere in the repo, and the filename fallback would return
    # it, because normalize_target strips the leading `./` and leaves exactly
    # the bare name the fallback keys on. Falling through would answer a
    # question the document did not ask, so the miss has to stay a miss.
    #
    # The `./` form is load-bearing in this test. Written with `../context/`
    # it passes whether or not the fall-through exists, because `../` survives
    # normalization and matches no key either way: the test would assert
    # nothing. Confirmed by mutation.
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/decisions/one.md": (
                f"# One\n\n## Body\n\n{FILLER} see [ctx](./card.md).\n"
            ),
            "repo-a/docs/elsewhere/card.md": f"# Card\n\n## Body\n\n{FILLER}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == set()
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("repo-a/docs/decisions/one.md", "./two.md", "repo-a/docs/decisions/two.md"),
        ("repo-a/docs/decisions/one.md", "../context/card.md", "repo-a/docs/context/card.md"),
        ("repo-a/docs/one.md", "../README.md", "repo-a/README.md"),
        ("repo-a/README.md", "docs/one.md", None),
        ("repo-a/README.md", "[[wikilink]]", None),
        # Escapes above the workspace root, at three depths.
        ("repo-a/docs/one.md", "../../../secrets.md", None),
        ("repo-a/one.md", "../../x.md", None),
        ("one.md", "../x.md", None),
    ],
)
def test_resolve_relative_path_arithmetic(source: str, target: str, expected: str | None):
    # Asserted on the function rather than through an index, because the
    # escape rejection is not observable end to end: a path that escapes
    # normalizes to one starting with `..`, which matches no stored document
    # anyway, so removing the check changes no edge. Tested here the return
    # value is the outcome, and the rejection is pinned. Confirmed by
    # mutation: the end-to-end form of this test survived removing the check.
    assert links.resolve_relative_path(source, target) == expected


def test_a_bare_filename_resolves_in_its_own_repo_and_not_across_repos(tmp_path: Path):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(
        workspace / "repo-a/docs/one.md",
        f"# One\n\n## Body\n\n{FILLER} see [notes](notes.md).\n",
    )
    _write(workspace / "repo-a/docs/notes.md", f"# A notes\n\n## Body\n\n{FILLER}\n")
    _write(workspace / "repo-b/docs/notes.md", f"# B notes\n\n## Body\n\n{FILLER}\n")
    write_registry(workspace, ["repo-a", "repo-b"])
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {("repo-a/docs/one.md", "repo-a/docs/notes.md", "links_to")}
    finally:
        conn.close()


def test_an_ambiguous_in_repo_name_is_counted_as_declined_not_missing(tmp_path: Path):
    # Two candidates in the same repo. The resolver refuses to pick, which is
    # correct, and the count has to say so: reported as "names no document" it
    # reads as a content gap someone should close by writing a file, when the
    # only fix is a more specific reference.
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(
        workspace / "repo-a/docs/one.md",
        f"# One\n\n## Body\n\n{FILLER} see [notes](notes.md).\n",
    )
    _write(workspace / "repo-a/docs/x/notes.md", f"# X notes\n\n## Body\n\n{FILLER}\n")
    _write(workspace / "repo-a/docs/y/notes.md", f"# Y notes\n\n## Body\n\n{FILLER}\n")
    write_registry(workspace, ["repo-a"])
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0
    assert stats.link_targets_unlinkable == 1
    assert stats.link_targets_unresolved == 0


def test_a_remote_url_is_not_a_link_target(tmp_path: Path):
    # The inline-link pattern captured the whole URL, so a document citing a
    # file on GitHub stored a target that no corpus change could ever resolve
    # and that inflated the unresolved count permanently.
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(
        workspace / "repo-a/docs/one.md",
        f"# One\n\n## Body\n\n{FILLER} see "
        "[spec](https://example.invalid/org/repo/blob/main/docs/notes.md).\n",
    )
    _write(workspace / "repo-a/docs/notes.md", f"# Notes\n\n## Body\n\n{FILLER}\n")
    write_registry(workspace, ["repo-a"])
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0
    # Not counted at all, in either bucket: it never became a target.
    assert stats.link_targets_unresolved == 0
    assert stats.link_targets_unlinkable == 0
    conn, _vec = db.open_index(db_path)
    try:
        stored = [row["target"] for row in conn.execute("SELECT target FROM doc_link_targets")]
        assert stored == []
    finally:
        conn.close()


# --- the PPR channel inside search -----------------------------------------


HUB_QUERY = "sprinkler"


def _hub_corpus() -> dict[str, str]:
    """Three documents matching the query, all linking to a hub that does not."""
    files = {
        "repo-a/docs/hub.md": f"# Hub\n\n## Shared reference\n\n{FILLER} the shared reference.\n"
    }
    for name in ("one", "two", "three"):
        files[f"repo-a/docs/seed-{name}.md"] = (
            f"# Seed {name}\n\n## Schedule\n\n"
            f"{HUB_QUERY} {HUB_QUERY} {HUB_QUERY} {FILLER} see [[hub]].\n"
        )
    return files


def test_ppr_channel_promotes_a_linked_hub_that_never_matches_the_query(tmp_path: Path):
    _workspace, db_path = _build(tmp_path, _hub_corpus())
    conn, vec_ok = db.open_index(db_path)
    try:
        # The hub's own text contains no query term, so lexical retrieval
        # alone can never surface it.
        lexical = search_mod._lexical_ranked_ids(conn, HUB_QUERY, search_mod.CANDIDATE_POOL)
        hub_id = _doc_id(conn, "repo-a/docs/hub.md")
        hub_chunks = {
            row["id"] for row in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (hub_id,))
        }
        assert not (hub_chunks & set(lexical))

        response = search_mod.search(conn, vec_ok, HUB_QUERY, limit=10, embedder=StubEmbedder())
        assert hub_chunks & {hit.chunk_id for hit in response.hits}
    finally:
        conn.close()


def test_ppr_channel_is_seeded_only_from_fused_hits(tmp_path: Path):
    _workspace, db_path = _build(tmp_path, _hub_corpus())
    conn, _vec = db.open_index(db_path)
    try:
        assert graph.ppr_ranked_chunk_ids(conn, [], 50) == []
    finally:
        conn.close()


def test_ppr_channel_caps_chunks_per_document(tmp_path: Path):
    many = "\n\n".join(f"## Section {i}\n\n{FILLER}" for i in range(6))
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/alpha.md": f"# Alpha\n\n## Body\n\n{FILLER} see [[beta]].\n",
            "repo-a/docs/beta.md": f"# Beta\n\n{many}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        beta = _doc_id(conn, "repo-a/docs/beta.md")
        beta_chunks = [
            row["id"] for row in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (beta,))
        ]
        assert len(beta_chunks) > graph.MAX_CHUNKS_PER_DOC
        alpha_chunk = conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ?", (_doc_id(conn, "repo-a/docs/alpha.md"),)
        ).fetchone()["id"]
        ranked = graph.ppr_ranked_chunk_ids(conn, [alpha_chunk], 50)
        assert len([c for c in ranked if c in beta_chunks]) == graph.MAX_CHUNKS_PER_DOC
    finally:
        conn.close()


def test_search_without_links_contributes_nothing_and_still_works(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/alpha.md": f"# Alpha\n\n## Body\n\n{HUB_QUERY} {FILLER}\n",
            "repo-a/docs/beta.md": f"# Beta\n\n## Body\n\n{HUB_QUERY} {FILLER}\n",
        },
    )
    conn, vec_ok = db.open_index(db_path)
    try:
        assert graph.load_adjacency(conn) == {}
        chunk_ids = [row["id"] for row in conn.execute("SELECT id FROM chunks")]
        assert graph.ppr_ranked_chunk_ids(conn, chunk_ids, 50) == []
        response = search_mod.search(conn, vec_ok, HUB_QUERY, limit=10, embedder=StubEmbedder())
        assert response.hits
        assert all(hit.assembled == () for hit in response.hits)
    finally:
        conn.close()


def test_graph_channel_still_works_when_the_embedding_backend_is_down(
    tmp_path: Path, failing_embedder
):
    # Lexical-only degraded mode must keep the graph channel live: the graph
    # does not depend on the embedding backend at all.
    _workspace, db_path = _build(tmp_path, _hub_corpus())
    conn, vec_ok = db.open_index(db_path)
    try:
        response = search_mod.search(conn, vec_ok, HUB_QUERY, limit=10, embedder=failing_embedder)
        assert response.degraded is True
        # The channel that kept running has to be named. Reporting
        # MODE_LEXICAL here was the defect in issue #13: it told a caller the
        # page was lexical alone while the graph channel had re-ranked it.
        assert response.mode == db.MODE_LEXICAL_GRAPH
        assert response.channels_used == frozenset({"lexical", "graph"})
        hub_id = _doc_id(conn, "repo-a/docs/hub.md")
        hub_chunks = {
            row["id"] for row in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (hub_id,))
        }
        assert hub_chunks & {hit.chunk_id for hit in response.hits}
    finally:
        conn.close()


# --- context assembly -------------------------------------------------------


def _chain_corpus() -> dict[str, str]:
    return {
        "repo-a/docs/old.md": f"---\nsuperseded_by: mid\n---\n\n# Old\n\n## Body\n\n{FILLER}\n",
        "repo-a/docs/mid.md": (
            f"---\nsuperseded_by: new\n---\n\n# Mid\n\n## Body\n\n{FILLER} see [[side]].\n"
        ),
        "repo-a/docs/new.md": f"# New\n\n## Body\n\n{FILLER}\n",
        "repo-a/docs/side.md": f"# Side\n\n## Body\n\n{FILLER}\n",
    }


def test_assembly_walks_the_supersedence_chain_in_both_directions(tmp_path: Path):
    _workspace, db_path = _build(tmp_path, _chain_corpus())
    conn, _vec = db.open_index(db_path)
    try:
        refs = graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/mid.md"))
        by_path = {ref.path: ref.relation for ref in refs}
        assert by_path["repo-a/docs/new.md"] == graph.REL_SUPERSEDED_BY
        assert by_path["repo-a/docs/old.md"] == graph.REL_SUPERSEDES
    finally:
        conn.close()


def test_assembly_chain_is_transitive(tmp_path: Path):
    _workspace, db_path = _build(tmp_path, _chain_corpus())
    conn, _vec = db.open_index(db_path)
    try:
        refs = graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/old.md"))
        by_path = {ref.path: ref.relation for ref in refs}
        # old -> mid -> new: the far end of the chain is still reported.
        assert by_path["repo-a/docs/mid.md"] == graph.REL_SUPERSEDED_BY
        assert by_path["repo-a/docs/new.md"] == graph.REL_SUPERSEDED_BY
    finally:
        conn.close()


def test_assembly_includes_one_hop_links_in_both_directions(tmp_path: Path):
    _workspace, db_path = _build(tmp_path, _chain_corpus())
    conn, _vec = db.open_index(db_path)
    try:
        outward = graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/mid.md"))
        assert {ref.path: ref.relation for ref in outward}["repo-a/docs/side.md"] == (
            graph.REL_LINKS_TO
        )
        inward = graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/side.md"))
        assert {ref.path: ref.relation for ref in inward}["repo-a/docs/mid.md"] == (
            graph.REL_LINKED_FROM
        )
    finally:
        conn.close()


def test_assembly_reference_shape_is_compact(tmp_path: Path):
    _workspace, db_path = _build(tmp_path, _chain_corpus())
    conn, _vec = db.open_index(db_path)
    try:
        ref = graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/mid.md"))[0]
        assert ref.doc_id > 0
        assert ref.title
        assert ref.repo == "repo-a"
        assert ref.path.endswith(".md")
        assert ref.relation
    finally:
        conn.close()


def test_assembly_is_hard_capped(tmp_path: Path):
    links = " ".join(f"[[target-{i}]]" for i in range(20))
    files = {"repo-a/docs/hubdoc.md": f"# Hub doc\n\n## Body\n\n{FILLER} {links}\n"}
    for i in range(20):
        files[f"repo-a/docs/target-{i}.md"] = f"# Target {i}\n\n## Body\n\n{FILLER}\n"
    _workspace, db_path = _build(tmp_path, files)
    conn, _vec = db.open_index(db_path)
    try:
        refs = graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/hubdoc.md"))
        assert len(refs) == graph.MAX_ASSEMBLED_ITEMS
        refs_small = graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/hubdoc.md"), cap=3)
        assert len(refs_small) == 3
        assert graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/hubdoc.md"), cap=0) == ()
    finally:
        conn.close()


def test_assembly_terminates_on_a_supersedence_cycle(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/one.md": f"---\nsuperseded_by: two\n---\n\n# One\n\n## Body\n\n{FILLER}\n",
            "repo-a/docs/two.md": f"---\nsuperseded_by: one\n---\n\n# Two\n\n## Body\n\n{FILLER}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        refs = graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/one.md"))
        assert [ref.path for ref in refs] == ["repo-a/docs/two.md"]
    finally:
        conn.close()


def test_search_hits_carry_assembled_context(tmp_path: Path):
    _workspace, db_path = _build(tmp_path, _chain_corpus())
    conn, vec_ok = db.open_index(db_path)
    try:
        response = search_mod.search(conn, vec_ok, "watering schedule", limit=10)
        assembled = {hit.path: hit.assembled for hit in response.hits}
        assert any(refs for refs in assembled.values())
        for refs in assembled.values():
            assert len(refs) <= graph.MAX_ASSEMBLED_ITEMS
    finally:
        conn.close()


def test_get_chunk_carries_assembled_context(tmp_path: Path):
    _workspace, db_path = _build(tmp_path, _chain_corpus())
    conn, _vec = db.open_index(db_path)
    try:
        mid = _doc_id(conn, "repo-a/docs/mid.md")
        ref = conn.execute("SELECT ref FROM chunks WHERE doc_id = ?", (mid,)).fetchone()["ref"]
        hit = search_mod.get_chunk(conn, ref)
        assert hit is not None
        assert hit.doc_id == mid
        assert {ref.relation for ref in hit.assembled} >= {
            graph.REL_SUPERSEDED_BY,
            graph.REL_SUPERSEDES,
        }
    finally:
        conn.close()


@pytest.mark.parametrize("cap", [0, -1])
def test_assembly_with_a_non_positive_cap_is_empty(tmp_path: Path, cap: int):
    _workspace, db_path = _build(tmp_path, _chain_corpus())
    conn, _vec = db.open_index(db_path)
    try:
        assert graph.assemble_context(conn, _doc_id(conn, "repo-a/docs/mid.md"), cap=cap) == ()
    finally:
        conn.close()


def test_ppr_channel_prefers_chunks_the_other_channels_already_matched(tmp_path: Path):
    # A document's leading chunks are usually title and preamble. When the
    # base ranking already matched specific chunks of a promoted document,
    # the graph channel must vote for those, in that order, not for whatever
    # happens to come first by id.
    many = "\n\n".join(f"## Section {i}\n\n{FILLER}" for i in range(6))
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/alpha.md": f"# Alpha\n\n## Body\n\n{FILLER} see [[beta]].\n",
            "repo-a/docs/beta.md": f"# Beta\n\n{many}\n",
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        alpha_chunk = conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ?", (_doc_id(conn, "repo-a/docs/alpha.md"),)
        ).fetchone()["id"]
        beta_chunks = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ? ORDER BY id",
                (_doc_id(conn, "repo-a/docs/beta.md"),),
            )
        ]
        late_matches = [beta_chunks[4], beta_chunks[2]]
        ranked = graph.ppr_ranked_chunk_ids(conn, [alpha_chunk, *late_matches], 50)

        chosen = [c for c in ranked if c in beta_chunks]
        assert chosen[:2] == late_matches
        assert len(chosen) == graph.MAX_CHUNKS_PER_DOC
        # The third is padding, taken from the front as before.
        assert chosen[2] == beta_chunks[0]
    finally:
        conn.close()


def test_ppr_channel_seeds_only_from_the_head_of_the_candidate_list(tmp_path: Path):
    # The candidate list serves two purposes and they must stay separate: its
    # head seeds the walk, the whole list only orders chunk choice. Seeding
    # from the whole list lets weak tail candidates pull the walk somewhere
    # the query never pointed, which displaces strong hits instead of
    # extending them.
    files = {
        f"repo-a/docs/pad{i}.md": f"# Pad {i}\n\n{FILLER}\n" for i in range(graph.PPR_SEED_POOL)
    }
    files["repo-a/docs/alpha.md"] = f"# Alpha\n\n{FILLER} see [[beta]].\n"
    files["repo-a/docs/beta.md"] = f"# Beta\n\n{FILLER}\n"
    _workspace, db_path = _build(tmp_path, files)
    conn, _vec = db.open_index(db_path)
    try:
        pad_chunks = [
            conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ?",
                (_doc_id(conn, f"repo-a/docs/pad{i}.md"),),
            ).fetchone()["id"]
            for i in range(graph.PPR_SEED_POOL)
        ]
        alpha_chunk = conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ?", (_doc_id(conn, "repo-a/docs/alpha.md"),)
        ).fetchone()["id"]
        beta_chunk = conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ?", (_doc_id(conn, "repo-a/docs/beta.md"),)
        ).fetchone()["id"]

        # alpha sits past the seed pool: it orders chunks but must not seed.
        buried = graph.ppr_ranked_chunk_ids(conn, [*pad_chunks, alpha_chunk], 50)
        assert beta_chunk not in buried

        # The same alpha inside the head does seed, and promotes beta.
        surfaced = graph.ppr_ranked_chunk_ids(conn, [alpha_chunk, *pad_chunks], 50)
        assert beta_chunk in surfaced
    finally:
        conn.close()


def test_recent_computes_assembled_context(tmp_path: Path):
    # "not computed" must not masquerade as "no relations".
    _workspace, db_path = _build(tmp_path, _chain_corpus())
    conn, _vec = db.open_index(db_path)
    try:
        hits = search_mod.recent(conn, limit=10)
        assert any(hit.assembled for hit in hits)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("repo/docs/decisions/0006-a-slug.md", "0006"),
        ("decisions/0001-x.md", "0001"),
        # Right shape, wrong home: a four-digit prefix is a date in a gap
        # ledger and a version elsewhere, so the directory is what makes it a
        # decision number.
        ("repo/docs/gaps/0006-a-slug.md", None),
        ("repo/decisions/nested/0006-a-slug.md", None),
        # Right home, wrong shape.
        ("repo/decisions/006-a-slug.md", None),
        ("repo/decisions/00061-a-slug.md", None),
        ("repo/decisions/0006.md", None),
        ("repo/decisions/0006-.md", None),
    ],
)
def test_decision_number_needs_both_the_directory_and_the_prefix(path, expected):
    assert indexer._decision_number(path) == expected


def _adr_workspace(tmp_path: Path, citing_body: str, *, extra: dict[str, str] | None = None):
    """A workspace with one decision record and one document that cites it."""
    workspace = tmp_path / "workspace"
    _write(
        workspace / "repo-a/docs/decisions/0006-tenant-default.md",
        f"# Tenant default\n\n## Body\n\n{FILLER}\n",
    )
    _write(workspace / "repo-a/docs/citing.md", f"# Citing\n\n## Body\n\n{citing_body}\n")
    for rel, body in (extra or {}).items():
        _write(workspace / rel, body)
    write_registry(workspace, ["repo-a"])
    return workspace


def test_a_decision_cited_by_number_in_prose_becomes_an_edge(tmp_path: Path):
    # The form the corpus overwhelmingly uses to cite itself. It matches none
    # of the three path patterns, so before this it produced no edge at all.
    workspace = _adr_workspace(tmp_path, f"{FILLER} this extends decision 0006 in full.")
    db_path = tmp_path / "var" / "index.db"
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {
            ("repo-a/docs/citing.md", "repo-a/docs/decisions/0006-tenant-default.md", "links_to")
        }
    finally:
        conn.close()


@pytest.mark.parametrize(
    "keyword",
    ["decision", "decisions", "Decision", "ADR", "ADRs", "adr"],
)
def test_every_accepted_citation_keyword_produces_the_edge(tmp_path: Path, keyword: str):
    workspace = _adr_workspace(tmp_path, f"{FILLER} see {keyword} 0006 for the reasoning.")
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 1


def test_a_four_digit_number_without_a_keyword_is_not_a_citation(tmp_path: Path):
    # The keyword carries all of the precision. A bare four-digit number is a
    # year, a port, or an issue number far more often than a decision, and
    # 0006 here is a literal ADR number that still must not become an edge.
    workspace = _adr_workspace(
        tmp_path, f"{FILLER} in 2026 we bound port 0006 and closed issue 0006."
    )
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0
    assert stats.link_targets_unresolved == 0


def test_a_citation_list_names_every_record_in_it(tmp_path: Path):
    # "Related decisions: 0006 and 0007" is one match carrying two targets,
    # which is why the run is followed rather than stopping at the first.
    workspace = _adr_workspace(
        tmp_path,
        f"{FILLER} Related decisions: 0006 and 0007.",
        extra={
            "repo-a/docs/decisions/0007-other.md": f"# Other\n\n## Body\n\n{FILLER}\n",
        },
    )
    db_path = tmp_path / "var" / "index.db"
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {
            ("repo-a/docs/citing.md", "repo-a/docs/decisions/0006-tenant-default.md", "links_to"),
            ("repo-a/docs/citing.md", "repo-a/docs/decisions/0007-other.md", "links_to"),
        }
    finally:
        conn.close()


def test_a_numbered_document_outside_a_decisions_directory_does_not_answer_a_citation(
    tmp_path: Path,
):
    # A gap ledger named by date (2026-08-24-audit.md) and a decision record
    # named by number are the same shape. Only the directory separates them.
    workspace = tmp_path / "workspace"
    _write(
        workspace / "repo-a/docs/gaps/0006-audit.md",
        f"# Audit\n\n## Body\n\n{FILLER}\n",
    )
    _write(
        workspace / "repo-a/docs/citing.md",
        f"# Citing\n\n## Body\n\n{FILLER} this extends decision 0006 in full.\n",
    )
    write_registry(workspace, ["repo-a"])
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0
    assert stats.link_targets_unresolved == 1


def test_two_records_sharing_a_number_decline_the_citation(tmp_path: Path):
    # It has happened twice in the real corpus and merges cleanly, because the
    # slugs differ. The resolver must refuse rather than pick one, and the
    # refusal is counted as declined, not as a missing document.
    workspace = _adr_workspace(
        tmp_path,
        f"{FILLER} this extends decision 0006 in full.",
        extra={
            "repo-a/docs/decisions/0006-a-different-slug.md": (
                f"# Collision\n\n## Body\n\n{FILLER}\n"
            ),
        },
    )
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0
    assert stats.link_targets_unlinkable == 1
    assert stats.link_targets_unresolved == 0


def test_a_citation_of_a_number_no_record_carries_is_unresolved(tmp_path: Path):
    workspace = _adr_workspace(tmp_path, f"{FILLER} this extends decision 0099 in full.")
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0
    assert stats.link_targets_unresolved == 1
    assert stats.link_targets_unlinkable == 0


def test_a_record_linked_by_its_filename_resolves_by_name_not_by_number(tmp_path: Path):
    # A wikilink names the record's filename stem, which BEGINS with four
    # digits. The number branch tests the target with `fullmatch` precisely so
    # it cannot claim this: relaxed to `match` it would look up the whole stem
    # in the number map, miss, and report a working link as unresolved.
    workspace = _adr_workspace(tmp_path, f"{FILLER} see [[0006-tenant-default]] for the reasoning.")
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 1
    assert stats.link_targets_unresolved == 0


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tasks/review-ledger.md", True),
        ("some-repo/tasks/lessons.md", True),
        ("repo/docs/tasks/notes.md", True),
        # Not logs: the word has to be a directory, not a filename or a prefix.
        ("repo/docs/tasks.md", False),
        ("repo/tasksy/notes.md", False),
        ("the-brain/context/some-repo.md", False),
        ("the-brain/decisions/0006-x.md", False),
    ],
)
def test_work_log_is_identified_by_directory_not_by_name(path, expected):
    assert indexer._is_work_log(path) is expected


def test_a_decision_cited_from_a_work_log_is_not_an_edge(tmp_path: Path):
    # A ledger cites a decision because it was worked on, which says nothing
    # about subject. Measured: dropping these moved full recall 0.722 to 0.745
    # on the 36-query judged set with every other arm held identical.
    workspace = tmp_path / "workspace"
    _write(
        workspace / "repo-a/docs/decisions/0006-tenant-default.md",
        f"# Tenant default\n\n## Body\n\n{FILLER}\n",
    )
    _write(
        workspace / "repo-a/tasks/ledger.md",
        f"# Ledger\n\n## Body\n\n{FILLER} worked decision 0006 today.\n",
    )
    write_registry(workspace, ["repo-a"])
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0
    assert stats.link_targets_from_work_logs == 1
    # Counted apart from both existing reasons: it names a real document, and
    # the resolver never declined to guess, so either would be a false report.
    assert stats.link_targets_unresolved == 0
    assert stats.link_targets_unlinkable == 0


def test_a_work_log_can_still_link_by_path(tmp_path: Path):
    # Only the number citation is declined. A ledger that writes an actual
    # path is making a real reference and keeps its edge, so the rule cannot
    # be widened into "work logs have no links" without failing here.
    workspace = tmp_path / "workspace"
    _write(
        workspace / "repo-a/docs/decisions/0006-tenant-default.md",
        f"# Tenant default\n\n## Body\n\n{FILLER}\n",
    )
    _write(
        workspace / "repo-a/tasks/ledger.md",
        f"# Ledger\n\n## Body\n\n{FILLER} see repo-a/docs/decisions/0006-tenant-default.md.\n",
    )
    write_registry(workspace, ["repo-a"])
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 1
    assert stats.link_targets_from_work_logs == 0


def test_a_curated_card_citing_many_decisions_keeps_every_edge(tmp_path: Path):
    # The case a fan-out cap gets wrong. Six citations from a context card is
    # the shape the real repo cards have, and truncating them was measured to
    # collapse the paraphrase group's MRR contribution from +0.151 to +0.068.
    workspace = tmp_path / "workspace"
    numbers = ["0001", "0002", "0003", "0004", "0005", "0006"]
    for n in numbers:
        _write(
            workspace / f"repo-a/docs/decisions/{n}-slug.md",
            f"# Record {n}\n\n## Body\n\n{FILLER}\n",
        )
    cited = ", ".join(numbers)
    _write(
        workspace / "repo-a/docs/card.md",
        f"# Card\n\n## Body\n\n{FILLER} Related decisions: {cited}.\n",
    )
    write_registry(workspace, ["repo-a"])
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == len(numbers)
    assert stats.link_targets_from_work_logs == 0


# ---------------------------------------------------------------------------
# An unresolved supersedence claim is reported by name (issue #21)
#
# The ranking penalty is derived from the resolved ``superseded_by`` edge, so
# a frontmatter value that resolves to nothing now means NO penalty. That is
# the correct half of the fix -- the penalty and the displayed successor read
# one source and cannot disagree -- but it fails silently in the opposite
# direction: a record asserts it was replaced and is ranked as though it never
# was. These tests pin the compensating report.
# ---------------------------------------------------------------------------


def test_a_superseded_by_naming_no_document_is_reported_by_name(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/retired.md": (
                f"---\nsuperseded_by: a-document-that-does-not-exist.md\n---\n\n"
                f"# Retired\n\n## Body\n\n{FILLER}\n"
            ),
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        result = indexer.rebuild_link_graph(conn)
    finally:
        conn.close()
    # The path, not a count: naming the document is the entire point.
    assert result.superseded_by_unresolved == ("repo-a/docs/retired.md",)


def test_a_resolved_superseded_by_is_not_reported(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/active.md": f"# Active\n\n## Body\n\n{FILLER}\n",
            "repo-a/docs/retired.md": (
                f"---\nsuperseded_by: active.md\n---\n\n# Retired\n\n## Body\n\n{FILLER}\n"
            ),
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        result = indexer.rebuild_link_graph(conn)
        assert result.superseded_by_unresolved == ()
        assert (
            "repo-a/docs/retired.md",
            "repo-a/docs/active.md",
            "superseded_by",
        ) in _edges(conn)
    finally:
        conn.close()


def test_an_unresolved_ordinary_link_is_not_reported_as_supersedence(tmp_path: Path):
    """Only the ``superseded_by`` relation is reported.

    Most unresolved targets in a real corpus are ordinary body links, and they
    are already summarised by the unresolved counter. Reporting them here too
    would bury the supersedence claims this exists to surface.
    """
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/citing.md": (
                f"# Citing\n\n## Body\n\n{FILLER} see repo-a/docs/absent.md for more.\n"
            ),
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        result = indexer.rebuild_link_graph(conn)
        assert result.unresolved == 1
        assert result.superseded_by_unresolved == ()
    finally:
        conn.close()


def test_a_self_superseding_document_is_reported_too(tmp_path: Path):
    """A record naming itself is a broken claim with the same consequence.

    It is counted as unlinkable rather than unresolved, so a report keyed on
    the unresolved branch alone would miss it, while the document is equally
    left claiming a supersedence that no edge carries.
    """
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/loop.md": (
                f"---\nsuperseded_by: loop.md\n---\n\n# Loop\n\n## Body\n\n{FILLER}\n"
            ),
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        result = indexer.rebuild_link_graph(conn)
        assert result.superseded_by_unresolved == ("repo-a/docs/loop.md",)
    finally:
        conn.close()


def test_every_unresolved_supersedence_is_reported_and_sorted(tmp_path: Path):
    _workspace, db_path = _build(
        tmp_path,
        {
            "repo-a/docs/zeta.md": (
                f"---\nsuperseded_by: nowhere.md\n---\n\n# Zeta\n\n## Body\n\n{FILLER}\n"
            ),
            "repo-a/docs/alpha.md": (
                f"---\nsuperseded_by: nowhere.md\n---\n\n# Alpha\n\n## Body\n\n{FILLER}\n"
            ),
        },
    )
    conn, _vec = db.open_index(db_path)
    try:
        result = indexer.rebuild_link_graph(conn)
        assert result.superseded_by_unresolved == (
            "repo-a/docs/alpha.md",
            "repo-a/docs/zeta.md",
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Citation forms: what the corpus actually writes, and what must stay unmatched
# (findings from the cross-model review of the #29/#33 citation work)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keyword", ["record", "records", "Record"])
def test_record_is_not_a_citation_keyword(tmp_path: Path, keyword: str):
    """ "record 2026" is ordinary English; "decision 2026" is not.

    The keyword set is the entire false-positive defence, so a keyword that
    reads naturally in front of a year has to be excluded or the defence is
    only as good as the corpus's luck. The corpus cites nothing this way.
    """
    workspace = _adr_workspace(tmp_path, f"{FILLER} see {keyword} 0006 for the reasoning.")
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0


def test_the_annotated_list_form_the_corpus_actually_writes_names_every_record(tmp_path: Path):
    """``0006 (why), 0011 (why), 0016`` must yield all three.

    This is the form ``decisions/0018`` writes. The unannotated
    ``0006 and 0011`` that the original test used appears nowhere in the
    corpus, so the earlier test passed against a form nothing produces while
    the real one silently stopped at the first number.
    """
    workspace = _adr_workspace(
        tmp_path,
        f"{FILLER}\n\nRelated decisions: 0006 (the defect class this extends), "
        f"0011 (why reachability matters), 0016 (detection is live).",
        extra={
            "repo-a/docs/decisions/0011-reachability.md": (
                f"# Reachability\n\n## Body\n\n{FILLER}\n"
            ),
            "repo-a/docs/decisions/0016-detection-live.md": (
                f"# Detection\n\n## Body\n\n{FILLER}\n"
            ),
        },
    )
    db_path = tmp_path / "var" / "index.db"
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {
            ("repo-a/docs/citing.md", "repo-a/docs/decisions/0006-tenant-default.md", "links_to"),
            ("repo-a/docs/citing.md", "repo-a/docs/decisions/0011-reachability.md", "links_to"),
            ("repo-a/docs/citing.md", "repo-a/docs/decisions/0016-detection-live.md", "links_to"),
        }
    finally:
        conn.close()


def test_an_oxford_comma_does_not_truncate_the_list(tmp_path: Path):
    """``0006, 0011, and 0016``: the ", and" separator is a real corpus form."""
    workspace = _adr_workspace(
        tmp_path,
        f"{FILLER}\n\nSee decisions 0006, 0011, and 0016 for the reasoning.",
        extra={
            "repo-a/docs/decisions/0011-reachability.md": (
                f"# Reachability\n\n## Body\n\n{FILLER}\n"
            ),
            "repo-a/docs/decisions/0016-detection-live.md": (
                f"# Detection\n\n## Body\n\n{FILLER}\n"
            ),
        },
    )
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 3


def test_the_hyphen_form_is_not_a_citation(tmp_path: Path):
    """``ADR-0003`` must produce NO edge, and that is deliberate.

    The context cards use the hyphen form exclusively for ANOTHER repo's
    numbering (``context/some-repo.md`` cites some-repo's ADR-0001, 0003,
    0008 and 0009), and those records are not in this corpus. Matching it
    would resolve all five onto knowledge-repo decisions carrying the same
    numbers and unrelated subjects. Producing nothing is the correct answer,
    so it is pinned rather than left to chance.
    """
    workspace = _adr_workspace(tmp_path, f"{FILLER} see ADR-0006 for the reasoning.")
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0


def test_a_citation_inside_a_double_backtick_span_is_not_an_edge(tmp_path: Path):
    """Documentation about the syntax must not become an instance of it.

    The inline-code pattern used to be ``r"`[^`]*`"``, which matched the two
    LEADING backticks of a double-backtick span as an empty span, blanked
    them, and left the citation behind as prose.
    """
    workspace = _adr_workspace(tmp_path, f"{FILLER} write ``decision 0006`` to cite it.")
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 0


def test_a_work_log_declaring_its_own_supersedence_is_not_dropped(tmp_path: Path):
    """The work-log rule drops MENTIONS, not status claims about the log itself.

    The rule keys on the source being an append-only ledger, which is the
    right reason to ignore a citation it makes in passing. A ``superseded_by``
    is not a passing mention: dropping it would remove the ranking penalty and
    also skip the unresolved-supersedence report, leaving no trace anywhere.
    """
    workspace = tmp_path / "workspace"
    _write(
        workspace / "repo-a/docs/decisions/0006-tenant-default.md",
        f"# Tenant default\n\n## Body\n\n{FILLER}\n",
    )
    _write(
        workspace / "repo-a/tasks/todo.md",
        f"---\nsuperseded_by: '0006'\n---\n\n# Todo\n\n## Body\n\n{FILLER}\n",
    )
    write_registry(workspace, ["repo-a"])
    db_path = tmp_path / "var" / "index.db"
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert (
            "repo-a/tasks/todo.md",
            "repo-a/docs/decisions/0006-tenant-default.md",
            "superseded_by",
        ) in _edges(conn)
    finally:
        conn.close()


def _two_repo_adr_workspace(tmp_path: Path, *, citing_repo: str, body: str):
    """Two registered repos that each number a decision 0006, plus a citer.

    This is the collision that stops being hypothetical the moment more than
    one repo's ``decisions/`` tree is in the corpus. A decision number is a
    repo-LOCAL identifier: every repo numbers its records from 0001, so the
    overlap is the normal case, not an accident. In the live workspace two
    repos overlap on eleven numbers.
    """
    workspace = tmp_path / "workspace"
    _write(
        workspace / "repo-a/decisions/0006-tenant-default.md",
        f"# Tenant default\n\n## Body\n\n{FILLER}\n",
    )
    _write(
        workspace / "repo-b/decisions/0006-binary-name.md",
        f"# Binary name\n\n## Body\n\n{FILLER}\n",
    )
    _write(workspace / f"{citing_repo}/docs/citing.md", f"# Citing\n\n## Body\n\n{body}\n")
    write_registry(workspace, ["repo-a", "repo-b"])
    return workspace


def test_a_decision_number_resolves_inside_the_citing_documents_own_repo(tmp_path: Path):
    """The number means the citer's own record, and no ambiguity rule can see that.

    Both candidates are equally good workspace-wide, so declining, which is the
    correct answer for an ambiguous NAME, is the wrong answer here. Measured on
    the live corpus: without this, widening the walk destroys 25 existing edges,
    all of them context cards citing their own repo's decisions, which are the
    graph's hub documents.
    """
    workspace = _two_repo_adr_workspace(
        tmp_path, citing_repo="repo-a", body=f"{FILLER} this extends decision 0006 in full."
    )
    db_path = tmp_path / "var" / "index.db"
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {
            ("repo-a/docs/citing.md", "repo-a/decisions/0006-tenant-default.md", "links_to")
        }
    finally:
        conn.close()


def test_the_same_number_from_the_other_repo_resolves_to_that_repos_record(tmp_path: Path):
    """The mirror arm. Without it the test above passes on a resolver that
    simply always picks repo-a, e.g. by iteration order over the candidates."""
    workspace = _two_repo_adr_workspace(
        tmp_path, citing_repo="repo-b", body=f"{FILLER} this extends decision 0006 in full."
    )
    db_path = tmp_path / "var" / "index.db"
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {
            ("repo-b/docs/citing.md", "repo-b/decisions/0006-binary-name.md", "links_to")
        }
    finally:
        conn.close()


def test_two_records_sharing_a_number_inside_one_repo_are_still_declined(tmp_path: Path):
    """Repo scoping must not turn the collision it fixes into a licence to guess.

    Two records taking one number INSIDE a repo has happened three times here,
    and the resolver has always declined it. What this test pins is the OUTCOME
    (no edge) and the COUNTER it lands in (unlinkable, not unresolved), because
    the two call for opposite responses from whoever reads the report.

    What it deliberately does NOT claim, having been checked: it does not pin
    that the repo bucket declines rather than falling through to the global
    one. Those two are equivalent by construction, since the repo bucket is a
    subset of the global bucket, so an ambiguous repo bucket guarantees an
    ambiguous global bucket. A mutant that replaces the early return with a
    fall-through SURVIVES this test, and no fixture can change that. Said out
    loud because a test whose name promises more than it delivers is worse than
    an absent one.
    """
    workspace = tmp_path / "workspace"
    _write(
        workspace / "repo-a/decisions/0006-tenant-default.md",
        f"# Tenant default\n\n## Body\n\n{FILLER}\n",
    )
    _write(
        workspace / "repo-a/decisions/0006-a-second-record-taking-the-same-number.md",
        f"# Second record\n\n## Body\n\n{FILLER}\n",
    )
    _write(
        workspace / "repo-a/docs/citing.md",
        f"# Citing\n\n## Body\n\n{FILLER} this extends decision 0006 in full.\n",
    )
    write_registry(workspace, ["repo-a"])
    db_path = tmp_path / "var" / "index.db"
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == set()
    finally:
        conn.close()
    # Declined, not missing: the two counters call for opposite responses, and
    # reporting this as "unresolved" would send someone looking for a document
    # that exists twice.
    assert stats.link_targets_unlinkable == 1
    assert stats.link_targets_unresolved == 0


def test_a_repo_with_no_decisions_of_its_own_still_resolves_workspace_wide(tmp_path: Path):
    """The global fallback is load-bearing, not vestigial.

    The workspace root and the skills directory hold no ``decisions/``, so
    every decision citation they make can only be answered globally. A repo
    bucket miss must fall through rather than decline.
    """
    workspace = tmp_path / "workspace"
    _write(
        workspace / "repo-a/decisions/0006-tenant-default.md",
        f"# Tenant default\n\n## Body\n\n{FILLER}\n",
    )
    _write(
        workspace / "repo-b/docs/citing.md",
        f"# Citing\n\n## Body\n\n{FILLER} this extends decision 0006 in full.\n",
    )
    write_registry(workspace, ["repo-a", "repo-b"])
    db_path = tmp_path / "var" / "index.db"
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    conn, _vec = db.open_index(db_path)
    try:
        assert _edges(conn) == {
            ("repo-b/docs/citing.md", "repo-a/decisions/0006-tenant-default.md", "links_to")
        }
    finally:
        conn.close()
