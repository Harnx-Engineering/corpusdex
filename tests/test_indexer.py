from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from conftest import write_registry

from corpusdex import chunker, db, indexer, links


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_workspace(root: Path, extra_repos: list[str] | None = None) -> None:
    _write(root / "repo-a" / "AGENTS.md", "# Repo A agents\n\n## Rules\n\n" + "text " * 20)
    _write(root / "repo-a" / "README.md", "# Repo A\n\n## Overview\n\n" + "text " * 20)
    _write(
        root / "repo-a" / "docs" / "guide.md",
        "# Guide\n\n## Setup\n\n" + "setup instructions " * 20,
    )
    _write(root / "repo-a" / "docs" / "nested" / "deep.md", "# Deep\n\n" + "deep text " * 20)
    _write(root / "repo-a" / "tasks" / "todo.md", "# Todo\n\n" + "todo text " * 20)
    # Not corpus: no matching pattern (a bare file at repo root that is not in
    # ROOT_PROSE_FILES).
    _write(root / "repo-a" / "NOTES.md", "# Notes\n\nshould not be indexed\n")
    # Not corpus: inside a skipped directory.
    _write(root / "repo-a" / "node_modules" / "pkg" / "README.md", "# pkg\n\nvendored\n")
    _write(root / ".worktrees" / "wt1" / "AGENTS.md", "# worktree agents\n\nskip me\n")
    _write(root / "repo-a" / "var" / "generated.md", "# generated\n\nskip me\n")
    # A stale, unregistered sibling checkout: must not be indexed even though
    # it looks exactly like a real repo, because it is not in the registry.
    _write(root / "repo-a-stale-checkout" / "AGENTS.md", "# Stale copy\n\njunk\n")
    write_registry(root, ["repo-a", *(extra_repos or [])])


def test_discover_corpus_matches_expected_patterns(tmp_path: Path):
    _build_workspace(tmp_path)
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path for d in docs}

    assert "repo-a/AGENTS.md" in rel_paths
    assert "repo-a/README.md" in rel_paths
    assert "repo-a/docs/guide.md" in rel_paths
    assert "repo-a/docs/nested/deep.md" in rel_paths
    assert "repo-a/tasks/todo.md" in rel_paths

    assert "repo-a/NOTES.md" not in rel_paths
    assert not any("node_modules" in p for p in rel_paths)
    assert not any(".worktrees" in p for p in rel_paths)
    assert not any(p.startswith("repo-a/var/") for p in rel_paths)
    # Not in the registry, so not indexed even though it exists on disk.
    assert not any(p.startswith("repo-a-stale-checkout/") for p in rel_paths)

    repo_of = {d.rel_path: d.repo for d in docs}
    assert repo_of["repo-a/AGENTS.md"] == "repo-a"


def test_discover_corpus_raises_when_registry_missing(tmp_path: Path):
    _write(tmp_path / "repo-a" / "AGENTS.md", "# Repo A agents\n\n" + "text " * 20)
    # Deliberately no .harnx-repos.tsv.
    with pytest.raises(indexer.RegistryMissing):
        indexer.discover_corpus(tmp_path)


def test_discover_corpus_skips_registered_name_that_is_also_a_skip_dir(tmp_path: Path):
    # A pathological registry row naming a skip-dir must not cause a scan of
    # e.g. workspace_root/var as if it were a repo.
    write_registry(tmp_path, ["repo-a", "var"])
    _write(tmp_path / "repo-a" / "AGENTS.md", "# Repo A\n\n" + "text " * 20)
    docs = indexer.discover_corpus(tmp_path)
    assert all(d.repo != "var" for d in docs)


def test_discover_corpus_matches_nested_docs_dirs(tmp_path: Path):
    # docs/ nested one level inside the repo (e.g. a package subdirectory),
    # not only a top-level docs/.
    _write(
        tmp_path / "repo-a" / "service-a" / "docs" / "setup-guide.md",
        "# Auth and roles\n\n" + "text " * 20,
    )
    write_registry(tmp_path, ["repo-a"])
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path for d in docs}
    assert "repo-a/service-a/docs/setup-guide.md" in rel_paths


def test_discover_corpus_nested_docs_scan_does_not_cross_into_unregistered_siblings(
    tmp_path: Path,
):
    # Regression: the nested-docs walk (any depth of docs/ inside a repo) is
    # only safe when it is bounded to a single registered repo's own
    # directory tree. Reusing the same full-tree walk for the workspace
    # root's own docs match would cross into every sibling directory,
    # including an unregistered worktree checkout that happens to have its
    # own nested docs/ dir, silently reintroducing whole-workspace scanning.
    _write(tmp_path / "repo-a" / "AGENTS.md", "# Repo A\n\n" + "text " * 20)
    _write(
        tmp_path / "repo-a-some-other-worktree" / "service-a" / "docs" / "leaked.md",
        "# Leaked\n\nshould not be indexed\n",
    )
    write_registry(tmp_path, ["repo-a"])
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path for d in docs}
    assert not any("repo-a-some-other-worktree" in p for p in rel_paths)


def test_discover_corpus_skips_nested_claude_worktree_checkouts(tmp_path: Path):
    # Regression: per the workspace AGENTS.md worktree convention,
    # Claude-managed ephemeral worktrees live at
    # <repo>/.claude/worktrees/<agent-id>/, a full duplicate checkout nested
    # inside the registered repo's own directory tree (not a workspace-root
    # sibling, so the registry-scoping fix alone does not exclude it). The
    # repo's own deep docs walk must not index this duplicate copy alongside
    # the real one.
    _write(tmp_path / "repo-a" / "AGENTS.md", "# Repo A\n\n" + "text " * 20)
    _write(
        tmp_path / "repo-a" / "service-a" / "docs" / "real.md",
        "# Real\n\n" + "text " * 20,
    )
    _write(
        tmp_path
        / "repo-a"
        / ".claude"
        / "worktrees"
        / "agent-abc123"
        / "service-a"
        / "docs"
        / "real.md",
        "# Duplicate\n\n" + "text " * 20,
    )
    write_registry(tmp_path, ["repo-a"])
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path for d in docs}
    assert "repo-a/service-a/docs/real.md" in rel_paths
    assert not any("worktrees" in p for p in rel_paths)


def test_discover_corpus_matches_workspace_roots_own_docs_dir(tmp_path: Path):
    # The workspace root's own docs/ (a direct child, not deep) must still be
    # indexed, distinct from the deep-scan behaviour reserved for registered
    # repos.
    _write(tmp_path / "docs" / "workspace-guide.md", "# Workspace guide\n\n" + "text " * 20)
    _write(
        tmp_path / "docs" / "nested" / "deep-workspace-doc.md",
        "# Deep workspace doc\n\n" + "text " * 20,
    )
    write_registry(tmp_path, [])
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path for d in docs}
    assert "docs/workspace-guide.md" in rel_paths
    assert "docs/nested/deep-workspace-doc.md" in rel_paths


def test_discover_corpus_indexes_all_of_the_knowledge_repo_recursively(tmp_path, monkeypatch):
    """The knowledge repo is the one whose whole tree is corpus. Every other
    repo contributes only its documented subset, so ``glossary.md`` at the
    root of a repo is indexed here and would not be anywhere else."""
    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "the-brain")
    _write(tmp_path / "the-brain" / "decisions" / "0001-x.md", "# 0001\n\n" + "x " * 20)
    _write(tmp_path / "the-brain" / "glossary.md", "# Glossary\n\n" + "y " * 20)
    _write(tmp_path / "repo-a" / "glossary.md", "# Other glossary\n\n" + "y " * 20)
    write_registry(tmp_path, ["the-brain", "repo-a"])
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path for d in docs}
    assert "the-brain/decisions/0001-x.md" in rel_paths
    assert "the-brain/glossary.md" in rel_paths
    assert "repo-a/glossary.md" not in rel_paths


def test_discover_corpus_finds_skills_and_workspace_root_files(tmp_path: Path):
    _write(tmp_path / "AGENTS.md", "# Workspace agents\n\n" + "z " * 20)
    _write(tmp_path / ".claude" / "skills" / "foo" / "SKILL.md", "# Foo skill\n\n" + "w " * 20)
    write_registry(tmp_path, [])
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path: d.repo for d in docs}
    assert rel_paths["AGENTS.md"] == indexer.WORKSPACE_REPO
    assert rel_paths[".claude/skills/foo/SKILL.md"] == indexer.SKILLS_REPO


def test_classify_doc_type():
    assert indexer.classify_doc_type("the-brain", "decisions/0001-x.md", "the-brain") == "decision"
    assert indexer.classify_doc_type("the-brain", "gaps/2026-audit.md", "the-brain") == "gap"
    # The same directory name in a repo that is not the knowledge repo carries
    # no such meaning, so it must not inherit the classification.
    assert indexer.classify_doc_type("repo-a", "decisions/0001-x.md", "the-brain") == "doc"
    assert indexer.classify_doc_type("repo-a", "AGENTS.md") == "agents"
    assert indexer.classify_doc_type("repo-a", "README.md") == "readme"
    assert indexer.classify_doc_type("repo-a", "docs/guide.md") == "doc"
    assert indexer.classify_doc_type("repo-a", "tasks/todo.md") == "task"
    assert indexer.classify_doc_type(".claude", "skills/foo/SKILL.md") == "skill"


def test_classify_doc_type_handles_workspace_relative_rel_path():
    # Regression: discover_corpus's CorpusDoc.rel_path is workspace-relative
    # (repo-prefixed), which is exactly what reindex() passes to
    # classify_doc_type. Without stripping the prefix first, every
    # knowledge-repo decisions/context/architecture/gaps file misclassified as
    # generic "doc".
    def kind(rel_path: str, repo: str = "the-brain") -> str:
        return indexer.classify_doc_type(repo, rel_path, "the-brain")

    assert kind("the-brain/decisions/0001-x.md") == "decision"
    assert kind("the-brain/gaps/2026-audit.md") == "gap"
    assert kind("the-brain/context/repo-a.md") == "context"
    assert kind("the-brain/architecture/map.md") == "architecture"
    assert kind("repo-a/AGENTS.md", "repo-a") == "agents"
    assert kind("repo-a/docs/guide.md", "repo-a") == "doc"


# ---------------------------------------------------------------------------
# Reindex: idempotence, incrementality, removal
# ---------------------------------------------------------------------------


def test_reindex_first_run_indexes_everything(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.added == stats.docs_seen
    assert stats.changed == 0
    assert stats.removed == 0
    assert stats.unchanged == 0
    assert stats.chunks_written > 0


def test_reindex_second_run_over_unchanged_corpus_is_a_no_op(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    first = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    second = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert second.added == 0
    assert second.changed == 0
    assert second.removed == 0
    assert second.unchanged == first.docs_seen
    assert second.touched == 0
    assert second.embedded_chunks == 0  # nothing new to backfill


def test_reindex_detects_changed_content(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    # Force a distinct mtime and hash.
    time.sleep(0.01)
    target = workspace / "repo-a" / "AGENTS.md"
    target.write_text("# Repo A agents\n\n## Rules\n\n" + "changed content " * 20, encoding="utf-8")
    os.utime(target, None)

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert stats.changed == 1
    assert stats.added == 0
    assert stats.removed == 0

    conn = db.connect(db_path)
    try:
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE path = 'repo-a/AGENTS.md'"
        ).fetchone()
        assert row is not None
        rows = conn.execute(
            "SELECT c.body FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE d.path = 'repo-a/AGENTS.md'"
        ).fetchall()
        assert any("changed content" in r["body"] for r in rows)
    finally:
        conn.close()


def test_reindex_content_unchanged_mtime_touched_counts_as_unchanged(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    target = workspace / "repo-a" / "AGENTS.md"
    original_text = target.read_text(encoding="utf-8")
    time.sleep(0.01)
    target.write_text(original_text, encoding="utf-8")  # same bytes, new mtime

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert stats.changed == 0
    assert stats.added == 0
    assert stats.unchanged == stats.docs_seen


def test_reindex_same_mtime_different_size_is_detected(tmp_path: Path, stub_embedder):
    # The mtime-only fast path would wrongly skip this edit if the file's
    # mtime is pinned back to its original value after a content change of a
    # different length; the size gate must catch what mtime alone misses.
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    target = workspace / "repo-a" / "AGENTS.md"
    original_stat = target.stat()
    target.write_text(
        "# Repo A agents\n\n## Rules\n\n" + "substantially different longer content " * 20,
        encoding="utf-8",
    )
    os.utime(target, (original_stat.st_atime, original_stat.st_mtime))
    assert target.stat().st_mtime == original_stat.st_mtime
    assert target.stat().st_size != original_stat.st_size

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert stats.changed == 1


def test_reindex_removes_deleted_documents(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    first = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    (workspace / "repo-a" / "tasks" / "todo.md").unlink()

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert stats.removed == 1
    assert stats.unchanged == first.docs_seen - 1

    conn = db.connect(db_path)
    try:
        row = conn.execute("SELECT 1 FROM documents WHERE path = 'repo-a/tasks/todo.md'").fetchone()
        assert row is None
    finally:
        conn.close()


def test_reindex_full_forces_rechunk_of_unchanged_docs(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    first = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    stats = indexer.reindex(
        db_path=db_path, workspace_root=workspace, full=True, embedder=stub_embedder
    )
    # --full builds a replacement index in a scratch file and swaps it in, so
    # every document is a fresh insert rather than an in-place rewrite. What
    # matters is that nothing was skipped as unchanged.
    assert stats.added == first.docs_seen
    assert stats.changed == 0
    assert stats.unchanged == 0
    assert stats.chunks_written == first.chunks_written


def test_reindex_records_meta(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    conn = db.connect(db_path)
    try:
        assert db.get_meta(conn, db.META_LAST_REINDEX) is not None
        assert db.get_meta(conn, db.META_EMBED_STATUS) in {
            db.EMBED_STATUS_READY,
            db.EMBED_STATUS_UNAVAILABLE,
            db.EMBED_STATUS_DISABLED,
        }
    finally:
        conn.close()


def test_reindex_indexes_lexically_when_embedder_unavailable(tmp_path: Path, failing_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=failing_embedder)

    assert stats.added == stats.docs_seen
    assert stats.embedding_available is False
    assert stats.embedded_chunks == 0

    conn = db.connect(db_path)
    try:
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        assert chunk_count > 0
        assert db.get_meta(conn, db.META_EMBED_STATUS) in {
            db.EMBED_STATUS_UNAVAILABLE,
            db.EMBED_STATUS_DISABLED,
        }
    finally:
        conn.close()


def test_reindex_embeds_chunks_when_vec_available(tmp_path: Path, stub_embedder, vec_probe):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.embedding_available is True
    assert stats.embedded_chunks == stats.chunks_written
    assert stats.fully_embedded is True

    conn, vec_ok = db.open_index(db_path)
    try:
        assert vec_ok is True
        count = conn.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()["n"]
        assert count == stats.chunks_written
    finally:
        conn.close()


def test_reindex_raises_when_lock_already_held(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    with db.write_lock(db_path):
        with pytest.raises(db.IndexLocked):
            indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)


# ---------------------------------------------------------------------------
# Embedding backfill: recovery from a prior Ollama-down index, without --full
# ---------------------------------------------------------------------------


def test_reindex_backfills_embeddings_once_backend_recovers(
    tmp_path: Path, failing_embedder, stub_embedder, vec_probe
):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    # First run: backend down, index built lexical-only, 0 chunks embedded.
    first = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=failing_embedder)
    assert first.embedding_available is False
    assert first.vector_covered == 0

    conn = db.connect(db_path)
    try:
        assert db.get_meta(conn, db.META_EMBED_STATUS) != db.EMBED_STATUS_READY
    finally:
        conn.close()

    # Backend recovers. An ordinary (non --full) reindex over an otherwise
    # unchanged corpus must still backfill every chunk's missing vector.
    second = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert second.added == 0
    assert second.changed == 0
    assert second.embedding_available is True
    assert second.embedded_chunks == second.vector_total
    assert second.fully_embedded is True

    conn = db.connect(db_path)
    try:
        assert db.get_meta(conn, db.META_EMBED_STATUS) == db.EMBED_STATUS_READY
    finally:
        conn.close()

    # A third run must not re-embed anything: coverage is already complete.
    third = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert third.embedded_chunks == 0
    assert third.fully_embedded is True


def test_reindex_reports_not_ready_when_backfill_incomplete(tmp_path: Path, vec_probe):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    class DiesAfterProbeEmbedder:
        """Backend that answers the readiness probe but fails the real batch,
        so coverage stays incomplete even though the backend looked live."""

        model = "flaky-embed"

        def __init__(self):
            self.probed = False

        def probe(self):
            self.probed = True

        def embed(self, texts):
            from corpusdex.embedder import EmbeddingUnavailable

            raise EmbeddingUnavailable("stub: died mid-backfill")

    stats = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=DiesAfterProbeEmbedder()
    )
    assert stats.fully_embedded is False

    conn = db.connect(db_path)
    try:
        assert db.get_meta(conn, db.META_EMBED_STATUS) != db.EMBED_STATUS_READY
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Orphaned vector reconciliation: repairs a DB wedged by a prior partial run
# ---------------------------------------------------------------------------


def test_reindex_repairs_a_wedged_db_from_orphaned_vectors(
    tmp_path: Path, stub_embedder, vec_probe
):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    # Build a fully embedded index first.
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    conn, vec_ok = db.open_index(db_path)
    assert vec_ok is True
    try:
        # Simulate: a document's chunk was deleted by a run where the vector
        # extension was unavailable, so its vec_chunks row was never cleaned
        # up (indexer._delete_doc_vectors is a no-op when vec_ok is False).
        # Deleting the current max id means SQLite's next auto-assigned
        # INTEGER PRIMARY KEY value is that same freed id, reliably
        # reproducing the collision the reconciliation step must prevent.
        row = conn.execute("SELECT MAX(id) AS id FROM chunks").fetchone()
        orphan_chunk_id = row["id"]
        with conn:
            conn.execute("DELETE FROM chunks WHERE id = ?", (orphan_chunk_id,))
            # vec_chunks row for orphan_chunk_id deliberately left behind.
    finally:
        conn.close()

    # Now edit the file so a fresh insert happens; because SQLite reuses a
    # freed INTEGER PRIMARY KEY id when it was the max, the next inserted
    # chunk is likely to collide with the orphaned vec_chunks row without
    # the reconciliation step. Run reindex --full to force every document to
    # be rewritten (deleting and re-inserting every chunk row), which is
    # exactly the scenario that reproduces id reuse across the whole table.
    stats = indexer.reindex(
        db_path=db_path, workspace_root=workspace, full=True, embedder=stub_embedder
    )
    # Must not raise (a sqlite3.IntegrityError here means the DB is wedged).
    assert stats.errors == []

    conn, vec_ok = db.open_index(db_path)
    try:
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        vec_count = conn.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()["n"]
        # No leftover orphan: every vec row must reference a live chunk.
        orphans = conn.execute(
            "SELECT COUNT(*) AS n FROM vec_chunks WHERE chunk_id NOT IN (SELECT id FROM chunks)"
        ).fetchone()["n"]
        assert orphans == 0
        assert vec_count == chunk_count
    finally:
        conn.close()


def test_reconcile_orphaned_vectors_directly(tmp_path: Path, vec_probe):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    db_path = tmp_path / "index.db"
    conn, vec_ok = db.open_index(db_path, create=True)
    assert vec_ok is True
    try:
        import sqlite_vec

        with conn:
            conn.execute(
                "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
                "VALUES ('r', 'a.md', 'T', 'doc', 1.0, '1:h')"
            )
            conn.execute(
                "INSERT INTO chunks (id, ref, doc_id, heading_path, body) "
                "VALUES (1, 'corphan0000000000', 1, 'h', 'b')"
            )
            conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (1, sqlite_vec.serialize_float32([0.0] * db.EMBED_DIM)),
            )
            conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (99, sqlite_vec.serialize_float32([0.0] * db.EMBED_DIM)),  # orphan
            )
        with conn:
            indexer._reconcile_orphaned_vectors(conn)
        remaining = {
            r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM vec_chunks").fetchall()
        }
        assert remaining == {1}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Embed-model change invalidates and re-embeds
# ---------------------------------------------------------------------------


def test_a_model_of_a_different_width_rebuilds_the_vector_table(tmp_path: Path, vec_probe):
    """The point of #15: a 384-wide model must be a usable configuration.

    The width lives in the ``vec_chunks`` declaration, so a width change
    cannot be handled by deleting rows the way a model change can; the table
    itself has to be rebuilt.
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    class WidthEmbedder:
        def __init__(self, model, dim):
            self.model = model
            self.dim = dim

        def probe(self):
            pass

        def embed(self, texts):
            return [[0.1] * self.dim for _ in texts]

    first = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=WidthEmbedder("wide", 768)
    )
    assert first.embedded_chunks == first.vector_total
    conn = db.connect(db_path)
    try:
        assert db.vec_table_dim(conn) == 768
        assert db.get_meta(conn, db.META_EMBED_DIM) == "768"
    finally:
        conn.close()

    second = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=WidthEmbedder("narrow", 384)
    )
    assert any("dimension changed" in e for e in second.errors)
    # Every chunk re-embedded at the new width, not merely deleted.
    assert second.embedded_chunks == second.vector_total
    assert second.vector_total > 0
    conn = db.connect(db_path)
    try:
        assert db.vec_table_dim(conn) == 384
        assert db.get_meta(conn, db.META_EMBED_DIM) == "384"
    finally:
        conn.close()


def test_a_narrow_model_works_on_a_fresh_index(tmp_path: Path, vec_probe):
    """A fresh index is created at the default width, so the very first
    reindex with a narrow model must widen-then-rebuild rather than fail."""
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    class NarrowEmbedder:
        model = "narrow-only"
        dim = 384

        def probe(self):
            pass

        def embed(self, texts):
            return [[0.2] * 384 for _ in texts]

    stats = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=NarrowEmbedder()
    )
    assert stats.embedded_chunks == stats.vector_total
    assert stats.vector_total > 0
    conn = db.connect(db_path)
    try:
        assert db.vec_table_dim(conn) == 384
    finally:
        conn.close()


def test_a_dead_backend_does_not_rebuild_the_vector_table(tmp_path: Path, vec_probe):
    """``dim`` is None when the probe failed, and an unknown width must not be
    read as a width change -- that would discard every vector on an outage."""
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    class Ok:
        model = "m"
        dim = 768

        def probe(self):
            pass

        def embed(self, texts):
            return [[0.1] * 768 for _ in texts]

    from corpusdex.embedder import EmbeddingUnavailable

    class Dead:
        model = "m"
        dim = None

        def probe(self):
            raise EmbeddingUnavailable("backend down")

        def embed(self, texts):  # pragma: no cover - never called
            raise EmbeddingUnavailable("backend down")

    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=Ok())
    conn, _ = db.open_index(db_path)
    try:
        before = conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    finally:
        conn.close()
    assert before > 0

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=Dead())
    assert not any("dimension changed" in e for e in stats.errors)
    conn, _ = db.open_index(db_path)
    try:
        assert db.vec_table_dim(conn) == 768
        assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == before
    finally:
        conn.close()


def test_embed_model_change_invalidates_and_reembeds(tmp_path: Path, vec_probe):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    class NamedEmbedder:
        def __init__(self, model, dim=db.EMBED_DIM):
            self.model = model
            self.dim = dim

        def probe(self):
            pass

        def embed(self, texts):
            return [[0.1] * self.dim for _ in texts]

    first = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=NamedEmbedder("model-a")
    )
    assert first.embedded_chunks == first.vector_total

    conn = db.connect(db_path)
    try:
        assert db.get_meta(conn, db.META_EMBED_MODEL) == "model-a"
    finally:
        conn.close()

    # A different model must invalidate all previously-computed vectors and
    # re-embed everything, even though no document content changed. Since #55
    # that happens as a rebuild-and-swap rather than in place, which changes
    # what the counters mean: a fresh index takes every document as `added`
    # even though the corpus is identical, so `rebuilt` and `rebuild_reason`
    # are what carry the fact that nothing about the CORPUS changed.
    second = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=NamedEmbedder("model-b")
    )
    assert second.rebuilt is True
    assert "model-a" in second.rebuild_reason
    assert "model-b" in second.rebuild_reason
    assert second.added == second.docs_seen
    assert second.changed == 0
    assert second.embedded_chunks == second.vector_total

    conn = db.connect(db_path)
    try:
        assert db.get_meta(conn, db.META_EMBED_MODEL) == "model-b"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Registry row validation: reject names that could resolve outside the workspace
# ---------------------------------------------------------------------------


def test_read_registry_rejects_parent_traversal_repo_name(tmp_path: Path):
    _write(tmp_path / "repo-a" / "AGENTS.md", "# Repo A\n\n" + "text " * 20)
    (tmp_path / indexer.REGISTRY_FILENAME).write_text(
        "repo-a\thttps://example.invalid/repo-a\tmain\n"
        "../escape\thttps://example.invalid/escape\tmain\n",
        encoding="utf-8",
    )
    with pytest.raises(indexer.RegistryInvalid) as exc_info:
        indexer.discover_corpus(tmp_path)
    message = str(exc_info.value)
    assert "../escape" in message
    assert ".." in message


def test_read_registry_rejects_absolute_repo_name(tmp_path: Path):
    (tmp_path / indexer.REGISTRY_FILENAME).write_text(
        "/etc/passwd\thttps://example.invalid/x\tmain\n", encoding="utf-8"
    )
    with pytest.raises(indexer.RegistryInvalid) as exc_info:
        indexer.discover_corpus(tmp_path)
    message = str(exc_info.value)
    assert "/etc/passwd" in message


def test_read_registry_rejects_backslash_repo_name(tmp_path: Path):
    (tmp_path / indexer.REGISTRY_FILENAME).write_text(
        "foo\\bar\thttps://example.invalid/x\tmain\n", encoding="utf-8"
    )
    with pytest.raises(indexer.RegistryInvalid):
        indexer.discover_corpus(tmp_path)


def test_read_registry_accepts_ordinary_repo_names(tmp_path: Path):
    _write(tmp_path / "repo-a" / "AGENTS.md", "# Repo A\n\n" + "text " * 20)
    write_registry(tmp_path, ["repo-a"])
    docs = indexer.discover_corpus(tmp_path)  # must not raise
    assert any(d.repo == "repo-a" for d in docs)


# ---------------------------------------------------------------------------
# Embedding backfill: batches large id sets instead of one unbounded IN (...)
# ---------------------------------------------------------------------------


def test_backfill_missing_embeddings_batches_large_id_sets(tmp_path: Path, vec_probe):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    db_path = tmp_path / "index.db"
    conn, vec_ok = db.open_index(db_path, create=True)
    assert vec_ok is True
    try:
        chunk_total = indexer._BACKFILL_BATCH_SIZE * 2 + 200  # spans 3 batches
        with conn:
            conn.execute(
                "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
                "VALUES ('r', 'a.md', 'T', 'doc', 1.0, '1:h')"
            )
            for i in range(chunk_total):
                conn.execute(
                    "INSERT INTO chunks (ref, doc_id, heading_path, body) VALUES (?, 1, ?, ?)",
                    (db.chunk_ref("a.md", f"heading {i}", 0), f"heading {i}", f"body {i}"),
                )

        call_sizes: list[int] = []

        def embed_fn(texts):
            call_sizes.append(len(texts))
            return [[0.1] * db.EMBED_DIM for _ in texts]

        with conn:
            embedded, still_available = indexer._backfill_missing_embeddings(conn, embed_fn)

        assert embedded == chunk_total
        assert still_available is True
        assert sum(call_sizes) == chunk_total
        assert all(size <= indexer._BACKFILL_BATCH_SIZE for size in call_sizes)
        assert len(call_sizes) == 3  # 500 + 500 + 200

        vec_count = conn.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()["n"]
        assert vec_count == chunk_total
    finally:
        conn.close()


def test_reindex_rebuilds_a_stale_schema_version_index(tmp_path: Path, stub_embedder):
    # The index is a disposable projection, so a version bump must be
    # recoverable by an ordinary reindex rather than needing a manual delete.
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    conn, _vec_ok = db.open_index(db_path, create=True)
    with conn:
        db.set_meta(conn, db.META_SCHEMA_VERSION, str(db.SCHEMA_VERSION - 1))
    conn.close()

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert stats.added == stats.docs_seen
    conn, _vec_ok = db.open_index(db_path)
    try:
        assert db.get_meta(conn, db.META_SCHEMA_VERSION) == str(db.SCHEMA_VERSION)
    finally:
        conn.close()


def test_reindex_rebuilds_an_empty_index_file(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch()

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert stats.added == stats.docs_seen


def test_reindex_refuses_an_index_newer_than_this_build(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    conn, _vec_ok = db.open_index(db_path, create=True)
    with conn:
        db.set_meta(conn, db.META_SCHEMA_VERSION, str(db.SCHEMA_VERSION + 1))
    conn.close()

    with pytest.raises(db.SchemaVersionMismatch):
        indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)


def test_a_failed_rebuild_leaves_the_existing_index_intact(tmp_path: Path, stub_embedder):
    # The whole point of building into a scratch file: a rebuild that dies
    # partway must not replace a working index with an empty one.
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    first = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    before = db_path.read_bytes()

    class ExplodingEmbedder:
        model = "exploding"

        def probe(self) -> None:
            return None

        def embed(self, texts):
            raise RuntimeError("backend exploded mid-rebuild")

    with pytest.raises(RuntimeError):
        indexer.reindex(
            db_path=db_path,
            workspace_root=workspace,
            full=True,
            embedder=ExplodingEmbedder(),
        )

    assert db_path.read_bytes() == before
    assert not Path(f"{db_path}.rebuild").exists()
    conn, _vec_ok = db.open_index(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"] == (
            first.docs_seen
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# A changed link extractor invalidates the stored targets
#
# Link targets are extracted only while a document is chunked, and an
# incremental pass skips every document whose bytes have not moved. Without
# this, a smarter extractor shipped against an existing index changes nothing
# until somebody happens to run `--full`, and nothing anywhere says so.
# ---------------------------------------------------------------------------


def _citing_workspace(root: Path) -> None:
    filler = "notes about the watering schedule for the season. " * 6
    _write(
        root / "repo-a" / "docs" / "decisions" / "0006-default-setting.md",
        f"# Default setting\n\n## Body\n\n{filler}\n",
    )
    _write(
        root / "repo-a" / "docs" / "citing.md",
        f"# Citing\n\n## Body\n\n{filler} this extends decision 0006 in full.\n",
    )
    write_registry(root, ["repo-a"])


def test_reindex_stamps_the_link_extractor_version(tmp_path: Path, stub_embedder):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _citing_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert db.stored_meta(db_path, db.META_LINK_EXTRACTOR_VERSION) == str(links.EXTRACTOR_VERSION)


def test_a_stale_extractor_version_forces_a_rebuild_without_full(tmp_path: Path, stub_embedder):
    """The whole point: a plain `brain reindex` must pick the change up.

    The stored targets are emptied and the stamp aged, standing in for an
    index whose targets an older extractor wrote. Nothing on disk moves, so
    an ordinary incremental pass would skip every document and leave the
    graph exactly as the old extractor left it.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _citing_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    conn, _vec = db.open_index(db_path)
    with conn:
        conn.execute("DELETE FROM doc_link_targets")
        db.set_meta(conn, db.META_LINK_EXTRACTOR_VERSION, "0")
    conn.close()

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.link_edges == 1
    assert stats.unchanged == 0  # every document was re-read, not skipped


def test_an_index_with_no_extractor_stamp_rebuilds_once(tmp_path: Path, stub_embedder):
    """An index predating the stamp cannot be assumed to match.

    Its targets were written by an extractor whose version is unknown, so the
    only safe reading of a missing stamp is "not current".
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _citing_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    conn, _vec = db.open_index(db_path)
    with conn:
        conn.execute("DELETE FROM meta WHERE key = ?", (db.META_LINK_EXTRACTOR_VERSION,))
    conn.close()

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert stats.unchanged == 0


def test_a_matching_extractor_version_still_indexes_incrementally(tmp_path: Path, stub_embedder):
    """The guard must not turn every reindex into a full rebuild.

    Asserted because a version check that never matches is indistinguishable
    from a correct one on the test above, and would silently cost a full
    re-chunk and re-embed on every single run.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _citing_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.unchanged == 2
    assert stats.changed == 0
    assert stats.added == 0


def test_a_dead_backend_does_not_discard_vectors_on_an_apparent_model_change(
    tmp_path: Path, vec_probe
):
    """The width rebuild was gated on the probe having succeeded; the model
    DELETE next to it was not.

    An outage means the configured model was never reached, so "the stored
    model differs from the active one" is a statement about configuration, not
    about the vectors. Acting on it deletes usable vectors that the same run
    then cannot re-embed, and writes the new model name, so the next healthy
    run sees no change and never rebuilds.
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    from corpusdex.embedder import EmbeddingUnavailable

    class Ok:
        model = "m1"
        dim = 768

        def probe(self):
            pass

        def embed(self, texts):
            return [[0.1] * 768 for _ in texts]

    class DeadUnderANewName:
        model = "m2"
        dim = None

        def probe(self):
            raise EmbeddingUnavailable("backend down")

        def embed(self, texts):  # pragma: no cover - never reached
            raise EmbeddingUnavailable("backend down")

    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=Ok())
    conn, _ = db.open_index(db_path)
    try:
        before = conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    finally:
        conn.close()
    assert before > 0

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=DeadUnderANewName())

    # And it must not take the #55 rebuild path either. A rebuild ends in a
    # swap, and with the backend down the replacement would hold no vectors at
    # all, so swapping it in would destroy every usable vector: worse than the
    # in-place path this test was written to protect.
    assert stats.rebuilt is False

    conn, _ = db.open_index(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == before
        # And the unreachable model must not be recorded as the index's own,
        # or the next healthy run under m2 sees no change and skips the
        # invalidation this run deferred.
        assert db.get_meta(conn, db.META_EMBED_MODEL) != "m2"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Configurable corpus scope (#11)
# ---------------------------------------------------------------------------


def test_the_legacy_registry_filename_is_still_found(tmp_path: Path):
    """The workspace this engine grew up in has the old filename on disk. A
    rename that stops it indexing the moment it lands gets reverted, not
    finished, so the old name is searched after the new one."""
    _write(tmp_path / "repo-a" / "AGENTS.md", "# A\n\n" + "text " * 20)
    (tmp_path / ".harnx-repos.tsv").write_text(
        "repo-a\thttps://x.invalid\tmain\n", encoding="utf-8"
    )

    docs = indexer.discover_corpus(tmp_path)

    assert {d.rel_path for d in docs} == {"repo-a/AGENTS.md"}


def test_the_current_registry_filename_wins_over_the_legacy_one(tmp_path: Path, monkeypatch):
    _write(tmp_path / "repo-a" / "AGENTS.md", "# A\n\n" + "text " * 20)
    _write(tmp_path / "repo-b" / "AGENTS.md", "# B\n\n" + "text " * 20)
    write_registry(tmp_path, ["repo-a"])
    (tmp_path / ".harnx-repos.tsv").write_text(
        "repo-b\thttps://x.invalid\tmain\n", encoding="utf-8"
    )

    docs = indexer.discover_corpus(tmp_path)

    assert {d.rel_path for d in docs} == {"repo-a/AGENTS.md"}


def test_an_explicit_registry_file_replaces_the_search_rather_than_extending_it(
    tmp_path: Path, monkeypatch
):
    """A caller naming a registry is stating where scope comes from. Falling
    back to a differently-named file sitting next to it would let a registry
    they did not name decide the corpus."""
    _write(tmp_path / "repo-a" / "AGENTS.md", "# A\n\n" + "text " * 20)
    write_registry(tmp_path, ["repo-a"])
    monkeypatch.setenv("BRAIN_REGISTRY_FILE", ".absent-registry.tsv")

    with pytest.raises(indexer.RegistryMissing) as excinfo:
        indexer.discover_corpus(tmp_path)
    assert ".absent-registry.tsv" in str(excinfo.value)
    assert indexer.REGISTRY_FILENAME not in str(excinfo.value)


def test_the_knowledge_repo_defaults_to_the_repo_the_checkout_belongs_to(monkeypatch, tmp_path):
    """Not a constant: a constant naming this repo silently stops matching the
    day the package is renamed, and every dev checkout then resolves as
    'installed'. Installed from a wheel there is no checkout, so no repo gets
    whole-tree treatment unless one is named.

    Belongs to, not sits in. The two are the same directory for an ordinary
    checkout and differ for a linked worktree, which is the case that broke.
    """
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)

    checkout = tmp_path / "some-checkout"
    checkout.mkdir()
    monkeypatch.setattr(indexer.db, "source_checkout_root", lambda: checkout)
    assert indexer.knowledge_repo() == "some-checkout"

    monkeypatch.setattr(indexer.db, "source_checkout_root", lambda: None)
    assert indexer.knowledge_repo() is None

    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "named-explicitly")
    assert indexer.knowledge_repo() == "named-explicitly"


def _linked_worktree_of(canonical: Path, slug: str) -> Path:
    """Lay out a linked worktree of ``canonical`` the way git does on disk.

    ``tests/test_db.py`` proves this layout against real ``git worktree add``;
    it is reproduced here so these tests can place the worktree wherever the
    case needs it without building a repository each time.
    """
    meta = canonical / ".git" / "worktrees" / slug
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "commondir").write_text("../..\n", encoding="utf-8")
    linked = canonical.parent / ".worktrees" / canonical.name / slug
    linked.mkdir(parents=True, exist_ok=True)
    (linked / ".git").write_text(f"gitdir: {meta}\n", encoding="utf-8")
    return linked


def test_running_from_a_worktree_still_names_the_repo_it_belongs_to(monkeypatch, tmp_path):
    """The regression. A worktree directory is named for the branch, so taking
    its name gave a value matching no registered repo, and the knowledge repo
    silently became "none of them" -- with a zero exit and a success line."""
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)
    canonical = tmp_path / "the-brain"
    canonical.mkdir()
    linked = _linked_worktree_of(canonical, "issue-86-knowledge-repo")

    monkeypatch.setattr(indexer.db, "source_checkout_root", lambda: linked)
    assert indexer.knowledge_repo() == "the-brain"


def test_a_worktree_run_still_indexes_the_knowledge_repo_in_full(monkeypatch, tmp_path):
    """End to end, because naming the repo correctly is only the mechanism.

    What the defect cost was documents. The load-bearing assertion is the one
    about ``context/``: it is reached ONLY by whole-tree treatment, so it fails
    the moment the repo is named wrongly. ``decisions/`` is asserted alongside
    it as a control in the other direction -- it lives in ``PROSE_DIR_NAMES``,
    so ordinary treatment finds it either way, and a test built on it would
    have passed against the very defect it claims to cover.
    """
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)
    canonical = tmp_path / "the-brain"
    _write(canonical / "context" / "repo-a.md", "# repo-a\n\n" + "w " * 20)
    _write(canonical / "gaps" / "2026-09-07-audit.md", "# Audit\n\n" + "v " * 20)
    _write(canonical / "decisions" / "0001-x.md", "# 0001\n\n" + "x " * 20)
    _write(canonical / "docs" / "guide.md", "# Guide\n\n" + "y " * 20)
    _write(tmp_path / "repo-a" / "context" / "not-corpus.md", "# Nope\n\n" + "z " * 20)
    write_registry(tmp_path, ["the-brain", "repo-a"])
    linked = _linked_worktree_of(canonical, "issue-86-knowledge-repo")

    monkeypatch.setattr(indexer.db, "source_checkout_root", lambda: linked)
    rel_paths = {d.rel_path for d in indexer.discover_corpus(tmp_path)}
    # Whole-tree only. These are what the defect actually dropped.
    assert "the-brain/context/repo-a.md" in rel_paths
    assert "the-brain/gaps/2026-09-07-audit.md" in rel_paths
    # Prose dirs, found under either treatment.
    assert "the-brain/decisions/0001-x.md" in rel_paths
    assert "the-brain/docs/guide.md" in rel_paths
    # Unchanged for every other repo: whole-tree treatment is not contagious.
    assert "repo-a/context/not-corpus.md" not in rel_paths


def test_a_run_resolves_the_knowledge_repo_exactly_once(tmp_path, monkeypatch):
    """One resolution feeds discovery, classification, reporting and the warning.

    It reads the environment AND the filesystem, so two resolutions inside one
    run agree only by luck. Resolving separately let a run choose its corpus
    with one answer and describe it with another, which is the same defect
    class as #86 with a shorter window.
    """
    _write(tmp_path / "the-brain" / "context" / "c.md", "# C\n\n" + "x " * 20)
    write_registry(tmp_path, ["the-brain"])
    calls = []
    real = indexer.knowledge_repo

    def counting():
        calls.append(1)
        return real()

    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "the-brain")
    monkeypatch.setattr(indexer, "knowledge_repo", counting)
    stats = indexer.reindex(
        db_path=tmp_path / "index.db",
        workspace_root=tmp_path,
        embedder=None,
    )
    assert stats.knowledge_repo == "the-brain"
    assert len(calls) == 1, f"resolved {len(calls)} times"


def test_a_symlinked_corpus_root_is_in_scope_when_discovery_treats_it(monkeypatch, tmp_path):
    """Scope must agree with discovery about what a root is, not about its label.

    ``BRAIN_CORPUS_ROOTS`` entries are resolved before discovery compares
    basenames, so a symlink named ``alias`` pointing at ``the-brain`` IS the
    knowledge repo there. A scope check reading the unresolved name would call
    the same run out of scope and report that nothing got whole-tree treatment
    while ``context/`` was being indexed.
    """
    canonical = tmp_path / "the-brain"
    _write(canonical / "context" / "c.md", "# C\n\n" + "x " * 20)
    alias = tmp_path / "alias"
    alias.symlink_to(canonical, target_is_directory=True)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(alias))
    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "the-brain")

    rel_paths = {d.rel_path for d in indexer.discover_corpus(None)}
    assert "the-brain/context/c.md" in rel_paths
    assert indexer.knowledge_repo_in_scope("the-brain", None) is True


def test_a_corpus_root_that_does_not_exist_is_not_in_scope(monkeypatch, tmp_path):
    """The other direction of the same disagreement.

    Discovery skips a root that is not a directory, so naming one gives no
    repo whole-tree treatment. Matching on the basename alone would report it
    in scope and suppress the warning that is the operator's only signal.
    """
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(tmp_path / "the-brain"))
    assert indexer.knowledge_repo_in_scope("the-brain", None) is False


def test_a_registered_repo_with_no_directory_is_not_in_scope(monkeypatch, tmp_path):
    """A registry row is a claim; discovery requires the directory to be there."""
    write_registry(tmp_path, ["the-brain", "repo-a"])
    _write(tmp_path / "repo-a" / "docs" / "d.md", "# D\n\n" + "x " * 20)
    assert indexer.knowledge_repo_in_scope("repo-a", tmp_path) is True
    assert indexer.knowledge_repo_in_scope("the-brain", tmp_path) is False


def test_the_knowledge_repo_is_reported_only_when_it_names_a_repo_in_scope(
    tmp_path: Path, monkeypatch
):
    """Reporting the resolved NAME would assert whole-tree treatment that did
    not happen: running the engine against a workspace that is not its own is
    ordinary, and there the name legitimately matches nothing."""
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)
    monkeypatch.delenv("BRAIN_CORPUS_ROOTS", raising=False)
    write_registry(tmp_path, ["the-brain", "repo-a"])
    # Both halves of what discovery requires: the row AND the directory. See
    # test_a_registered_repo_with_no_directory_is_not_in_scope for the row
    # without the directory, which discovery skips.
    (tmp_path / "the-brain").mkdir()
    (tmp_path / "repo-a").mkdir()

    assert indexer.knowledge_repo_in_scope("the-brain", tmp_path) is True
    assert indexer.knowledge_repo_in_scope("somewhere-else", tmp_path) is False
    assert indexer.knowledge_repo_in_scope(None, tmp_path) is False
    # No scope to match against is not a match.
    assert indexer.knowledge_repo_in_scope("the-brain", None) is False


def test_scope_for_configured_roots_comes_from_the_roots_not_the_registry(
    tmp_path: Path, monkeypatch
):
    """Explicit roots ARE the corpus and the registry is not consulted, so a
    knowledge repo has to be matched against the same source scope came from.
    Checking the registry here would report a repo that was never indexed."""
    root = tmp_path / "elsewhere" / "the-brain"
    root.mkdir(parents=True)
    write_registry(tmp_path, ["something-unrelated"])
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(root))

    assert indexer.knowledge_repo_in_scope("the-brain", tmp_path) is True
    assert indexer.knowledge_repo_in_scope("something-unrelated", tmp_path) is False


def test_an_exclusion_without_a_slash_matches_any_path_segment(tmp_path: Path, monkeypatch):
    _write(tmp_path / "repo-a" / "docs" / "keep.md", "# Keep\n\n" + "text " * 20)
    _write(tmp_path / "repo-a" / "docs" / "vendor" / "drop.md", "# Drop\n\n" + "text " * 20)
    write_registry(tmp_path, ["repo-a"])
    monkeypatch.setenv("BRAIN_EXCLUDE", "vendor")

    rel_paths = {d.rel_path for d in indexer.discover_corpus(tmp_path)}

    assert "repo-a/docs/keep.md" in rel_paths
    assert "repo-a/docs/vendor/drop.md" not in rel_paths


def test_an_exclusion_with_a_slash_matches_the_whole_relative_path(tmp_path: Path, monkeypatch):
    _write(tmp_path / "repo-a" / "docs" / "keep.md", "# Keep\n\n" + "text " * 20)
    _write(tmp_path / "repo-b" / "docs" / "drop.md", "# Drop\n\n" + "text " * 20)
    write_registry(tmp_path, ["repo-a", "repo-b"])
    monkeypatch.setenv("BRAIN_EXCLUDE", "repo-b/docs/*")

    rel_paths = {d.rel_path for d in indexer.discover_corpus(tmp_path)}

    assert "repo-a/docs/keep.md" in rel_paths
    assert "repo-b/docs/drop.md" not in rel_paths


def test_explicit_corpus_roots_replace_registry_mode(tmp_path: Path, monkeypatch):
    """The answer to 'I am not in that workspace'. Scope stays one closed
    question with one answer: with roots configured, a registry sitting in the
    workspace is not consulted at all."""
    outside = tmp_path / "elsewhere" / "project-x"
    _write(outside / "README.md", "# X\n\n" + "text " * 20)
    _write(outside / "docs" / "deep" / "note.md", "# Note\n\n" + "text " * 20)
    workspace = tmp_path / "workspace"
    _write(workspace / "repo-a" / "AGENTS.md", "# A\n\n" + "text " * 20)
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(outside))

    docs = indexer.discover_corpus(workspace)

    assert {d.rel_path for d in docs} == {"project-x/README.md", "project-x/docs/deep/note.md"}
    assert {d.repo for d in docs} == {"project-x"}


def test_explicit_corpus_roots_need_no_workspace_root_at_all(tmp_path: Path, monkeypatch):
    """The installation this setting exists to serve has no workspace root, so
    resolving one anyway would raise NotConfigured before scope is even read."""
    root = tmp_path / "project-x"
    _write(root / "README.md", "# X\n\n" + "text " * 20)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(root))
    # isolate_settings already cleared it; nothing here re-sets it.

    docs = indexer.discover_corpus()

    assert {d.rel_path for d in docs} == {"project-x/README.md"}


def test_several_corpus_roots_sharing_no_parent_each_keep_their_own_prefix(
    tmp_path: Path, monkeypatch
):
    """Deriving one prefix from a common parent would produce
    ``../../elsewhere/x.md`` for every root but one."""
    first = tmp_path / "a" / "one"
    second = tmp_path / "b" / "two"
    _write(first / "README.md", "# One\n\n" + "text " * 20)
    _write(second / "README.md", "# Two\n\n" + "text " * 20)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", os.pathsep.join([str(first), str(second)]))

    rel_paths = {d.rel_path for d in indexer.discover_corpus()}

    assert rel_paths == {"one/README.md", "two/README.md"}


def test_two_roots_with_the_same_basename_are_refused(tmp_path: Path, monkeypatch):
    """Issue #71. The test above says "sharing no parent", which reads as if
    the colliding case had been considered; it had not.

    A document is named for its root's BASENAME, so two roots ending in the
    same directory name produce two files carrying one ``rel_path``, which
    ``path TEXT NOT NULL UNIQUE`` cannot hold.
    """
    first = tmp_path / "w1" / "mydocs"
    second = tmp_path / "w2" / "mydocs"
    _write(first / "README.md", "# One\n\n" + "alpha " * 20)
    _write(second / "README.md", "# Two\n\n" + "beta " * 20)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", os.pathsep.join([str(first), str(second)]))

    with pytest.raises(indexer.CorpusRootsCollide) as excinfo:
        indexer.discover_corpus()

    # Both roots by name, because "two roots collide" is not actionable
    # without knowing which two: they share the only part of the path the
    # message would otherwise print.
    message = str(excinfo.value)
    assert str(first) in message
    assert str(second) in message
    assert "mydocs" in message


def test_the_same_root_named_twice_is_not_a_collision(tmp_path: Path, monkeypatch):
    """The control that keeps the check from being "two roots is an error".

    One root listed twice, or written with a trailing slash, is one directory
    named twice and already dedupes on the resolved path. Keying the check on
    the NAME alone would reject a configuration that works today.
    """
    root = tmp_path / "w1" / "mydocs"
    _write(root / "README.md", "# One\n\n" + "alpha " * 20)

    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", os.pathsep.join([str(root), str(root)]))
    assert {d.rel_path for d in indexer.discover_corpus()} == {"mydocs/README.md"}

    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", os.pathsep.join([str(root), f"{root}/"]))
    assert {d.rel_path for d in indexer.discover_corpus()} == {"mydocs/README.md"}


def test_a_colliding_root_added_later_leaves_the_existing_index_untouched(
    tmp_path: Path, monkeypatch, stub_embedder
):
    """The silent branch, which is the one worth refusing.

    With an index already built, the colliding document did not raise:
    ``existing`` is a snapshot read before the loop, so the second document
    compared against the stale row the first had already deleted, matched its
    fingerprint and was counted ``unchanged``. The citation then served the
    other file's content and flipped on every run (measured: alpha, beta,
    alpha, beta) with ``stats.errors`` empty.

    Refusing at discovery is what makes this safe: nothing is written, so the
    index still answers with what it had, and the next run with a corrected
    configuration is an ordinary no-op rather than a repair.
    """
    first = tmp_path / "w1" / "mydocs"
    second = tmp_path / "w2" / "mydocs"
    db_path = tmp_path / "var" / "index.db"
    _write(first / "README.md", "# One\n\n" + "unique-alpha-one " * 20)
    _write(second / "README.md", "# Two\n\n" + "unique-beta-two " * 20)

    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(first))
    indexer.reindex(db_path=db_path, embedder=stub_embedder)

    def stored_bodies() -> list[str]:
        conn = db.connect(db_path)
        try:
            return [r["body"] for r in conn.execute("SELECT body FROM chunks ORDER BY id")]
        finally:
            conn.close()

    before = stored_bodies()
    assert any("unique-alpha-one" in b for b in before)

    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", os.pathsep.join([str(first), str(second)]))
    with pytest.raises(indexer.CorpusRootsCollide):
        indexer.reindex(db_path=db_path, embedder=stub_embedder)

    # Byte-identical, not merely "still alpha": a refusal that rewrote rows
    # and happened to keep the same content would pass a weaker check.
    assert stored_bodies() == before

    # And the fix is an ordinary run, not a rebuild.
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(first))
    recovered = indexer.reindex(db_path=db_path, embedder=stub_embedder)
    assert recovered.unchanged == 1
    assert recovered.added == 0
    assert stored_bodies() == before


def test_roots_that_do_not_exist_cannot_collide(tmp_path: Path, monkeypatch):
    """A root that is not a directory contributes no documents, so it cannot
    claim a citation path either. Checking it anyway would reject a
    configuration that lists a not-yet-created root harmlessly today."""
    real = tmp_path / "w1" / "mydocs"
    _write(real / "README.md", "# One\n\n" + "alpha " * 20)
    missing = tmp_path / "w2" / "mydocs"  # never created

    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", os.pathsep.join([str(real), str(missing)]))

    assert {d.rel_path for d in indexer.discover_corpus()} == {"mydocs/README.md"}


def test_a_configured_root_that_is_the_knowledge_repo_is_walked_in_full(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "the-brain"
    _write(root / "glossary.md", "# Glossary\n\n" + "text " * 20)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(root))
    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "the-brain")

    assert {d.rel_path for d in indexer.discover_corpus()} == {"the-brain/glossary.md"}

    # And an ordinary root contributes only the documented subset, so the same
    # file is not corpus when the repo is not the knowledge repo.
    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "something-else")
    assert indexer.discover_corpus() == []


def test_exclusions_apply_to_explicit_roots_too(tmp_path: Path, monkeypatch):
    root = tmp_path / "project-x"
    _write(root / "docs" / "keep.md", "# Keep\n\n" + "text " * 20)
    _write(root / "docs" / "vendor" / "drop.md", "# Drop\n\n" + "text " * 20)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(root))
    monkeypatch.setenv("BRAIN_EXCLUDE", "vendor")

    assert {d.rel_path for d in indexer.discover_corpus()} == {"project-x/docs/keep.md"}


def test_reindex_runs_against_explicit_roots_with_no_workspace_configured(
    tmp_path: Path, monkeypatch, stub_embedder
):
    root = tmp_path / "project-x"
    _write(root / "README.md", "# X\n\n## Section\n\n" + "content here " * 20)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(root))

    # Deleting the env vars is not enough to reproduce the installed state:
    # workspace_root() falls back to the source checkout's parent, which
    # exists on any dev machine, so the guard under test would never be
    # reached and the test would pass with the guard removed. Make the call
    # raise the way it does for a user who has no checkout at all.
    def no_workspace():
        raise db.NotConfigured("no workspace root")

    monkeypatch.setattr(indexer.db, "workspace_root", no_workspace)

    stats = indexer.reindex(db_path=tmp_path / "var" / "index.db", embedder=stub_embedder)

    assert stats.docs_seen == 1
    assert stats.added == 1


def test_a_model_change_leaves_the_live_index_intact_until_the_swap(
    tmp_path: Path, vec_probe, monkeypatch
):
    """#55: the window a concurrent reader could fall into is gone, not described.

    Updating in place emptied the vector table and refilled it over a whole
    embedding run, and readers take no lock, so a search inside that window
    got a lexical-only page from an index that could have answered fully.

    Asserted from inside ``swap_index``, which is the last instant the old
    index is still the live one. Whatever a reader would have seen at the
    worst possible moment is what this observes: if the destructive work had
    happened in place, the count here would be 0.
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)

    class NamedEmbedder:
        def __init__(self, model):
            self.model = model
            self.dim = db.EMBED_DIM

        def probe(self):
            pass

        def embed(self, texts):
            return [[0.1] * self.dim for _ in texts]

    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=NamedEmbedder("model-a"))
    conn, _ = db.open_index(db_path)
    try:
        before = conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    finally:
        conn.close()
    assert before > 0

    observed: dict[str, object] = {}
    real_swap = db.swap_index

    def watching_swap(work_path, target_path):
        reader_conn, _vec_ok = db.open_index(target_path)
        try:
            observed["vectors"] = reader_conn.execute(
                "SELECT count(*) FROM vec_chunks"
            ).fetchone()[0]
            observed["model"] = db.get_meta(reader_conn, db.META_EMBED_MODEL)
        finally:
            reader_conn.close()
        return real_swap(work_path, target_path)

    monkeypatch.setattr(db, "swap_index", watching_swap)
    stats = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=NamedEmbedder("model-b")
    )

    assert stats.rebuilt is True
    # The live index still held every old vector, under the old model name,
    # right up to the swap.
    assert observed["vectors"] == before
    assert observed["model"] == "model-a"
    # And afterwards it is the new one, complete.
    conn, _ = db.open_index(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == before
        assert db.get_meta(conn, db.META_EMBED_MODEL) == "model-b"
    finally:
        conn.close()


def test_discover_corpus_matches_decisions_and_specs_directories(tmp_path: Path):
    """A registered repo may keep its durable prose outside ``docs/``.

    ``docs`` alone was the whole proxy for "written to be read later", which
    held until a registered repo turned out not to use it: one keeps its
    ADRs in ``decisions/`` and its RFCs in ``specs/``, including the rules
    governing how the rest of the pack may be edited. Its ``docs/`` WAS
    indexed, so a query returned plausible results and gave no signal that the
    decision record behind them had never been searched.
    """
    _write(tmp_path / "repo-a" / "decisions" / "0001-pick-a-runtime.md", "# ADR\n\n" + "t " * 20)
    _write(tmp_path / "repo-a" / "specs" / "0000-rfc-process.md", "# RFC\n\n" + "t " * 20)
    # Nested one level down, the same way docs/ is matched at any depth.
    _write(tmp_path / "repo-a" / "engine" / "specs" / "0002-wire.md", "# Wire\n\n" + "t " * 20)
    write_registry(tmp_path, ["repo-a"])
    rel_paths = {d.rel_path for d in indexer.discover_corpus(tmp_path)}
    assert "repo-a/decisions/0001-pick-a-runtime.md" in rel_paths
    assert "repo-a/specs/0000-rfc-process.md" in rel_paths
    assert "repo-a/engine/specs/0002-wire.md" in rel_paths


def test_a_prose_directory_inside_a_skipped_tree_is_still_skipped(tmp_path: Path):
    """Widening the accepted names must not widen what the skip list guards.

    An in-flight agent worktree nested inside a repo carries a full duplicate
    of that repo, `decisions/` included, so this is the shape in which the
    widening would multiply every ADR by the number of live worktrees.
    """
    _write(tmp_path / "repo-a" / "AGENTS.md", "# A\n\n" + "t " * 20)
    _write(
        tmp_path / "repo-a" / ".claude" / "worktrees" / "agent-1" / "decisions" / "0001-x.md",
        "# duplicate\n\nshould not be indexed\n",
    )
    _write(
        tmp_path / "repo-a" / "node_modules" / "pkg" / "specs" / "0000-vendor.md",
        "# vendored\n\nshould not be indexed\n",
    )
    write_registry(tmp_path, ["repo-a"])
    rel_paths = {d.rel_path for d in indexer.discover_corpus(tmp_path)}
    assert not any("worktrees" in p for p in rel_paths)
    assert not any("node_modules" in p for p in rel_paths)


def test_governance_is_root_prose_but_the_boilerplate_pair_is_not(tmp_path: Path):
    """Pins a deliberate EXCLUSION, which is the half that rots quietly.

    ``SECURITY.md`` and ``CONTRIBUTING.md`` look like obvious omissions next to
    ``GOVERNANCE.md``, and the reason they are absent is measurement rather
    than taste: 18 of the 19 registered repos carrying ``SECURITY.md`` hold a
    byte-identical 52-line copy, and ``CONTRIBUTING.md`` is the same template
    with the repo name substituted. The largest retrieval defect ever measured
    against this index was near-duplicate crowding, so admitting eighteen
    copies of one document would cost ranking and buy nothing. Without this
    test the next reader adds them as an oversight.
    """
    for name in ("GOVERNANCE.md", "SECURITY.md", "CONTRIBUTING.md", "NOTES.md"):
        _write(tmp_path / "repo-a" / name, f"# {name}\n\n" + "text " * 20)
    write_registry(tmp_path, ["repo-a"])
    rel_paths = {d.rel_path for d in indexer.discover_corpus(tmp_path)}
    assert "repo-a/GOVERNANCE.md" in rel_paths
    assert "repo-a/SECURITY.md" not in rel_paths
    assert "repo-a/CONTRIBUTING.md" not in rel_paths
    assert "repo-a/NOTES.md" not in rel_paths


def test_the_workspace_roots_own_prose_walk_still_does_not_cross_into_siblings(
    tmp_path: Path,
):
    """The shallow branch grew from one directory to three; it must stay shallow.

    The workspace root's walk is deliberately NOT the full-tree one: rooted
    there, a full walk crosses into every sibling directory including
    unregistered stale checkouts, reintroducing the whole-workspace scan the
    registry exists to prevent. Iterating three names instead of one is exactly
    the kind of edit that invites collapsing them into a single walk.
    """
    _write(tmp_path / "decisions" / "0001-workspace-level.md", "# WS ADR\n\n" + "t " * 20)
    _write(tmp_path / "specs" / "0000-workspace-rfc.md", "# WS RFC\n\n" + "t " * 20)
    _write(
        tmp_path / "unregistered-checkout" / "decisions" / "0001-leaked.md",
        "# leaked\n\nshould not be indexed\n",
    )
    _write(
        tmp_path / "unregistered-checkout" / "deep" / "specs" / "0000-leaked.md",
        "# leaked\n\nshould not be indexed\n",
    )
    write_registry(tmp_path, [])
    rel_paths = {d.rel_path for d in indexer.discover_corpus(tmp_path)}
    assert "decisions/0001-workspace-level.md" in rel_paths
    assert "specs/0000-workspace-rfc.md" in rel_paths
    assert not any("unregistered-checkout" in p for p in rel_paths)


# ---------------------------------------------------------------------------
# Symlinked and oversized documents (#25)
# ---------------------------------------------------------------------------


def test_a_symlinked_document_is_not_read(tmp_path: Path, stub_embedder):
    """The walk already declines symlinked DIRECTORIES; a symlinked file was
    the remaining way a path outside the corpus roots could be read.

    The link here points outside the workspace entirely, which is the case
    that matters: today every corpus root is a repo we control, so the damage
    is bounded by discovery rather than by any check, and that bound goes away
    the moment the corpus can be aimed at an arbitrary directory.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    outside = tmp_path / "outside" / "private.md"
    _write(outside, "# Private\n\n" + "secret " * 40)
    (workspace / "repo-a" / "docs" / "linked.md").symlink_to(outside)

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.rejected == 1
    assert any("linked.md" in e and "symlink" in e for e in stats.errors)
    conn = db.connect(db_path)
    try:
        # The document is absent, and so is its content: a row keyed on the
        # link path with the target's body would be the same disclosure.
        assert conn.execute(
            "SELECT 1 FROM documents WHERE path = 'repo-a/docs/linked.md'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM chunks WHERE body LIKE '%secret%'"
        ).fetchone() is None
    finally:
        conn.close()


def test_a_document_larger_than_the_cap_is_not_read(tmp_path: Path, stub_embedder, monkeypatch):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    monkeypatch.setattr(indexer, "MAX_DOCUMENT_BYTES", 2048)
    _write(workspace / "repo-a" / "docs" / "huge.md", "# Huge\n\n" + "x " * 4000)

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.rejected == 1
    assert any("huge.md" in e and "exceeds" in e for e in stats.errors)
    conn = db.connect(db_path)
    try:
        assert conn.execute(
            "SELECT 1 FROM documents WHERE path = 'repo-a/docs/huge.md'"
        ).fetchone() is None
    finally:
        conn.close()


def test_a_document_at_the_cap_is_still_read(tmp_path: Path, stub_embedder, monkeypatch):
    """The boundary, in the direction that costs a document if it is wrong.

    Without this the predicate could use ``>=`` and every test above still
    passes, because they all sit far from the edge.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    body = "# Exact\n\n" + "y " * 200
    path = workspace / "repo-a" / "docs" / "exact.md"
    _write(path, body)
    monkeypatch.setattr(indexer, "MAX_DOCUMENT_BYTES", path.stat().st_size)

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.rejected == 0
    conn = db.connect(db_path)
    try:
        assert conn.execute(
            "SELECT 1 FROM documents WHERE path = 'repo-a/docs/exact.md'"
        ).fetchone() is not None
    finally:
        conn.close()


def test_a_document_that_becomes_rejected_loses_its_stored_row(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """A rejection has to reach an index built before the rule existed.

    This is the case a bare ``continue`` gets wrong and no other test here
    catches: ``seen_paths`` is built from the whole corpus BEFORE the read
    loop, so skipping the read leaves the path looking still-offered and the
    removal sweep never touches it. The document would keep answering queries
    from content nothing reads any more, which is worse than never having
    indexed it, because the staleness is invisible.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    doc = workspace / "repo-a" / "docs" / "grows.md"
    _write(doc, "# Grows\n\n" + "small " * 20)

    first = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert first.rejected == 0
    conn = db.connect(db_path)
    try:
        assert conn.execute(
            "SELECT 1 FROM documents WHERE path = 'repo-a/docs/grows.md'"
        ).fetchone() is not None
    finally:
        conn.close()

    monkeypatch.setattr(indexer, "MAX_DOCUMENT_BYTES", 8)
    second = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert second.rejected >= 1
    conn = db.connect(db_path)
    try:
        assert conn.execute(
            "SELECT 1 FROM documents WHERE path = 'repo-a/docs/grows.md'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM chunks WHERE body LIKE '%small%'"
        ).fetchone() is None
    finally:
        conn.close()


def test_an_unreadable_document_keeps_its_row_unlike_a_rejected_one(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """The control arm for the test above, and the reason the two paths differ.

    A read failure is a fact about this moment: a file mid-write, a permission
    that comes back. Deleting the stored row on a blip would lose good content
    that nothing is wrong with. A rejection is a fact about the document. If
    both branches discarded the path this test fails; if neither did, the test
    above fails. Neither passes alone.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    doc = workspace / "repo-a" / "docs" / "blips.md"
    _write(doc, "# Blips\n\n" + "content " * 20)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    real_lstat = os.lstat

    def failing_lstat(path, *a, **kw):
        if str(path).endswith("blips.md"):
            raise PermissionError(13, "Permission denied")
        return real_lstat(path, *a, **kw)

    monkeypatch.setattr(indexer.os, "lstat", failing_lstat)
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.rejected == 0
    assert stats.removed == 0
    assert any("blips.md" in e for e in stats.errors)
    conn = db.connect(db_path)
    try:
        assert conn.execute(
            "SELECT 1 FROM documents WHERE path = 'repo-a/docs/blips.md'"
        ).fetchone() is not None
    finally:
        conn.close()


def test_the_shipped_cap_is_bounded_on_both_sides():
    """Pins the CONSTANT, which every test above monkeypatches away.

    Written because a mutant that reverted ``MAX_DOCUMENT_BYTES`` to an
    absurd value survived the whole suite: each size test sets its own cap, so
    all of them keep passing while the shipped build has no effective bound.
    The mechanism was pinned and the value was not.

    Both sides are load-bearing and they fail in opposite directions. Too
    high and the cap is decorative, which is the mutant. Too low and it starts
    refusing real documents, which is the expensive failure: the corpus goes
    quiet about something it holds. The floor is set from measurement rather
    than taste, against the largest document in the workspace this indexes, a
    review ledger at 890 KB growing about 43 KB a day; a 1 MiB cap would have
    begun refusing it within a week of being written.
    """
    assert indexer.MAX_DOCUMENT_BYTES >= 4 * 1024 * 1024
    assert indexer.MAX_DOCUMENT_BYTES <= 64 * 1024 * 1024
    # Well clear of a single chunk, or the cap would be refusing documents the
    # chunker is built to split rather than bounding a pathological read.
    assert indexer.MAX_DOCUMENT_BYTES > 100 * chunker.MAX_CHUNK_CHARS


# ---------------------------------------------------------------------------
# A vec table the process cannot reach (#72)
# ---------------------------------------------------------------------------


def test_an_unreachable_vec_table_refuses_the_run_before_mutating(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """The corrupting state is a vec table that EXISTS while the module did not
    load, and the refusal has to land before any write.

    Without the module SQLite cannot instantiate ``vec0``, so DELETE and DROP
    against ``vec_chunks`` both fail while ``DELETE FROM chunks`` succeeds.
    A pass here removes a chunk and cannot remove its vector; SQLite reissues
    the freed id, and the next chunk to take it inherits a vector describing
    text that no longer exists. Nothing downstream can see it: the vector is
    not an orphan so the reconcile sweep misses it, the backfill's LEFT JOIN
    finds a vector present, and status reports the index fully embedded.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    conn = db.connect(db_path)
    try:
        before = conn.execute("SELECT id, body FROM chunks ORDER BY id").fetchall()
        before_rows = [(r["id"], r["body"]) for r in before]
    finally:
        conn.close()

    (workspace / "repo-a" / "docs" / "guide.md").write_text(
        "# Guide\n\n## Section\nreplaced " * 30, encoding="utf-8"
    )
    monkeypatch.setattr(db, "load_vec", lambda conn: False)

    with pytest.raises(db.VectorTableUnreachable) as exc:
        indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert "sqlite-vec" in str(exc.value)
    assert "--full" in str(exc.value)

    # Nothing was written. A refusal that fires after the first delete would
    # leave exactly the damage it exists to prevent, and every assertion above
    # would still pass.
    conn = db.connect(db_path)
    try:
        after = conn.execute("SELECT id, body FROM chunks ORDER BY id").fetchall()
        assert [(r["id"], r["body"]) for r in after] == before_rows
    finally:
        conn.close()


def test_an_index_with_no_vec_table_still_reindexes_without_the_extension(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """The control arm: this is what stops the guard being over-broad.

    An index built without sqlite-vec has no ``vec_chunks`` at all, so there
    is no second half to keep in step and an incremental run is correct. That
    is the lexical-only mode this engine supports on purpose. A guard written
    as ``not vec_ok`` alone passes the test above and fails this one.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    monkeypatch.setattr(db, "load_vec", lambda conn: False)

    first = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert first.added > 0

    conn = db.connect(db_path)
    try:
        assert db.has_vec_table(conn) is False
    finally:
        conn.close()

    (workspace / "repo-a" / "docs" / "guide.md").write_text(
        "# Guide\n\n## Section\nchanged " * 30, encoding="utf-8"
    )
    second = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    assert second.changed == 1


def test_a_full_rebuild_is_the_way_out_of_an_unreachable_vec_table(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """The remedy the error message names has to actually work.

    An error that points at a command which also fails is worse than no
    message. A full rebuild goes to a scratch file, and a scratch file created
    while the extension is unavailable gets no vec table, so the rebuild both
    succeeds and produces an index this state cannot corrupt.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    monkeypatch.setattr(db, "load_vec", lambda conn: False)

    stats = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=stub_embedder, full=True
    )

    assert stats.added > 0
    conn = db.connect(db_path)
    try:
        assert db.has_vec_table(conn) is False
    finally:
        conn.close()


def test_the_index_recovers_once_the_extension_returns(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """Recovery must be an ordinary run, not another manual step.

    After the lexical-only rebuild above, the next run with the extension
    present has to recreate the vec table and embed every chunk. Without this
    the escape hatch strands the user in a permanently vector-less index.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _build_workspace(workspace)
    monkeypatch.setattr(db, "load_vec", lambda conn: False)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)
    monkeypatch.undo()

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=stub_embedder)

    assert stats.embedded_chunks > 0
    assert stats.vector_covered == stats.vector_total
    conn = db.connect(db_path)
    try:
        assert db.has_vec_table(conn) is True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Knowledge repo: reported on the run, and warned about only when NAMED
# ---------------------------------------------------------------------------


def test_a_run_reports_the_repo_that_got_whole_tree_treatment(
    tmp_path: Path, stub_embedder, monkeypatch
):
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)
    workspace = tmp_path / "workspace"
    _build_workspace(workspace, extra_repos=["the-brain"])
    _write(workspace / "the-brain" / "decisions" / "0001-x.md", "# 0001\n\n" + "x " * 20)
    # The engine running from inside the workspace it indexes: the shape every
    # developer checkout has, and the one the resolved name has to serve.
    monkeypatch.setattr(indexer.db, "source_checkout_root", lambda: workspace / "the-brain")

    stats = indexer.reindex(
        db_path=tmp_path / "var" / "index.db",
        workspace_root=workspace,
        embedder=stub_embedder,
    )

    assert stats.knowledge_repo == "the-brain"
    assert stats.errors == []


def test_a_run_against_someone_elses_workspace_reports_no_knowledge_repo(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """Pointing the engine at a workspace that is not its own is supported and
    ordinary (that is what ``BRAIN_WORKSPACE_ROOT`` is for), and there the
    resolved name legitimately matches nothing. Reporting the name anyway
    would claim whole-tree treatment that did not happen; warning about it
    would fire on a correct configuration and teach the reader to skip the
    line."""
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)
    workspace = tmp_path / "workspace"
    _build_workspace(workspace)
    outside = tmp_path / "somewhere-else"
    outside.mkdir()
    monkeypatch.setattr(indexer.db, "source_checkout_root", lambda: outside)

    stats = indexer.reindex(
        db_path=tmp_path / "var" / "index.db",
        workspace_root=workspace,
        embedder=stub_embedder,
    )

    assert stats.knowledge_repo is None
    assert stats.errors == []


def test_a_named_knowledge_repo_that_is_out_of_scope_is_reported(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """The one case with no benign reading: the operator stated an intention
    the run could not honour, and every knowledge repo's decisions/, context/,
    architecture/ and gaps/ directories were left out because of it."""
    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "a-repo-that-is-not-registered")
    workspace = tmp_path / "workspace"
    _build_workspace(workspace)

    stats = indexer.reindex(
        db_path=tmp_path / "var" / "index.db",
        workspace_root=workspace,
        embedder=stub_embedder,
    )

    assert stats.knowledge_repo is None
    assert len(stats.errors) == 1
    assert "a-repo-that-is-not-registered" in stats.errors[0]
    assert "BRAIN_KNOWLEDGE_REPO" in stats.errors[0]


def test_a_named_knowledge_repo_that_is_in_scope_is_not_reported(
    tmp_path: Path, stub_embedder, monkeypatch
):
    """The mirror direction, which nothing else here would catch: a warning
    that also fires on the correct configuration is worse than no warning."""
    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "the-brain")
    workspace = tmp_path / "workspace"
    _build_workspace(workspace, extra_repos=["the-brain"])
    _write(workspace / "the-brain" / "decisions" / "0001-x.md", "# 0001\n\n" + "x " * 20)

    stats = indexer.reindex(
        db_path=tmp_path / "var" / "index.db",
        workspace_root=workspace,
        embedder=stub_embedder,
    )

    assert stats.knowledge_repo == "the-brain"
    assert stats.errors == []


def test_a_knowledge_repo_change_reclassifies_the_rows_it_retains(tmp_path, monkeypatch):
    """``doc_type`` depends on the knowledge repo, and unchanged files skip it.

    The unchanged fast path returns before ``classify_doc_type`` runs, so a run
    that moves the knowledge repo leaves retained rows carrying the previous
    run's classification. Most whole-tree files leave the corpus entirely when
    the repo stops being the knowledge repo and the removal pass handles them;
    the survivors are the ones the ordinary corpus rules ALSO match, which is
    why this fixture uses ``decisions/docs/``: it is corpus either way, and
    classified differently by each.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# Repo A\n\n## Rules\n\n" + "text " * 20)
    _write(
        workspace / "repo-a" / "decisions" / "docs" / "note.md",
        "# Note\n\n## Body\n\n" + "note text " * 20,
    )
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))

    def doc_type_of(path: str) -> str:
        conn, _ = db.open_index(db_path)
        try:
            row = conn.execute(
                "SELECT doc_type FROM documents WHERE path = ?", (path,)
            ).fetchone()
        finally:
            conn.close()
        assert row is not None, f"{path} is not in the index"
        return row["doc_type"]

    target = "repo-a/decisions/docs/note.md"

    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "repo-a")
    indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert doc_type_of(target) == "decision"

    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "repo-b")
    moved = indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert moved.changed == 0, "the bytes did not change; this must be the retained path"
    assert moved.reclassified == 1
    assert doc_type_of(target) == "doc", (
        "the repo no longer receives whole-tree treatment, so the row must "
        "not keep asserting that it does"
    )

    settled = indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert settled.reclassified == 0, "nothing moved, so nothing may be rewritten"

    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "repo-a")
    back = indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert back.reclassified == 1
    assert doc_type_of(target) == "decision", "the correction has to run both ways"


def test_an_index_predating_the_stamp_is_reconciled_once(tmp_path, monkeypatch):
    """An index with no recorded knowledge repo holds rows classified under an
    unknown configuration, and that is the state worth clearing.

    The reconciliation computes the correct classification from THIS run's
    configuration rather than inferring anything about the old one, so running
    it without a recorded value to compare against is a correction, not a
    guess. The run stamps the value afterwards, which is what makes it happen
    exactly once instead of on every reindex.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# Repo A\n\n## Rules\n\n" + "text " * 20)
    _write(
        workspace / "repo-a" / "decisions" / "docs" / "note.md",
        "# Note\n\n## Body\n\n" + "note text " * 20,
    )
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))

    def doc_type_of(path: str) -> str:
        conn, _ = db.open_index(db_path)
        try:
            row = conn.execute(
                "SELECT doc_type FROM documents WHERE path = ?", (path,)
            ).fetchone()
        finally:
            conn.close()
        return row["doc_type"]

    target = "repo-a/decisions/docs/note.md"

    # Build with no knowledge repo, then forge a pre-stamp index: the row is
    # removed and the classification left as it was, which is what an index
    # built by an older version in a different configuration looks like.
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)
    indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert doc_type_of(target) == "doc"
    conn, _ = db.open_index(db_path)
    try:
        conn.execute("DELETE FROM meta WHERE key = ?", (db.META_KNOWLEDGE_REPO,))
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "repo-a")
    first = indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert first.changed == 0, "the bytes did not change; this must be the retained path"
    assert first.reclassified == 1
    assert doc_type_of(target) == "decision"

    # The count alone cannot show the scan was skipped: reconciliation is
    # idempotent, so a run that performs it needlessly also reports zero.
    # Observing the call is the only way to tell "did not need to change
    # anything" from "did not look".
    calls = []
    real = indexer._reclassify_after_knowledge_repo_change

    def counted(conn, knowledge):
        calls.append(knowledge)
        return real(conn, knowledge)

    monkeypatch.setattr(indexer, "_reclassify_after_knowledge_repo_change", counted)
    second = indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert second.reclassified == 0
    assert calls == [], (
        "the stamp written by the previous run is what lets a steady-state "
        "reindex skip the scan rather than repeat it"
    )


def test_classify_doc_type_treats_none_as_no_knowledge_repo(monkeypatch):
    """``None`` is a real value for this argument, not a request to go and look.

    A workspace with no knowledge repo passes ``None``. A default that resolves
    the environment on ``None`` turns that into a second resolution, per call,
    of a value the caller had already settled -- and the two agree only by
    luck, because the second reads the environment and the filesystem again.
    """
    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "the-brain")
    assert indexer.classify_doc_type("the-brain", "decisions/0001-x.md", None) == "doc", (
        "an explicit None means no repo receives whole-tree treatment"
    )
    # The default, by contrast, is what asks the environment.
    assert indexer.classify_doc_type("the-brain", "decisions/0001-x.md") == "decision"


def test_an_out_of_scope_knowledge_repo_does_not_classify_as_one(tmp_path, monkeypatch):
    """The index must not contradict the label it stamps on itself.

    ``BRAIN_KNOWLEDGE_REPO=workspace`` names the workspace pseudo-repo, which
    is never registered, so nothing receives whole-tree treatment and the run
    records no knowledge repo. Classifying with the RAW name anyway published
    ``decisions/0001-x.md`` as a ``decision`` under a stamp saying no repo was
    in scope: a row and the index's own metadata asserting opposite things.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# Repo A\n\n## Rules\n\n" + "text " * 20)
    _write(workspace / "AGENTS.md", "# Workspace\n\n## Overview\n\n" + "text " * 20)
    _write(workspace / "decisions" / "0001-x.md", "# 0001\n\n## Body\n\n" + "x " * 20)
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "workspace")

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert stats.knowledge_repo is None

    conn, _ = db.open_index(db_path)
    try:
        row = conn.execute(
            "SELECT doc_type FROM documents WHERE path = ?", ("decisions/0001-x.md",)
        ).fetchone()
        stamp = db.get_meta(conn, db.META_KNOWLEDGE_REPO)
    finally:
        conn.close()
    assert stamp == "", "no repo was in scope, and the run must say so"
    assert row["doc_type"] == "doc", (
        "the stamp says no repo received whole-tree treatment, so no row may "
        "carry a classification that only whole-tree treatment produces"
    )


def test_the_reconciliation_runs_after_the_removal_sweep(tmp_path, monkeypatch):
    """Rows on their way out must not be counted as corrections.

    A file that is corpus ONLY because its repo is the knowledge repo leaves the
    corpus when that stops being true, and the removal sweep deletes it.
    Reconciling before that sweep would rewrite and count rows that are about
    to disappear, inflating a number an operator reads as work done.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# Repo A\n\n## Rules\n\n" + "text " * 20)
    # Corpus only under whole-tree treatment: context/ matches no ordinary rule.
    _write(workspace / "repo-a" / "context" / "note.md", "# Note\n\n## Body\n\n" + "x " * 20)
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))

    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "repo-a")
    indexer.reindex(db_path=db_path, workspace_root=workspace)
    conn, _ = db.open_index(db_path)
    try:
        present = conn.execute(
            "SELECT doc_type FROM documents WHERE path = ?", ("repo-a/context/note.md",)
        ).fetchone()
    finally:
        conn.close()
    assert present is not None and present["doc_type"] == "context"

    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO")
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert stats.removed == 1, "the file is no longer corpus, so it must be deleted"
    assert stats.reclassified == 0, (
        "a deleted row is not a corrected row; counting it would report work "
        "that did not happen to a document that no longer exists"
    )


def test_the_common_no_knowledge_repo_steady_state_skips_the_scan(tmp_path, monkeypatch):
    """The arm the comparison is easiest to get wrong on.

    A workspace with no knowledge repo stamps the empty string and resolves
    ``None``, so the two sides of the comparison are ``""`` and ``None`` and
    only agree after normalisation. Dropping that normalisation reconciles on
    every single run forever, and because reconciliation is idempotent the
    COUNT stays zero, so nothing that watches the count can notice. Only
    observing the call can.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# Repo A\n\n## Rules\n\n" + "text " * 20)
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)

    indexer.reindex(db_path=db_path, workspace_root=workspace)
    conn, _ = db.open_index(db_path)
    try:
        assert db.get_meta(conn, db.META_KNOWLEDGE_REPO) == ""
    finally:
        conn.close()

    calls = []
    real = indexer._reclassify_after_knowledge_repo_change

    def counted(conn, knowledge):
        calls.append(knowledge)
        return real(conn, knowledge)

    monkeypatch.setattr(indexer, "_reclassify_after_knowledge_repo_change", counted)
    indexer.reindex(db_path=db_path, workspace_root=workspace)
    assert calls == [], (
        'the stamp is "" and the run resolves None; those are the same answer '
        "and a run that treats them as a change never stops reconciling"
    )


def _stored_paths(db_path: Path) -> list[str]:
    conn = db.connect(db_path)
    try:
        return [r["path"] for r in conn.execute("SELECT path FROM documents ORDER BY path")]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "configured",
    [
        pytest.param(lambda root: f"{root},{root.parent / 'other'}", id="comma-separator"),
        pytest.param(lambda root: str(root.parent / "renamed"), id="moved-root"),
    ],
)
def test_roots_that_name_no_directory_are_refused_and_the_index_is_kept(
    tmp_path: Path, monkeypatch, stub_embedder, configured
):
    """Issue #78: both routes emptied the index with errors empty and a zero exit.

    Asserted on the index contents after the refusal, not only on the raise: the
    damage was the removal sweep, so the observable is that it never ran.
    """
    root = tmp_path / "w1" / "mydocs"
    db_path = tmp_path / "var" / "index.db"
    _write(root / "README.md", "# One\n\n" + "kept content " * 20)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(root))
    indexer.reindex(db_path=db_path, embedder=stub_embedder)
    before = _stored_paths(db_path)
    assert before == ["mydocs/README.md"]

    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", configured(root))
    with pytest.raises(indexer.CorpusRootsUnusable):
        indexer.reindex(db_path=db_path, embedder=stub_embedder)

    assert _stored_paths(db_path) == before


def test_a_comma_separated_value_is_named_as_the_likely_typo(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", f"{tmp_path / 'a'},{tmp_path / 'b'}")
    with pytest.raises(indexer.CorpusRootsUnusable) as excinfo:
        indexer.discover_corpus()
    message = str(excinfo.value)
    assert repr(os.pathsep) in message
    assert "','" in message


def test_some_missing_roots_are_still_skipped_rather_than_refused(tmp_path: Path, monkeypatch):
    """The narrowness control: a partly present configuration is legitimate."""
    present = tmp_path / "w1" / "mydocs"
    _write(present / "README.md", "# One\n\n" + "text " * 20)
    monkeypatch.setenv(
        "BRAIN_CORPUS_ROOTS", os.pathsep.join([str(tmp_path / "absent"), str(present)])
    )
    assert {d.rel_path for d in indexer.discover_corpus()} == {"mydocs/README.md"}


def test_a_present_but_empty_root_is_not_refused(tmp_path: Path, monkeypatch):
    """The refusal is about roots that are not directories, not about a corpus
    that happens to be empty: an existing directory is a usable root."""
    empty = tmp_path / "w1" / "empty"
    empty.mkdir(parents=True)
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", str(empty))
    assert indexer.discover_corpus() == []
