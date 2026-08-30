from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from conftest import write_registry

from corpusdex import db, indexer, links


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
    # Not corpus: no matching pattern (bare file at repo root, not one of the
    # three recognised root filenames).
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
        tmp_path / "repo-a" / "audit-api" / "docs" / "auth-and-roles.md",
        "# Auth and roles\n\n" + "text " * 20,
    )
    write_registry(tmp_path, ["repo-a"])
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path for d in docs}
    assert "repo-a/audit-api/docs/auth-and-roles.md" in rel_paths


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
        tmp_path / "repo-a-some-other-worktree" / "audit-api" / "docs" / "leaked.md",
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
        tmp_path / "repo-a" / "audit-api" / "docs" / "real.md",
        "# Real\n\n" + "text " * 20,
    )
    _write(
        tmp_path
        / "repo-a"
        / ".claude"
        / "worktrees"
        / "agent-abc123"
        / "audit-api"
        / "docs"
        / "real.md",
        "# Duplicate\n\n" + "text " * 20,
    )
    write_registry(tmp_path, ["repo-a"])
    docs = indexer.discover_corpus(tmp_path)
    rel_paths = {d.rel_path for d in docs}
    assert "repo-a/audit-api/docs/real.md" in rel_paths
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
    # re-embed everything, even though no document content changed.
    second = indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=NamedEmbedder("model-b")
    )
    assert second.added == 0
    assert second.changed == 0
    assert second.embedded_chunks == second.vector_total
    assert any("model changed" in e for e in second.errors)

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
        root / "repo-a" / "docs" / "decisions" / "0006-tenant-default.md",
        f"# Tenant default\n\n## Body\n\n{filler}\n",
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

    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=DeadUnderANewName())

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


def test_the_knowledge_repo_defaults_to_the_checkout_it_runs_from(monkeypatch, tmp_path):
    """Not a constant: a constant naming this repo silently stops matching the
    day the package is renamed, and every dev checkout then resolves as
    'installed'. Installed from a wheel there is no checkout, so no repo gets
    whole-tree treatment unless one is named."""
    monkeypatch.delenv("BRAIN_KNOWLEDGE_REPO", raising=False)

    monkeypatch.setattr(indexer.db, "source_checkout_root", lambda: tmp_path / "some-checkout")
    assert indexer.knowledge_repo() == "some-checkout"

    monkeypatch.setattr(indexer.db, "source_checkout_root", lambda: None)
    assert indexer.knowledge_repo() is None

    monkeypatch.setenv("BRAIN_KNOWLEDGE_REPO", "named-explicitly")
    assert indexer.knowledge_repo() == "named-explicitly"


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
