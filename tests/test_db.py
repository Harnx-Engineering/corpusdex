from __future__ import annotations

import functools
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from corpusdex import db


def test_workspace_root_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(tmp_path))
    assert db.workspace_root() == tmp_path.resolve()


def test_default_db_path_env_override(monkeypatch, tmp_path: Path):
    target = tmp_path / "custom" / "index.db"
    monkeypatch.setenv("BRAIN_DB", str(target))
    assert db.default_db_path() == target.resolve()


def _install_package_at(root: Path, monkeypatch) -> None:
    """Pretend the package lives at ``root/corpusdex`` for path resolution."""
    package = root / "corpusdex"
    package.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(db, "__file__", str(package / "db.py"))


def test_the_real_checkout_is_found_by_its_declared_name():
    root = db.source_checkout_root()
    assert root is not None
    assert (root / "pyproject.toml").is_file()
    assert db.PROJECT_NAME in (root / "pyproject.toml").read_text(encoding="utf-8")


def test_a_pyproject_naming_another_project_is_not_our_checkout(tmp_path: Path, monkeypatch):
    """The venv-inside-another-project case, which is the common install shape.

    Matching any ``pyproject.toml`` claimed the HOST project's root, so the
    index was written into their repo and the corpus root became their parent
    directory -- indexing an unrelated tree.
    """
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "someone-elses-app"\nversion = "1.0"\n', encoding="utf-8"
    )
    _install_package_at(tmp_path / ".venv" / "lib" / "site-packages", monkeypatch)
    assert db.source_checkout_root() is None


def test_a_malformed_pyproject_does_not_crash_resolution(tmp_path: Path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("this is not [ valid toml", encoding="utf-8")
    _install_package_at(tmp_path / "site-packages", monkeypatch)
    assert db.source_checkout_root() is None


def test_state_dir_is_var_inside_a_source_checkout(monkeypatch):
    monkeypatch.delenv("BRAIN_STATE_DIR", raising=False)
    root = db.source_checkout_root()
    assert root is not None
    assert db.state_dir() == root / "var"


def test_state_dir_leaves_site_packages_alone_when_installed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BRAIN_STATE_DIR", raising=False)
    site_packages = tmp_path / "site-packages"
    _install_package_at(site_packages, monkeypatch)
    resolved = db.state_dir()
    assert site_packages not in resolved.parents
    assert resolved.name == db.PROJECT_NAME


def test_state_dir_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BRAIN_STATE_DIR", str(tmp_path / "elsewhere"))
    assert db.state_dir() == (tmp_path / "elsewhere").resolve()
    assert db.default_db_path() == (tmp_path / "elsewhere" / "index.db").resolve()


def test_workspace_root_refuses_to_guess_when_installed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BRAIN_WORKSPACE_ROOT", raising=False)
    _install_package_at(tmp_path / "site-packages", monkeypatch)
    with pytest.raises(db.NotConfigured):
        db.workspace_root()


def test_repo_root_refuses_to_guess_when_installed(tmp_path: Path, monkeypatch):
    _install_package_at(tmp_path / "site-packages", monkeypatch)
    with pytest.raises(db.NotConfigured):
        db.repo_root()


def test_the_write_lock_does_not_import_fcntl_at_module_scope():
    """``fcntl`` is absent on Windows, so importing it eagerly killed every
    entry point at import, not only the writers."""
    source = Path(db.__file__).read_text(encoding="utf-8")
    assert "\nimport fcntl\n" not in source
    assert "except ModuleNotFoundError" in source


def test_lock_helpers_round_trip_on_this_platform(tmp_path: Path):
    import os as _os

    path = tmp_path / "probe.lock"
    fd = _os.open(path, _os.O_CREAT | _os.O_RDWR, 0o644)
    try:
        db._lock_exclusive(fd)
        db._unlock(fd)
        db._lock_exclusive(fd)
        db._unlock(fd)
    finally:
        _os.close(fd)


def test_vec_schema_rejects_a_nonpositive_dimension():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            db.vec_schema(bad)


def test_vec_table_dim_reads_the_declared_width(tmp_path: Path, vec_probe):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    conn, vec = db.open_index(tmp_path / "index.db", create=True)
    try:
        assert vec
        assert db.vec_table_dim(conn) == db.EMBED_DIM
    finally:
        conn.close()


def test_vec_table_dim_is_none_when_the_table_is_absent(tmp_path: Path):
    conn = db.connect(tmp_path / "index.db")
    try:
        db.init_schema(conn, vec=False)
        assert db.vec_table_dim(conn) is None
    finally:
        conn.close()


def test_recreate_vec_table_changes_the_width_and_discards_vectors(tmp_path: Path, vec_probe):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    import sqlite_vec

    conn, vec = db.open_index(tmp_path / "index.db", create=True)
    try:
        assert vec
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (1, sqlite_vec.serialize_float32([0.0] * db.EMBED_DIM)),
        )
        conn.commit()
        assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 1

        db.recreate_vec_table(conn, 384)
        conn.commit()
        assert db.vec_table_dim(conn) == 384
        assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 0
        # And the new width is the one the table now enforces.
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (1, sqlite_vec.serialize_float32([0.0] * 384)),
        )
        conn.commit()
    finally:
        conn.close()


def test_init_schema_creates_core_tables(tmp_path: Path):
    conn = db.connect(tmp_path / "index.db")
    try:
        db.init_schema(conn, vec=False)
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        assert {"documents", "chunks", "chunks_fts", "meta"} <= tables
        assert db.get_meta(conn, db.META_SCHEMA_VERSION) == str(db.SCHEMA_VERSION)
    finally:
        conn.close()


def test_init_schema_is_idempotent(tmp_path: Path):
    conn = db.connect(tmp_path / "index.db")
    try:
        db.init_schema(conn, vec=False)
        db.init_schema(conn, vec=False)  # must not raise
        assert db.get_meta(conn, db.META_SCHEMA_VERSION) == str(db.SCHEMA_VERSION)
    finally:
        conn.close()


def test_fts_trigger_keeps_index_in_sync(tmp_path: Path):
    conn = db.connect(tmp_path / "index.db")
    try:
        db.init_schema(conn, vec=False)
        with conn:
            conn.execute(
                "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
                "VALUES ('repo', 'a.md', 'Title', 'doc', 1.0, 'hash')"
            )
            conn.execute(
                "INSERT INTO chunks (ref, doc_id, heading_path, body) "
                "VALUES ('cfts0000000000000', 1, 'Title > Section', "
                "'findableuniquephrase here')"
            )
        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'findableuniquephrase'"
        ).fetchall()
        assert len(rows) == 1

        with conn:
            conn.execute("DELETE FROM chunks WHERE id = 1")
        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'findableuniquephrase'"
        ).fetchall()
        assert rows == []
    finally:
        conn.close()


def test_documents_path_is_unique(tmp_path: Path):
    conn = db.connect(tmp_path / "index.db")
    try:
        db.init_schema(conn, vec=False)
        with conn:
            conn.execute(
                "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
                "VALUES ('repo', 'a.md', 'Title', 'doc', 1.0, 'hash')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
                    "VALUES ('repo', 'a.md', 'Other title', 'doc', 2.0, 'hash2')"
                )
    finally:
        conn.close()


def test_write_lock_is_exclusive_and_non_blocking(tmp_path: Path):
    db_path = tmp_path / "index.db"
    with db.write_lock(db_path):
        with pytest.raises(db.IndexLocked):
            with db.write_lock(db_path):
                pass  # pragma: no cover - must not be reached


def test_write_lock_releases_after_context_exit(tmp_path: Path):
    db_path = tmp_path / "index.db"
    with db.write_lock(db_path):
        pass
    # A second, later acquisition must succeed now that the first released.
    with db.write_lock(db_path):
        pass


def test_open_index_degrades_gracefully_when_vec_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(db, "load_vec", lambda conn: False)
    conn, vec_ok = db.open_index(tmp_path / "index.db", create=True)
    try:
        assert vec_ok is False
        assert db.has_vec_table(conn) is False
    finally:
        conn.close()


def test_open_index_creates_vec_table_when_extension_loads(tmp_path: Path, vec_probe: bool):
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    conn, vec_ok = db.open_index(tmp_path / "index.db", create=True)
    try:
        assert vec_ok is True
        assert db.has_vec_table(conn) is True
    finally:
        conn.close()


def test_open_index_raises_on_schema_version_mismatch(tmp_path: Path):
    db_path = tmp_path / "index.db"
    conn, _vec_ok = db.open_index(db_path, create=True)
    with conn:
        db.set_meta(conn, db.META_SCHEMA_VERSION, str(db.SCHEMA_VERSION + 1))
    conn.close()

    with pytest.raises(db.SchemaVersionMismatch) as exc_info:
        db.open_index(db_path)
    message = str(exc_info.value)
    assert str(db.SCHEMA_VERSION + 1) in message
    # The advice a read surface prints has to be a command that actually
    # recovers. `reindex --full` does not: it opens the index the same way
    # and would hit this very error.
    assert "brain reindex" in message
    assert "--full" not in message


def _stamp(db_path: Path, version: str) -> None:
    conn, _vec_ok = db.open_index(db_path, create=True)
    with conn:
        conn.execute(
            "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
            "VALUES ('repo-a', 'repo-a/AGENTS.md', 'A', 'agents', 1.0, '1:x')"
        )
        db.set_meta(conn, db.META_SCHEMA_VERSION, version)
    conn.close()


def _table_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        conn.close()


def test_reader_never_writes_ddl_into_a_version_mismatched_index(tmp_path: Path):
    # A reader that ran CREATE TABLE IF NOT EXISTS was upgrading, lock-free
    # and outside the single-writer lock, an index it had already decided it
    # could not read.
    db_path = tmp_path / "index.db"
    _stamp(db_path, str(db.SCHEMA_VERSION - 1))
    before = _table_names(db_path)
    before_size = db_path.stat().st_size

    with pytest.raises(db.SchemaVersionMismatch):
        db.open_index(db_path)

    assert _table_names(db_path) == before
    assert db_path.stat().st_size == before_size


def test_reader_refuses_an_empty_index_file_instead_of_minting_one(tmp_path: Path):
    # is_file() guards absence, not emptiness: a 0-byte file used to be
    # minted into a full schema by a reader and served as an empty index.
    db_path = tmp_path / "index.db"
    db_path.touch()

    with pytest.raises(db.IndexMissing):
        db.open_index(db_path)

    assert db_path.stat().st_size == 0


def test_reader_refuses_a_file_that_is_not_a_database(tmp_path: Path):
    db_path = tmp_path / "index.db"
    db_path.write_bytes(b"not a sqlite database at all")

    with pytest.raises(db.IndexMissing):
        db.open_index(db_path)


def test_writer_refuses_an_index_newer_than_this_build(tmp_path: Path):
    # An older checkout must not silently downgrade a healthy newer index.
    db_path = tmp_path / "index.db"
    _stamp(db_path, str(db.SCHEMA_VERSION + 1))

    with pytest.raises(db.SchemaVersionMismatch) as exc_info:
        db.open_index(db_path, create=True)
    assert "older than the index" in str(exc_info.value)

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    finally:
        conn.close()


def test_stored_schema_version_does_not_create_or_modify(tmp_path: Path):
    missing = tmp_path / "absent.db"
    assert db.stored_schema_version(missing) is None
    assert not missing.exists()

    empty = tmp_path / "empty.db"
    empty.touch()
    assert db.stored_schema_version(empty) is None
    assert empty.stat().st_size == 0


def test_swap_index_replaces_the_live_file_and_clears_stale_sidecars(tmp_path: Path):
    live = tmp_path / "index.db"
    _stamp(live, str(db.SCHEMA_VERSION))
    stale_wal = Path(f"{live}-wal")
    stale_wal.write_bytes(b"stale log")

    fresh = tmp_path / "index.db.rebuild"
    _stamp(fresh, str(db.SCHEMA_VERSION))

    db.swap_index(fresh, live)

    assert not fresh.exists()
    assert not stale_wal.exists()
    conn, _vec_ok = db.open_index(live)
    conn.close()


def test_open_index_raises_index_missing_when_db_absent_and_create_is_false(tmp_path: Path):
    db_path = tmp_path / "does-not-exist" / "index.db"
    with pytest.raises(db.IndexMissing) as exc_info:
        db.open_index(db_path)
    message = str(exc_info.value)
    assert "brain reindex" in message
    # A reader that refuses to create must also not have left a file behind.
    assert not db_path.exists()


def test_open_index_does_not_create_db_file_on_a_fresh_checkout(tmp_path: Path):
    db_path = tmp_path / "var" / "index.db"
    with pytest.raises(db.IndexMissing):
        db.open_index(db_path)
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_open_index_with_create_true_builds_the_db_from_nothing(tmp_path: Path):
    db_path = tmp_path / "var" / "index.db"
    conn, _vec_ok = db.open_index(db_path, create=True)
    try:
        assert db_path.is_file()
    finally:
        conn.close()

    # Once the file exists, a plain reader (create defaults to False) can
    # open it without needing create=True again.
    conn, _vec_ok = db.open_index(db_path)
    conn.close()


def test_a_pyproject_whose_project_key_is_not_a_table_does_not_crash(tmp_path: Path, monkeypatch):
    """``project = "x"`` is valid TOML. Reading it with ``.get()`` raises
    AttributeError, and because source_checkout_root() walks EVERY parent, one
    such file anywhere above the package kills every entry point rather than
    just failing to match. Malformed-TOML coverage does not reach this: the
    file parses fine."""
    root = tmp_path / "outer"
    (root / "inner").mkdir(parents=True)
    (root / "pyproject.toml").write_text('project = "not-a-table"\n', encoding="utf-8")
    _install_package_at(root / "inner", monkeypatch)

    assert db.source_checkout_root() is None


def test_recreate_vec_table_does_not_commit_the_callers_pending_work(
    tmp_path: Path, vec_probe
):
    """The rebuild must join the caller's transaction, not end it.

    ``executescript()`` issues an implicit COMMIT before running, so the
    reindex's half-written documents were durably committed the moment a width
    change was detected, and the DROP then sat outside any transaction. Under
    ``execute()`` both the pending DML and the DDL are one unit: a later
    failure in the same reindex rolls the vectors back rather than leaving the
    index permanently emptied.
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    import sqlite_vec

    conn, vec = db.open_index(tmp_path / "index.db", create=True)
    try:
        assert vec
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (1, sqlite_vec.serialize_float32([0.0] * db.EMBED_DIM)),
        )
        conn.commit()

        # Pending caller work, exactly as reindex has in flight when it
        # notices the width change.
        conn.execute(
            "INSERT INTO documents(repo, path, title, doc_type, mtime, content_hash)"
            " VALUES ('r', 'a.md', 'A', 'doc', 1.0, 'h')"
        )
        db.recreate_vec_table(conn, 384)
        conn.rollback()

        assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
        assert db.vec_table_dim(conn) == db.EMBED_DIM
        assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 1

        # And with nothing in flight either. sqlite3's legacy mode opens a
        # transaction only ahead of DML, so DDL on its own autocommits: this
        # is the case reindex actually hits, since the width check runs before
        # any document is written.
        assert not conn.in_transaction
        db.recreate_vec_table(conn, 384)
        conn.rollback()
        assert db.vec_table_dim(conn) == db.EMBED_DIM
        assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 1
    finally:
        conn.close()


def test_stale_embed_model_reports_only_a_real_disagreement():
    """Both names present and different is the only mismatch.

    The three ``None`` cases are distinct situations collapsed on purpose,
    because every one of them carries the same instruction: do nothing. An
    index with no recorded model and a caller that cannot name its own model
    are both unknown, and treating unknown as disagreement would degrade
    correct configurations.
    """
    assert db.stale_embed_model("model-one", "model-two") == "model-one"
    assert db.stale_embed_model("model-one", "model-one") is None
    assert db.stale_embed_model(None, "model-two") is None
    assert db.stale_embed_model("model-one", None) is None
    assert db.stale_embed_model(None, None) is None


def test_stale_embed_model_treats_an_empty_recorded_name_as_absent():
    """An empty string in ``meta`` is a row that never got a real value, not
    a model called "". Reporting it would name nothing useful in the reason
    the caller then prints."""
    assert db.stale_embed_model("", "model-two") is None
    assert db.stale_embed_model("model-one", "") is None


def _real_git_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a real canonical checkout and a real linked worktree.

    Built with git rather than by hand on purpose: every other case in this
    file can be written from what the code expects, but the ONE thing worth
    proving here is that git's actual on-disk shape is the shape being parsed.
    A hand-built fixture would pass against a parser written from the same
    belief, and the defect this guards was exactly a belief about layout.
    """
    git = shutil.which("git")
    if git is None:  # pragma: no cover - git is present wherever this repo is
        pytest.skip("git is not available")
    canonical = tmp_path / "the-repo"
    canonical.mkdir()
    run = functools.partial(
        subprocess.run, cwd=canonical, check=True, capture_output=True, text=True
    )
    run([git, "init", "-q", "-b", "main"])
    run([git, "config", "user.email", "tests@example.invalid"])
    run([git, "config", "user.name", "tests"])
    (canonical / "seed.txt").write_text("seed\n", encoding="utf-8")
    run([git, "add", "seed.txt"])
    run([git, "commit", "-qm", "seed"])
    linked = tmp_path / ".worktrees" / "the-repo" / "issue-1-some-task"
    run([git, "worktree", "add", "-q", "-b", "issue-1-some-task", str(linked)])
    return canonical, linked


def test_a_linked_worktree_resolves_to_the_checkout_it_belongs_to(tmp_path: Path):
    """The defect: the worktree DIRECTORY is named for the task, not the repo.

    Nothing about ``issue-1-some-task`` names ``the-repo``, and every caller
    that wants to know which repo's content this is was reading that name.
    """
    canonical, linked = _real_git_worktree(tmp_path)
    assert linked.name != canonical.name
    assert db.canonical_checkout(linked) == canonical.resolve()
    assert db.canonical_checkout(linked).name == "the-repo"


def test_an_ordinary_checkout_is_returned_unchanged(tmp_path: Path):
    """A ``.git`` DIRECTORY is the canonical case and must not be rewritten."""
    canonical, _ = _real_git_worktree(tmp_path)
    assert db.canonical_checkout(canonical) == canonical


def test_a_directory_that_is_not_a_checkout_at_all_is_returned_unchanged(tmp_path: Path):
    plain = tmp_path / "no-git-here"
    plain.mkdir()
    assert db.canonical_checkout(plain) == plain


def _real_git_submodule(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a real superproject, a real submodule, and a worktree of it.

    The hand-built versions of these two cases were the weakest fixtures in
    this file, and by the same argument :func:`_real_git_worktree` makes: the
    two branches they cover are the ones that DISCRIMINATE, and writing the
    layout from the parser's own belief about it proves only that the belief
    is self-consistent. ``protocol.file.allow`` is needed because git refuses
    local-path submodules by default since CVE-2022-39253; nothing leaves this
    directory.
    """
    git = shutil.which("git")
    if git is None:  # pragma: no cover - git is present wherever this repo is
        pytest.skip("git is not available")

    def init(at: Path) -> None:
        at.mkdir(parents=True, exist_ok=True)
        run = functools.partial(
            subprocess.run, cwd=at, check=True, capture_output=True, text=True
        )
        run([git, "init", "-q", "-b", "main"])
        run([git, "config", "user.email", "tests@example.invalid"])
        run([git, "config", "user.name", "tests"])
        (at / "seed.txt").write_text("seed\n", encoding="utf-8")
        run([git, "add", "seed.txt"])
        run([git, "commit", "-qm", "seed"])

    origin = tmp_path / "origin-of-vendored"
    init(origin)
    superproject = tmp_path / "super"
    init(superproject)
    sup = functools.partial(
        subprocess.run, cwd=superproject, check=True, capture_output=True, text=True
    )
    sup([git, "-c", "protocol.file.allow=always", "submodule", "add", "-q",
         str(origin), "vendored"])
    sup([git, "commit", "-qm", "add submodule"])
    submodule = superproject / "vendored"
    linked = tmp_path / "sub-worktrees" / "slug"
    subprocess.run(
        [git, "worktree", "add", "-q", "-b", "slug", str(linked)],
        cwd=submodule, check=True, capture_output=True, text=True,
    )
    return superproject, submodule, linked


def test_a_submodule_keeps_its_own_name(tmp_path: Path):
    """A submodule's ``.git`` is a FILE too, so the file alone cannot decide.

    Its metadata directory is ``.git/modules/<name>`` and carries no
    ``commondir``; a rule keyed on "``.git`` is a file" would resolve a
    submodule to its SUPERPROJECT and index the submodule's tree under the
    wrong repo name. A submodule is its own repo and keeps its own name.
    """
    _, submodule, _ = _real_git_submodule(tmp_path)
    # Real git writes this one RELATIVE, which the hand-built version of this
    # test did not, so the resolution of the gitdir line is exercised here too.
    assert "gitdir: ../.git/modules/vendored" in (submodule / ".git").read_text()
    assert db.canonical_checkout(submodule) == submodule


def test_a_worktree_of_a_submodule_keeps_its_own_name(tmp_path: Path):
    """A worktree of a SUBMODULE has a commondir, so the commondir check alone
    admits it -- and it points at ``.git/modules/<name>``, not at a ``.git``.
    Following it anyway would name the submodule's worktree after the
    SUPERPROJECT, indexing one repo's content under another repo's name. The
    check that the target is really a ``.git`` is what stops that.
    """
    superproject, _, linked = _real_git_submodule(tmp_path)
    # The shape the branch depends on, asserted against what git wrote rather
    # than against what this test would have written: a commondir IS present,
    # so the commondir check alone admits this, and it targets
    # .git/modules/vendored.
    meta = Path((linked / ".git").read_text().split("gitdir:", 1)[1].strip())
    assert (meta / "commondir").is_file()
    resolved = (meta / (meta / "commondir").read_text().strip()).resolve()
    assert resolved == (superproject / ".git" / "modules" / "vendored").resolve()
    assert db.canonical_checkout(linked) == linked


def test_a_relative_gitdir_resolves_against_the_worktree(tmp_path: Path):
    """git writes an absolute gitdir today, but the format permits a relative
    one and a checkout moved with its metadata intact produces one. Resolving
    it against the process's cwd instead of the worktree would silently point
    at whatever happens to sit there."""
    canonical = tmp_path / "the-repo"
    meta = canonical / ".git" / "worktrees" / "slug"
    meta.mkdir(parents=True)
    (meta / "commondir").write_text("../..\n", encoding="utf-8")
    linked = tmp_path / "trees" / "slug"
    linked.mkdir(parents=True)
    (linked / ".git").write_text("gitdir: ../../the-repo/.git/worktrees/slug\n", encoding="utf-8")
    assert db.canonical_checkout(linked) == canonical.resolve()


@pytest.mark.parametrize(
    "marker",
    [
        "",
        "not a gitdir line\n",
        "gitdir:\n",
        "gitdir: /nonexistent/path/that/does/not/exist\n",
    ],
    ids=["empty", "no-gitdir-key", "empty-gitdir", "gitdir-missing"],
)
def test_an_unreadable_link_leaves_the_directory_as_its_own_answer(tmp_path: Path, marker: str):
    """This runs on the way to NAMING a repo, not on the way to deleting
    anything, so a checkout git cannot explain is better described by the
    directory it sits in than by an exception out of a reindex."""
    root = tmp_path / "puzzling"
    root.mkdir()
    (root / ".git").write_text(marker, encoding="utf-8")
    assert db.canonical_checkout(root) == root


def test_git_metadata_that_is_not_utf8_leaves_the_directory_as_its_own_answer(tmp_path):
    """The contract is that anything git cannot explain returns ``root``.

    ``UnicodeDecodeError`` is a ``ValueError``, so an ``except OSError`` guard
    does not hold it. Git writes a gitdir path as raw bytes, so a checkout
    under a path that is not valid UTF-8 produces a file this reads and cannot
    decode, and letting that escape crashes reindex and status from the one
    function written to fail soft.
    """
    root = tmp_path / "wt"
    root.mkdir()
    (root / ".git").write_bytes(b"gitdir: /somewhere/\xff\xfe/path\n")
    assert db.canonical_checkout(root) == root


def test_a_commondir_that_is_not_utf8_leaves_the_directory_as_its_own_answer(tmp_path):
    """The same hazard on the second read, which has its own try block."""
    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "meta"
    gitdir.mkdir()
    (root / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    (gitdir / "commondir").write_bytes(b"../\xff\xfe/.git\n")
    assert db.canonical_checkout(root) == root


def test_the_workspace_root_is_the_canonical_checkouts_parent(tmp_path: Path, monkeypatch):
    """A linked worktree sits at ``<workspace>/.worktrees/<repo>/<slug>``, so
    its own parent holds one repo's worktrees and no registry. Deriving the
    workspace from it made every worktree run refuse before indexing."""
    monkeypatch.delenv("BRAIN_WORKSPACE_ROOT", raising=False)
    canonical, linked = _real_git_worktree(tmp_path)
    monkeypatch.setattr(db, "source_checkout_root", lambda: linked)
    assert db.workspace_root() == tmp_path.resolve()
    assert db.workspace_root() != linked.parent
