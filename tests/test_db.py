from __future__ import annotations

import sqlite3
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
