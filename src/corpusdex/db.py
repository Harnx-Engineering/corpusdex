"""SQLite storage for the brain index.

The index is a single, disposable file (``var/index.db`` by default). Markdown in
the workspace is the source of truth, so any corruption is repaired by deleting
the file and reindexing.

Four storage layers live in the same database:

* ``documents`` / ``chunks``: the canonical rows, one chunk per indexed section.
* ``chunks_fts``: an FTS5 external-content index over the chunk text, kept in
  sync by triggers.
* ``vec_chunks``: a sqlite-vec virtual table holding the chunk embeddings. It is
  created only when the sqlite-vec extension actually loads, and every access is
  guarded by :func:`load_vec` returning True for the live connection.
* ``doc_link_targets`` / ``doc_links``: the document link graph.
  ``doc_link_targets`` holds the raw, still-unresolved link text extracted
  from each document and is owned by that document (cascade-deleted with it);
  ``doc_links`` holds the resolved document-to-document edges and is rebuilt
  wholesale at the end of every reindex. Resolution cannot happen while
  documents are being read one at a time, because a link routinely points at
  a document that has not been indexed yet, and a re-chunked document is
  reinserted under a new id.

Writers take an exclusive, non-blocking lock on ``index.lock`` beside the
database; readers take nothing. The lock is ``flock`` on POSIX and
``msvcrt.locking`` on Windows -- ``fcntl`` does not exist there, and importing
it unconditionally killed every entry point at import time, not just the
writers.

Only the writer creates or alters the index file: :func:`open_index` defaults
to ``create=False``, so a read surface opened before the first ``brain
reindex`` raises :class:`IndexMissing` instead of silently minting an empty
database, and a read surface never executes schema DDL at all. Rebuilding a
stale index happens in a scratch file that is swapped into place atomically
(:func:`swap_index`), so a reader can never observe a half-built index.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import sqlite3
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

from .config import env

try:  # POSIX
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - exercised only on Windows
    fcntl = None
    try:
        import msvcrt
    except ModuleNotFoundError:
        # Neither locking primitive exists. Import must still succeed, or we
        # reintroduce at this line exactly the import-time total failure that
        # moving `fcntl` out of module scope was meant to remove; the write
        # lock reports the platform when a writer actually asks for it.
        msvcrt = None

SCHEMA_VERSION = 3

#: Dimension of the default embedding model (``nomic-embed-text``). It is the
#: dimension a fresh index is created with, NOT a property of the index: the
#: live width is whatever ``vec_chunks`` was declared with, readable through
#: :func:`vec_table_dim` and recorded in ``meta`` under
#: :data:`META_EMBED_DIM`. A model of a different width is a supported
#: configuration; see :func:`recreate_vec_table`.
EMBED_DIM = 768

CHUNK_REF_PREFIX = "c"
CHUNK_REF_HEX = 16

BUSY_TIMEOUT_MS = 5000

META_SCHEMA_VERSION = "schema_version"
META_EMBED_MODEL = "embed_model"
META_EMBED_STATUS = "embed_status"
META_LAST_REINDEX = "last_reindex_at"
META_LINK_EXTRACTOR_VERSION = "link_extractor_version"
META_EMBED_DIM = "embed_dim"

EMBED_STATUS_READY = "ready"
EMBED_STATUS_UNAVAILABLE = "unavailable"
EMBED_STATUS_DISABLED = "disabled"

#: Retrieval mode: the channels that actually produced a result page, joined
#: in pipeline order. It is RENDERED from
#: :attr:`corpusdex.search.SearchResponse.channels_used`, never computed
#: from whether the embedding backend was reachable, so it cannot contradict
#: that field. The previous two-value form was derived from availability and
#: was false in both directions: ``lexical-only`` was returned for pages the
#: graph channel had re-ranked, and ``hybrid`` was returned to a caller that
#: had narrowed ``channels`` to lexical, because nothing had failed. Naming
#: the channels removes the vocabulary that made those claims possible: there
#: is no summary word left to be wrong, only a list that is either right or a
#: bug in one line of rendering.
#:
#: The constants below name the reachable combinations. They are not an
#: exhaustive value space: any subset a caller narrows to renders the same
#: way, which is the point of rendering rather than mapping.
MODE_NONE = "none"
MODE_LEXICAL = "lexical"
MODE_LEXICAL_VECTOR = "lexical+vector"
MODE_LEXICAL_GRAPH = "lexical+graph"
MODE_FULL = "lexical+vector+graph"


class UnsupportedPlatform(RuntimeError):
    """Raised when this platform offers no file-locking primitive.

    Raised by the writer, not at import: a read surface has no lock to take
    and must keep working.
    """


class NotConfigured(RuntimeError):
    """Raised when a path the caller must supply cannot be derived.

    Installed as a wheel there is no source checkout to resolve against, so
    the workspace root and the repo-relative eval assets have no defensible
    default. Failing here is the point: the previous code fell back to a
    directory inside ``site-packages``, which silently pointed the corpus and
    the database at the install prefix.
    """


class IndexLocked(RuntimeError):
    """Raised when another process already holds the index write lock."""


class SchemaVersionMismatch(RuntimeError):
    """Raised when the on-disk index schema version does not match this build.

    The index is disposable by design (see module docstring), so recovery is
    always available; this only guards against a code upgrade silently
    reading or writing rows in a shape it no longer understands.
    """


class IndexMissing(RuntimeError):
    """Raised when a read surface opens the index before it has ever been built.

    Only ``brain reindex`` (the writer, holding the single-writer lock, see
    :func:`write_lock`) is allowed to create the index file and run its
    schema DDL. Every read surface (``brain search``, ``brain get``,
    ``brain recent``, ``brain status``, and the MCP server) must fail here
    instead of silently minting an empty index on a fresh checkout, which
    would then report zero documents forever with no signal that nothing was
    ever indexed.
    """


def chunk_ref(doc_path: str, heading_path: str, ordinal: int) -> str:
    """Return the stable, reindex-surviving identity for one chunk.

    Derived from *position*, not content: the document's workspace-relative
    path, the heading path within it, and an ordinal disambiguating repeated
    identical heading paths in the same document. Deliberately not a hash of
    the body, so editing a section's prose keeps its reference resolvable and
    a later read returns the current text of the section that was asked for.
    What must not survive is a reference to a section that no longer exists,
    and that is exactly what this kills: the ref stops resolving and the
    caller gets a clean miss.

    ``chunks.id`` cannot serve this purpose and must never be handed out as a
    durable handle. It is a rowid alias, so a changed document's chunks are
    deleted and reinserted under fresh ids, and SQLite reuses freed ids on
    later inserts. A held id therefore does not fail when it goes stale; it
    silently resolves to unrelated content (issue #23).

    The ordinal counts occurrences of the *same* heading path rather than the
    chunk's position in the document, so inserting a new section does not
    renumber every chunk below it and invalidate their refs.

    The ``c`` prefix guarantees the value never parses as an integer, so a
    surface that previously accepted a rowid cannot silently keep accepting
    one.
    """
    payload = b"\x00".join(
        (doc_path.encode("utf-8"), heading_path.encode("utf-8"), str(ordinal).encode("ascii"))
    )
    return f"{CHUNK_REF_PREFIX}{hashlib.sha256(payload).hexdigest()[:CHUNK_REF_HEX]}"


PROJECT_NAME = "corpusdex"


def _declares_this_project(pyproject: Path) -> bool:
    """Return True when ``pyproject`` is THIS project's manifest.

    Matching on the file's existence alone was wrong in both directions: a
    wheel install has no manifest above the package and fell through to a
    positional guess, while a copy vendored inside another project would have
    claimed that project's root. Only the declared name settles it.
    """
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    project = data.get("project")
    # `project = "x"` is valid TOML and would make .get() raise AttributeError
    # here, which resolves paths for EVERY entry point -- so an unrelated
    # manifest in any ancestor directory could kill the whole tool.
    if not isinstance(project, dict):
        return False
    return project.get("name") == PROJECT_NAME


def source_checkout_root() -> Path | None:
    """Return this package's own source checkout root, or None when installed.

    Resolved from this file upwards so nothing depends on an absolute path.
    None is a real answer, not a failure: installed as a wheel there IS no
    checkout, and every caller that needs one has to say so rather than
    receive a path inside ``site-packages``.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and _declares_this_project(pyproject):
            return candidate
    return None


def repo_root() -> Path:
    """Return this package's own source checkout root.

    Only for repo-relative DEVELOPMENT assets (the eval query set), which no
    wheel ships. Derived state goes to :func:`state_dir` instead, which works
    either way.
    """
    root = source_checkout_root()
    if root is None:
        raise NotConfigured(
            "no source checkout around "
            f"{Path(__file__).resolve().parent}; this command needs one"
        )
    return root


def _user_state_dir() -> Path:
    """Return the per-user directory for application state on this platform."""
    if sys.platform == "win32":  # pragma: no cover - Windows only
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    base = os.environ.get("XDG_STATE_HOME")
    return Path(base) if base else Path.home() / ".local" / "state"


def state_dir() -> Path:
    """Return the directory holding derived state, wherever we were installed.

    In a source checkout this is the gitignored ``var/``, which keeps the
    developer experience unchanged. Installed as a wheel it is a per-user
    state directory: ``site-packages`` is the wrong place to write a 23 MB
    database and is routinely read-only.
    """
    override = env("BRAIN_STATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    root = source_checkout_root()
    if root is not None:
        return root / "var"
    return _user_state_dir() / PROJECT_NAME


def workspace_root() -> Path:
    """Return the workspace root that holds the repo checkouts.

    Raises rather than guessing when installed: the corpus root decides what
    this tool will READ, so inferring it from where pip happened to put the
    package is how an install ends up indexing an unrelated tree.
    """
    override = env("BRAIN_WORKSPACE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    root = source_checkout_root()
    if root is None:
        raise NotConfigured(
            "no workspace root: set BRAIN_WORKSPACE_ROOT to the directory "
            "holding the repo checkouts to index"
        )
    return root.parent


def var_dir() -> Path:
    """Deprecated alias for :func:`state_dir`."""
    return state_dir()


def default_db_path() -> Path:
    override = env("BRAIN_DB")
    if override:
        return Path(override).expanduser().resolve()
    return state_dir() / "index.db"


def _lock_exclusive(fd: int) -> None:
    """Take an exclusive, non-blocking lock on ``fd``, raising OSError if held.

    ``msvcrt.locking`` locks a byte range starting at the current file
    position, so the offset is reset first; locking one byte past EOF is the
    documented idiom and needs no content in the file. ``LK_NBLCK`` fails
    immediately, matching ``LOCK_NB`` rather than the retrying ``LK_LOCK``.
    """
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if msvcrt is None:  # pragma: no cover - neither primitive on this platform
        raise UnsupportedPlatform(
            f"no file-locking primitive on {sys.platform!r}: the index needs "
            "either fcntl or msvcrt to enforce a single writer"
        )
    os.lseek(fd, 0, os.SEEK_SET)  # pragma: no cover - Windows only
    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # pragma: no cover - Windows only


def _unlock(fd: int) -> None:
    """Release the lock taken by :func:`_lock_exclusive`."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return
    if msvcrt is None:  # pragma: no cover - never locked, nothing to release
        return
    os.lseek(fd, 0, os.SEEK_SET)  # pragma: no cover - Windows only
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # pragma: no cover - Windows only


def lock_path_for(db_path: Path) -> Path:
    return db_path.parent / "index.lock"


@contextlib.contextmanager
def write_lock(db_path: Path) -> Iterator[None]:
    """Hold the single-writer lock for ``db_path``.

    Non-blocking on purpose: a CLI should say "a reindex is already running"
    instead of hanging behind another process.
    """
    path = lock_path_for(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            _lock_exclusive(fd)
        except OSError as exc:
            raise IndexLocked(f"another writer holds {path}") from exc
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (creating on demand) the index database."""
    path = Path(db_path) if db_path is not None else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def load_vec(conn: sqlite3.Connection) -> bool:
    """Try to load sqlite-vec into ``conn``.

    Returns False (never raises) when the interpreter refuses extension loading
    or the extension is missing, so callers degrade to lexical-only search.
    """
    if not hasattr(conn, "enable_load_extension"):
        return False
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.NotSupportedError, sqlite3.OperationalError):
        return False
    try:
        sqlite_vec.load(conn)
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return False
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.enable_load_extension(False)
    return True


def has_vec_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
    ).fetchone()
    return row is not None


_CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    repo         TEXT NOT NULL,
    path         TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    mtime        REAL NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_repo_idx ON documents(repo);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    ref           TEXT NOT NULL,
    doc_id        INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading_path  TEXT NOT NULL,
    body          TEXT NOT NULL,
    decided_on    TEXT,
    superseded_by TEXT,
    tags          TEXT
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id);

-- ``ref`` is the only handle a caller is given (see :func:`chunk_ref`), so it
-- has to resolve to at most one row. UNIQUE makes a derivation collision a
-- loud insert failure during reindex rather than a silent ambiguity at read
-- time, where the reader would get whichever row SQLite happened to return.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_ref_idx ON chunks(ref);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body,
    heading_path,
    content = 'chunks',
    content_rowid = 'id',
    tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, body, heading_path)
    VALUES (new.id, new.body, new.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, body, heading_path)
    VALUES ('delete', old.id, old.body, old.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, body, heading_path)
    VALUES ('delete', old.id, old.body, old.heading_path);
    INSERT INTO chunks_fts(rowid, body, heading_path)
    VALUES (new.id, new.body, new.heading_path);
END;

CREATE TABLE IF NOT EXISTS doc_link_targets (
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target   TEXT NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY (doc_id, target, relation)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS doc_links (
    src_doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    dst_doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    relation   TEXT NOT NULL,
    PRIMARY KEY (src_doc_id, dst_doc_id, relation)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS doc_links_dst_idx ON doc_links(dst_doc_id, relation);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

def vec_schema(dim: int) -> str:
    """Return the ``vec_chunks`` DDL for an index of width ``dim``."""
    if dim <= 0:
        raise ValueError(f"embedding dimension must be positive, got {dim}")
    return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""


_VEC_DIM_RE = re.compile(r"FLOAT\s*\[\s*(\d+)\s*\]", re.IGNORECASE)


def vec_table_dim(conn: sqlite3.Connection) -> int | None:
    """Return the width ``vec_chunks`` was declared with, or None if absent.

    Read back from the stored DDL rather than from a constant, because the
    table is what actually rejects a mismatched vector. A constant can drift
    from an index built by an earlier configuration; the declaration cannot.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
    ).fetchone()
    if row is None or not row["sql"]:
        return None
    match = _VEC_DIM_RE.search(row["sql"])
    return int(match.group(1)) if match else None


def recreate_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Rebuild ``vec_chunks`` at width ``dim``, discarding every vector.

    A width change cannot be handled by deleting rows the way a model change
    can: the width lives in the table declaration, so the table itself has to
    go. Every vector is invalid at the new width anyway, so nothing is lost
    that a backfill will not rebuild.
    """
    # Two separate reasons this has to be a transaction, both of which cost
    # the whole vector table when they are not:
    #
    # execute(), never executescript(): executescript issues an implicit
    # COMMIT before running, which both durably lands whatever the caller had
    # in flight and puts the DROP outside the caller's transaction.
    #
    # And an explicit BEGIN, because sqlite3's legacy autocommit mode opens a
    # transaction only ahead of DML. DDL alone runs in autocommit, so without
    # this the DROP is already committed by the time `with conn:` sees the
    # exception it is supposed to undo, leaving an empty table at the new
    # width while the run that was to refill it has aborted.
    if not conn.in_transaction:
        conn.execute("BEGIN")
    conn.execute("DROP TABLE IF EXISTS vec_chunks")
    conn.execute(vec_schema(dim))


def init_schema(
    conn: sqlite3.Connection, *, vec: bool = False, embed_dim: int = EMBED_DIM
) -> None:
    """Create the schema if absent. Safe to call on every open."""
    with conn:
        conn.executescript(_CORE_SCHEMA)
        if vec:
            conn.executescript(vec_schema(embed_dim))
        if get_meta(conn, META_SCHEMA_VERSION) is None:
            set_meta(conn, META_SCHEMA_VERSION, str(SCHEMA_VERSION))


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def stale_embed_model(stored_model: str | None, active_model: str | None) -> str | None:
    """Return ``stored_model`` when it disagrees with ``active_model``, else ``None``.

    A pure comparison rather than a query, because its two callers read the
    stored name at different points: search holds a connection, status has
    already closed one by the time it knows the active model.

    ``None`` means "nothing to report", which deliberately covers three
    different situations because all three carry the same instruction, do
    nothing: the names agree; the index predates the ``embed_model`` row and
    has no stored name; or the caller cannot name its active model. Absence on
    either side is unknown, not equal, and treating unknown as a mismatch
    would degrade every search against an older index for no reason.

    This is the one place the comparison lives. Two vector sets produced by
    different models occupy unrelated spaces, so the same predicate has to
    decide both whether a search may use its vector channel and whether
    ``brain status`` calls the configuration stale. A second copy would be
    free to drift into answering those two questions differently, and the
    surface that answered "healthy" would be the one people believe.
    """
    if not stored_model or not active_model:
        return None
    return None if stored_model == active_model else stored_model


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def stored_meta(db_path: Path, key: str) -> str | None:
    """Read one ``meta`` value from an existing index without modifying it.

    Opened read-only for the reason given in :func:`stored_schema_version`,
    which is the same reader specialised to one key.
    """
    try:
        uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    except (sqlite3.OperationalError, ValueError):
        return None
    try:
        conn.row_factory = sqlite3.Row
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()
        if present is None:
            return None
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


def stored_schema_version(db_path: Path) -> str | None:
    """Read an existing index's stored schema version without modifying it.

    Opened read-only on purpose. The ordinary :func:`connect` sets
    ``journal_mode = WAL``, which writes a header, so it would turn a 0-byte
    file into a valid empty database just by being asked what version it is.

    Returns None when the file is not a usable index: empty, not a SQLite
    database, or missing the ``meta`` row that every index the writer stamps
    has. That is deliberately indistinguishable from corruption, because the
    only correct response to any of them is the same rebuild.
    """
    try:
        uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    except (sqlite3.OperationalError, ValueError):
        return None
    try:
        conn.row_factory = sqlite3.Row
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()
        if present is None:
            return None
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (META_SCHEMA_VERSION,)
        ).fetchone()
        return None if row is None else row["value"]
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


def discard_index(db_path: Path) -> None:
    """Delete an index file and its WAL sidecars.

    Only used on a scratch build path (see :func:`swap_index`), never on a
    live index: destroying the live file in place is what left an empty
    index behind when a rebuild failed partway.
    """
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        with contextlib.suppress(FileNotFoundError):
            candidate.unlink()


def swap_index(src: Path, dst: Path) -> None:
    """Atomically move a freshly built index over the live one.

    ``src`` must already be closed, so SQLite has checkpointed and removed
    its own WAL. The destination's sidecars are removed first: they describe
    the file being replaced, and leaving them beside the new one would let
    SQLite apply a foreign write-ahead log to it.
    """
    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(FileNotFoundError):
            Path(str(dst) + suffix).unlink()
    os.replace(src, dst)
    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(FileNotFoundError):
            Path(str(src) + suffix).unlink()


def open_index(
    db_path: Path | None = None, *, create: bool = False
) -> tuple[sqlite3.Connection, bool]:
    """Open the index, load sqlite-vec when possible, and ensure the schema.

    Returns the connection and whether vector operations are usable on it.

    ``create`` defaults to False: a missing index file raises
    :class:`IndexMissing` rather than being silently created empty. Only the
    writer (``brain reindex``, called while holding :func:`write_lock`)
    should pass ``create=True``; every read surface (CLI search/get/recent/
    status, the MCP server) must use the default so a fresh checkout with no
    index yet reports a clean "no index" error instead of minting one.

    A read surface (``create=False``) never executes DDL. It validates the
    stored version through :func:`stored_schema_version` first and raises
    rather than touching the file, because a reader that "just" ran
    ``CREATE TABLE IF NOT EXISTS`` was silently upgrading a mismatched index
    it had already decided it could not read, lock-free and outside the
    single-writer lock. An unreadable file (empty, not a database, or
    unstamped) raises :class:`IndexMissing`; a stamped but different version
    raises :class:`SchemaVersionMismatch`. Neither is fixable by a reader,
    since rebuilding is a write.

    The writer (``create=True``, holding :func:`write_lock`) creates the
    schema when the file is absent, and otherwise refuses a version it does
    not recognise instead of destroying it in place. Rebuilding a stale index
    is :func:`corpusdex.indexer.reindex`'s job and goes through a scratch
    file and :func:`swap_index`, so a half-finished rebuild can never be
    served as if it were the real index.
    """
    resolved_path = Path(db_path) if db_path is not None else default_db_path()
    if not resolved_path.is_file():
        if not create:
            raise IndexMissing(f"no index at {resolved_path}; run `brain reindex` to build it")
        conn = connect(resolved_path)
        vec = load_vec(conn)
        init_schema(conn, vec=vec)
        return conn, vec

    stored_version = stored_schema_version(resolved_path)
    if stored_version is None:
        raise IndexMissing(
            f"the index at {resolved_path} is empty or unreadable; "
            "run `brain reindex` to rebuild it"
        )
    try:
        stored = int(stored_version)
    except ValueError:
        raise SchemaVersionMismatch(
            f"the index at {resolved_path} has an unreadable schema version "
            f"{stored_version!r}; run `brain reindex` to rebuild it"
        ) from None
    if stored != SCHEMA_VERSION:
        if stored > SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"index schema version {stored} is newer than this build's version "
                f"{SCHEMA_VERSION}: the code reading the index is older than the "
                "index itself. The usual cause is a long-running process (the MCP "
                "server, an editor plugin) that imported corpusdex before the "
                "checkout was upgraded and still holds the old version in memory; "
                "restart it and it will read this index correctly. If instead the "
                "checkout really is behind, update it. Only delete the index file "
                "and run `brain reindex` if neither applies."
            )
        raise SchemaVersionMismatch(
            f"index schema version {stored} does not match this build's "
            f"version {SCHEMA_VERSION}; run `brain reindex` to rebuild it"
        )

    conn = connect(resolved_path)
    vec = load_vec(conn)
    if create:
        init_schema(conn, vec=vec)
    return conn, vec
