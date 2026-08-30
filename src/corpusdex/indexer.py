"""Incremental corpus discovery and indexing.

Corpus scope has exactly one source, and which one is a configuration
choice. ``BRAIN_CORPUS_ROOTS`` names directories to index directly and
replaces everything below; otherwise scope is driven by the workspace repo
registry (see :func:`registry_filenames`, one
``name<TAB>origin-url<TAB>branch`` row per registered repo): each registered
repo's ``docs/**/*.md`` (at any depth inside that
repo's own directory tree, e.g. ``audit-api/docs/x.md``, not only a top-level
``docs/``), top-level ``AGENTS.md`` / ``CLAUDE.md`` / ``README.md``, and
``tasks/*.md``; plus the workspace root's own ``AGENTS.md`` / ``CLAUDE.md`` /
``README.md``, its own direct ``docs/**/*.md``, and its own ``tasks/*.md``;
plus workspace-level ``.claude/skills/*/SKILL.md``; plus the knowledge repo's own
Markdown in full (see :func:`knowledge_repo`: one repo in a workspace is the
brain itself, so all of it is corpus, not just its ``docs`` subtree). Any path
matching ``BRAIN_EXCLUDE`` is dropped in either mode. If the registry file is
missing and no explicit roots are configured, :func:`discover_corpus` raises
:class:`RegistryMissing` rather than silently falling back to scanning
every sibling directory in the workspace (which previously pulled in stale
checkout copies and unrelated personal directories into the corpus). The
workspace root's own docs match is deliberately NOT a full recursive walk of
the workspace root: unlike a single registered repo's own directory (which
``os.walk`` can never escape into a sibling of), a full-tree walk rooted at
the workspace root would cross into every sibling directory including
unregistered worktree checkouts, reintroducing exactly the whole-workspace
scan the registry exists to prevent.

Reindexing is incremental: a document whose (mtime, size) is unchanged since
the last index is skipped without reading it; a document whose bytes hash
unchanged (mtime drifted but content did not, e.g. after a checkout) only
refreshes its stored mtime. Only genuinely new or changed content is
re-chunked; embedding is handled separately by a backfill pass that embeds
every chunk currently lacking a vector, so an index built while the embedding
backend was down is fully caught up by the next ordinary reindex, not only by
``--full``. A document no longer present in the corpus is removed along with
its chunks and vectors. Writers hold the single-writer lock in
``var/index.lock`` (see :func:`corpusdex.db.write_lock`) for the whole run,
acquired before any schema initialization or write.

Each indexed document also contributes its raw link targets (see
:mod:`corpusdex.links`); those are resolved into the document graph by
:func:`rebuild_link_graph` once every insert and delete in the run has landed.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import posixpath
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import db
from .chunker import chunk_markdown, split_frontmatter
from .config import env, env_list
from .embedder import EmbeddingUnavailable, OllamaEmbedder, default_embedder
from .links import (
    EXTRACTOR_VERSION,
    RELATION_SUPERSEDED_BY,
    extract_links,
    normalize_target,
    resolution_keys,
    resolve_relative_path,
)

#: A link target that is exactly four digits, i.e. a decision record cited by
#: number. Checked with ``fullmatch`` so it can never claim a filename that
#: merely begins with digits.
_ADR_NUMBER_RE = re.compile(r"\d{4}")

#: A decision record's filename. Deliberately the same shape
#: ``tests/test_decision_records.py`` enforces on the real ``decisions/``
#: directory, so "what is a decision record" has one definition rather than a
#: hygiene rule and a resolver rule that can drift apart. The slug must be
#: non-empty: ``0006-.md`` satisfies neither.
_DECISION_FILENAME_RE = re.compile(r"^(\d{4})-[^/]+\.md$", re.IGNORECASE)

#: Decision records are numbered per directory, so the directory is part of
#: the identity. Keying on the filename alone would let any four-digit-prefixed
#: document anywhere in the corpus answer a citation of "decision 0006",
#: including a dated gap ledger or another repo's own numbering.
_DECISION_DIR = "decisions"


#: Directory name marking an append-only work log. A ledger cites a decision
#: because that decision was WORKED ON in some session, which is a statement
#: about scheduling, not about subject, and the two documents it co-locates
#: usually have nothing to do with each other. Left in, they also grow without
#: bound: every session appends, so the fan-out only ever increases and the
#: edge count stops being reproducible across rebuilds.
_LOG_DIR = "tasks"


def _is_work_log(path: str) -> bool:
    """Is this document an append-only work log rather than an argument?

    Matched on the containing directory rather than on fan-out. A cap was
    measured and rejected: at 5 it also truncates the curated repo context
    cards (one repo's card cites seven decisions, another's five), and the
    paraphrase group's MRR contribution collapses from +0.151
    to +0.068 because those card edges are what it retrieves through. A cap
    high enough to spare the cards works only because the threshold currently
    falls between the two populations by luck, and would start truncating a
    card the moment one grew. The distinction is document KIND, so that is
    what the rule tests.
    """
    normalised = path.replace("\\", "/")
    return normalised.startswith(f"{_LOG_DIR}/") or f"/{_LOG_DIR}/" in normalised


def _decision_number(path: str) -> str | None:
    """Return the decision number a document carries, or None if it is not one.

    A decision record is ``<...>/decisions/NNNN-slug.md``. Both halves are
    required: the directory establishes whose numbering it is, and the
    ``NNNN-`` prefix establishes that the leading digits are a record number
    rather than a date or a version.
    """
    head, _, name = path.replace("\\", "/").rpartition("/")
    if posixpath.basename(head) != _DECISION_DIR:
        return None
    match = _DECISION_FILENAME_RE.match(name)
    return match.group(1) if match is not None else None


SKIP_DIR_NAMES = frozenset(
    {
        ".worktrees",
        # Per the workspace AGENTS.md worktree convention, Claude-managed
        # ephemeral worktrees live at ``<repo>/.claude/worktrees/<agent-id>``:
        # a full duplicate checkout nested inside the repo itself, not just a
        # workspace-root sibling. Left unskipped, a registered repo's own
        # deep docs walk (see _repo_root_patterns) would index every such
        # in-flight agent worktree's copy of the same docs alongside the
        # real ones.
        "worktrees",
        "node_modules",
        ".git",
        "var",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "__pycache__",
        "dist",
        "build",
    }
)
WORKSPACE_REPO = "workspace"
SKILLS_REPO = ".claude"

#: Registry filenames, most current first. The generic name is what a new
#: workspace should create; the second is the name this engine shipped with
#: before the split and is still what the workspace it grew up in has on disk.
#: Resolution is by existence rather than by a flag day, for the same reason
#: :data:`corpusdex.config.LEGACY_ENV_ALIASES` exists: a rename that stops
#: an existing workspace from indexing the moment it lands gets reverted
#: rather than finished. ``BRAIN_REGISTRY_FILE`` overrides both.
REGISTRY_FILENAMES = (".corpus-repos.tsv", ".harnx-repos.tsv")

#: The name a caller writing a fresh registry should use, and what the
#: not-found error names.
REGISTRY_FILENAME = REGISTRY_FILENAMES[0]


def registry_filenames() -> tuple[str, ...]:
    """Return the registry filenames to look for, in order.

    An explicit ``BRAIN_REGISTRY_FILE`` replaces the search rather than
    extending it: a caller naming a file is stating where scope comes from,
    and quietly falling back to a differently-named file found next to it
    would let a stale registry decide the corpus.
    """
    configured = env("BRAIN_REGISTRY_FILE")
    if configured:
        return (configured,)
    return REGISTRY_FILENAMES


def knowledge_repo() -> str | None:
    """Return the repo whose whole tree is curated knowledge, if any.

    One repo in a workspace is the brain itself: every Markdown file in it is
    corpus rather than only its ``docs`` subtree, and its ``decisions/``,
    ``context/``, ``architecture/`` and ``gaps/`` directories carry meaning
    that the same directory names elsewhere do not.

    That was pinned to this workspace's own repo name. It now defaults to the
    directory the engine is running from when that is a source checkout, which
    reproduces the previous value here and, unlike a constant, survives the
    package being renamed. Installed from a wheel there is no such directory,
    so the default is None and no repo gets whole-tree treatment unless
    ``BRAIN_KNOWLEDGE_REPO`` names one.
    """
    configured = env("BRAIN_KNOWLEDGE_REPO")
    if configured:
        return configured
    checkout = db.source_checkout_root()
    return checkout.name if checkout is not None else None


def _exclusion_patterns() -> list[str]:
    """Return the configured corpus exclusion globs."""
    return env_list("BRAIN_EXCLUDE")


def _is_excluded(rel_path: str, patterns: list[str]) -> bool:
    """Return whether ``rel_path`` matches any exclusion pattern.

    A pattern containing no ``/`` matches any single path SEGMENT, so
    ``BRAIN_EXCLUDE=vendor`` excludes ``a/vendor/b.md`` the way a reader
    expects; ``fnmatch`` alone would not, because ``*`` does not cross
    separators here by convention and a bare name would only ever match a
    top-level entry. A pattern containing ``/`` is matched against the whole
    relative path instead, which is how a caller narrows to one place
    (``some-repo/docs/*``) rather than to a name.
    """
    if not patterns:
        return False
    segments = rel_path.split("/")
    for pattern in patterns:
        if "/" in pattern:
            if fnmatch.fnmatchcase(rel_path, pattern):
                return True
        elif any(fnmatch.fnmatchcase(segment, pattern) for segment in segments):
            return True
    return False


def configured_corpus_roots() -> list[Path]:
    """Return explicitly configured corpus roots, or an empty list.

    ``BRAIN_CORPUS_ROOTS`` is the answer to "I am not in that workspace": a
    list of directories to index directly, each contributing documents under a
    repo named for its own directory. Setting it replaces registry mode
    entirely, so scope stays a single closed question with one answer rather
    than a union of two mechanisms that have to agree.
    """
    return [Path(raw).expanduser() for raw in env_list("BRAIN_CORPUS_ROOTS")]


class RegistryMissing(RuntimeError):
    """Raised when the workspace repo registry file cannot be found.

    Corpus scope is driven entirely by the registry so a relocated or
    incomplete workspace fails loudly instead of silently scanning every
    sibling directory.
    """


class RegistryInvalid(RuntimeError):
    """Raised when a row in the workspace repo registry is unsafe to use.

    A registered repo name is joined directly onto ``workspace_root`` (see
    :func:`discover_corpus`) to locate that repo's directory. A name
    containing a path separator or ``..`` can walk outside the workspace
    (silently pulling arbitrary filesystem content into the corpus), and an
    absolute name replaces the join outright (``Path`` semantics: joining an
    absolute path discards the left-hand side), which previously surfaced as
    an unhandled ``ValueError`` deep inside path-relativization rather than a
    clear error naming the offending row.
    """


@dataclass(frozen=True)
class CorpusDoc:
    """One Markdown file discovered in the workspace corpus."""

    repo: str
    rel_path: str
    abs_path: Path


@dataclass
class ReindexStats:
    """Summary of one :func:`reindex` run."""

    docs_seen: int = 0
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks_written: int = 0
    embedded_chunks: int = 0
    vector_total: int = 0
    vector_covered: int = 0
    embedding_available: bool = False
    link_edges: int = 0
    link_targets_unresolved: int = 0
    #: Targets that were dropped on purpose rather than for want of a
    #: document: an ambiguous name, or a document linking to itself. Split out
    #: from ``link_targets_unresolved`` so the reported number is diagnostic.
    #: Lumped together, a healthy corpus still shows a large "unresolved"
    #: figure and there is no way to tell whether it is a content gap worth
    #: fixing or the resolver correctly refusing to guess.
    link_targets_unlinkable: int = 0
    #: Decision citations declined because their source is an append-only work
    #: log (see :func:`_is_work_log`). Counted apart from the two above rather
    #: than folded into either: these were declined on the SOURCE document's
    #: kind, before resolution was attempted, so reporting them as "names no
    #: document" or as "ambiguous" would assert a resolution result that was
    #: never computed. Some of them may well name nothing; the point is that
    #: this counter does not claim to know, and dropping them silently is the
    #: failure mode it exists to prevent.
    link_targets_from_work_logs: int = 0
    #: Documents whose ``superseded_by`` frontmatter resolved to no edge,
    #: by path. A list rather than a count because the point is to name them:
    #: the ranking penalty is derived from the resolved edge, so an entry here
    #: is a record that claims it was replaced and is ranked as though it was
    #: not, with no successor shown and nothing else to reveal it.
    superseded_by_unresolved: tuple[str, ...] = ()
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def touched(self) -> int:
        """Documents whose stored rows changed (added, changed, or removed)."""
        return self.added + self.changed + self.removed

    @property
    def fully_embedded(self) -> bool:
        """True when every known chunk has a vector (real, not just live-backend, coverage)."""
        return self.embedding_available and self.vector_covered >= self.vector_total


def _validate_repo_name(name: str, row_text: str, registry_name: str) -> None:
    """Reject a registry row whose repo name could resolve outside the workspace.

    ``workspace_root / name`` is how :func:`_discover_registered_workspace` locates a
    registered repo's directory; ``name`` must be a single, relative path
    segment for that join to always stay inside ``workspace_root``.
    """
    problems = []
    if "/" in name:
        problems.append("contains '/'")
    if "\\" in name:
        problems.append("contains '\\\\'")
    if ".." in name:
        problems.append("contains '..'")
    if Path(name).is_absolute():
        problems.append("is an absolute path")
    if problems:
        raise RegistryInvalid(
            f"invalid repo name {name!r} in {registry_name} row {row_text!r}: "
            f"{'; '.join(problems)} (must be a single path segment directly under "
            "the workspace root)"
        )


def _read_registry(workspace_root: Path) -> list[str]:
    """Return the registered repo names from the workspace repo registry.

    Raises :class:`RegistryMissing` if the file is absent so callers fail
    loudly rather than silently rescanning the whole workspace, and
    :class:`RegistryInvalid` if any row's repo name could resolve outside the
    workspace once joined onto ``workspace_root``.
    """
    candidates = registry_filenames()
    registry_path = next(
        (workspace_root / name for name in candidates if (workspace_root / name).is_file()),
        None,
    )
    if registry_path is None:
        looked_for = " or ".join(candidates)
        raise RegistryMissing(
            f"no repo registry ({looked_for}) found at {workspace_root}; corpus "
            "scope comes from registered repos only and this refuses to silently "
            "rescan every sibling directory. Point BRAIN_WORKSPACE_ROOT at a "
            f"workspace that has a {candidates[0]}, create one there, or set "
            "BRAIN_CORPUS_ROOTS to name the directories to index directly."
        )
    names: list[str] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        name = stripped.split("\t", 1)[0].strip()
        if name:
            _validate_repo_name(name, stripped, registry_path.name)
            names.append(name)
    return names


def _walk_markdown(base: Path) -> Iterator[Path]:
    """Yield every ``*.md`` under ``base``, pruning :data:`SKIP_DIR_NAMES`."""
    if not base.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIR_NAMES)
        for name in sorted(filenames):
            if name.endswith(".md"):
                yield Path(dirpath) / name


def _repo_root_patterns(root: Path, *, deep_docs: bool) -> Iterator[Path]:
    """Yield the repo-root files, its ``docs/**/*.md``, and ``tasks/*.md``.

    With ``deep_docs=True``, ``docs`` is matched at any depth under ``root``
    (e.g. ``audit-api/docs/x.md`` inside a repo that nests its Python package
    one level down), not only a top-level ``docs/`` directory, while still
    honouring :data:`SKIP_DIR_NAMES`. This full-tree walk is only safe when
    ``root`` is a single registered repo's own directory: :func:`os.walk`
    cannot escape into sibling directories, so it can never reach an
    unregistered checkout next to it.

    With ``deep_docs=False``, only a direct ``root/docs/`` subtree is walked.
    This is what the workspace root itself must use: a full-tree walk rooted
    at the workspace root would cross into every sibling directory (including
    unregistered stale checkouts and other worktrees), silently reintroducing
    the whole-workspace scan the repo registry exists to prevent.
    """
    for name in ("AGENTS.md", "CLAUDE.md", "README.md"):
        candidate = root / name
        if candidate.is_file():
            yield candidate
    if deep_docs:
        for md_path in _walk_markdown(root):
            parents = md_path.relative_to(root).parts[:-1]
            if "docs" in parents:
                yield md_path
    else:
        docs_dir = root / "docs"
        yield from _walk_markdown(docs_dir)
    tasks_dir = root / "tasks"
    if tasks_dir.is_dir():
        for candidate in sorted(tasks_dir.glob("*.md")):
            if candidate.is_file():
                yield candidate


def discover_corpus(workspace_root: Path | None = None) -> list[CorpusDoc]:
    """Enumerate the current on-disk corpus.

    Two modes, and exactly one is active. If ``BRAIN_CORPUS_ROOTS`` names any
    directories, those ARE the corpus and ``workspace_root`` is not consulted;
    otherwise scope comes from the workspace repo registry and a missing
    registry raises :class:`RegistryMissing` rather than falling back to
    scanning every sibling directory.

    Deterministic and side-effect free; does not touch the index.
    """
    excludes = _exclusion_patterns()
    roots = configured_corpus_roots()
    if roots:
        return _discover_explicit_roots(roots, excludes)
    if workspace_root is None:
        raise RegistryMissing(
            "no workspace root to scan and BRAIN_CORPUS_ROOTS is unset; set one "
            "of them so corpus scope has exactly one source"
        )
    return _discover_registered_workspace(Path(workspace_root).resolve(), excludes)


def _discover_explicit_roots(roots: list[Path], excludes: list[str]) -> list[CorpusDoc]:
    """Index each configured root as its own repo, named for its directory.

    A root is treated exactly as a registered repo is: its own top-level
    ``AGENTS.md`` / ``CLAUDE.md`` / ``README.md``, its ``docs`` subtrees at any
    depth, and its ``tasks/*.md``. The whole-tree treatment stays reserved for
    the knowledge repo, so pointing this at a source tree does not pull in
    every stray Markdown file it happens to contain.

    Paths are relative to each root, prefixed by that root's name, because
    configured roots need share no common parent -- deriving the prefix from
    one would produce ``../../elsewhere/x.md`` for the others.
    """
    found: dict[Path, tuple[str, str]] = {}
    knowledge = knowledge_repo()
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.is_dir():
            continue
        repo = root.name
        paths = (
            _walk_markdown(root) if repo == knowledge else _repo_root_patterns(root, deep_docs=True)
        )
        for path in paths:
            rel = posixpath.join(repo, path.relative_to(root).as_posix())
            if _is_excluded(rel, excludes):
                continue
            found[path] = (repo, rel)
    return _sorted_docs(found)


def _discover_registered_workspace(workspace_root: Path, excludes: list[str]) -> list[CorpusDoc]:
    """Scope the corpus to the repos named in the workspace registry."""
    registered_repos = _read_registry(workspace_root)
    knowledge = knowledge_repo()
    found: dict[Path, tuple[str, str]] = {}

    def record(path: Path, repo: str) -> None:
        rel = path.relative_to(workspace_root).as_posix()
        if _is_excluded(rel, excludes):
            return
        found[path] = (repo, rel)

    for path in _repo_root_patterns(workspace_root, deep_docs=False):
        record(path, WORKSPACE_REPO)

    for repo in registered_repos:
        child = workspace_root / repo
        if not child.is_dir() or repo in SKIP_DIR_NAMES:
            continue
        for path in _repo_root_patterns(child, deep_docs=True):
            record(path, repo)
        if repo == knowledge:
            for path in _walk_markdown(child):
                record(path, repo)

    skills_dir = workspace_root / ".claude" / "skills"
    if skills_dir.is_dir():
        for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
            if skill_file.is_file():
                record(skill_file, SKILLS_REPO)

    return _sorted_docs(found)


def _sorted_docs(found: dict[Path, tuple[str, str]]) -> list[CorpusDoc]:
    docs = [
        CorpusDoc(repo=repo, rel_path=rel_path, abs_path=path)
        for path, (repo, rel_path) in found.items()
    ]
    docs.sort(key=lambda d: d.rel_path)
    return docs


def classify_doc_type(repo: str, rel_path: str, knowledge: str | None = None) -> str:
    """Classify a corpus document for the ``documents.doc_type`` column.

    ``rel_path`` is workspace-relative (see :class:`CorpusDoc`), so it is
    repo-prefixed for every repo except :data:`WORKSPACE_REPO`; the prefix is
    stripped before matching directory-based rules like ``decisions/`` so
    those rules see the path as it looks from inside the repo.
    """
    if knowledge is None:
        knowledge = knowledge_repo()
    prefix = f"{repo}/"
    within_repo = rel_path[len(prefix) :] if rel_path.startswith(prefix) else rel_path
    parts = within_repo.split("/")
    name = parts[-1]
    parents = parts[:-1]
    if name == "SKILL.md":
        return "skill"
    if repo == knowledge:
        for dir_prefix, doc_type in (
            ("decisions/", "decision"),
            ("context/", "context"),
            ("architecture/", "architecture"),
            ("gaps/", "gap"),
        ):
            if within_repo.startswith(dir_prefix):
                return doc_type
    if name == "AGENTS.md":
        return "agents"
    if name == "CLAUDE.md":
        return "claude"
    if name == "README.md":
        return "readme"
    if "tasks" in parents:
        return "task"
    if "docs" in parents:
        return "doc"
    return "doc"


def _fingerprint(raw: bytes) -> str:
    """A ``size:sha256`` composite fingerprint stored in ``documents.content_hash``.

    Folding size into the stored fingerprint (rather than adding a schema
    column) lets the mtime fast-path also gate on size with no migration: a
    same-mtime, different-size edit changes the comparable prefix even before
    any hash is recomputed. Legacy rows written before this change hold a
    bare hex digest with no ``:`` separator, which never matches a real
    ``size:hash`` value, so they safely fall through to a one-time recompute.
    """
    return f"{len(raw)}:{hashlib.sha256(raw).hexdigest()}"


def _fingerprint_size(fingerprint: str) -> int | None:
    size_str, sep, _digest = fingerprint.partition(":")
    if not sep:
        return None
    try:
        return int(size_str)
    except ValueError:
        return None


def _delete_doc_vectors(conn, vec_ok: bool, doc_id: int) -> None:
    if not vec_ok or not db.has_vec_table(conn):
        return
    rows = conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    if not rows:
        return
    ids = [row["id"] for row in rows]
    placeholders = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", ids)  # noqa: S608


def _reconcile_orphaned_vectors(conn) -> None:
    """Delete any ``vec_chunks`` row whose owning chunk no longer exists.

    Repairs a database wedged by a previous run that deleted chunks while the
    vector extension was unavailable (so their vectors were never cleaned
    up): ``chunks.id`` is a plain ``INTEGER PRIMARY KEY`` and SQLite can
    reuse a freed id on a later insert, which would otherwise collide with
    the stale orphaned vector row's ``PRIMARY KEY`` and abort the whole
    reindex transaction on every subsequent run.
    """
    conn.execute("DELETE FROM vec_chunks WHERE chunk_id NOT IN (SELECT id FROM chunks)")


def _missing_vector_chunk_ids(conn) -> list[int]:
    rows = conn.execute(
        "SELECT chunks.id AS id FROM chunks "
        "LEFT JOIN vec_chunks ON vec_chunks.chunk_id = chunks.id "
        "WHERE vec_chunks.chunk_id IS NULL"
    ).fetchall()
    return [row["id"] for row in rows]


_BACKFILL_BATCH_SIZE = 500


def _backfill_missing_embeddings(conn, embed_fn) -> tuple[int, bool]:
    """Embed every chunk currently lacking a vector.

    Runs regardless of whether this reindex touched those chunks' documents,
    so an index built while the embedding backend was down is caught up by
    the next ordinary (non-``--full``) reindex once the backend recovers.
    Processes ids in batches of :data:`_BACKFILL_BATCH_SIZE` so a large
    backfill (a fresh index, or one recovering after a long Ollama outage)
    never builds a single ``IN (...)`` clause with an unbounded number of
    bound parameters. Returns ``(chunks_embedded, backend_still_available)``.
    """
    missing_ids = _missing_vector_chunk_ids(conn)
    if not missing_ids:
        return 0, True

    import sqlite_vec

    embedded = 0
    for start in range(0, len(missing_ids), _BACKFILL_BATCH_SIZE):
        batch_ids = missing_ids[start : start + _BACKFILL_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch_ids)
        rows = conn.execute(
            f"SELECT id, heading_path, body FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
            batch_ids,
        ).fetchall()
        texts = [f"{row['heading_path']}\n\n{row['body']}" for row in rows]
        try:
            vectors = embed_fn(texts)
        except EmbeddingUnavailable:
            return embedded, False

        for row, vector in zip(rows, vectors, strict=True):
            conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (row["id"], sqlite_vec.serialize_float32(vector)),
            )
            embedded += 1
    return embedded, True


@dataclass(frozen=True)
class LinkGraphResult:
    """What one link-graph rebuild resolved, and what it declined.

    A named result rather than a tuple because the tuple had already grown to
    four positions and every addition silently renumbers every unpacking site.
    The last field is not a count: an unresolved ``superseded_by`` has to be
    reported by NAME, because the whole complaint in issue #21 was that no
    query could say which document was affected.
    """

    edges: int = 0
    unresolved: int = 0
    unlinkable: int = 0
    from_work_logs: int = 0
    #: Paths of documents whose ``superseded_by`` frontmatter named something
    #: that resolved to no edge. Sorted, so the reindex report is stable.
    superseded_by_unresolved: tuple[str, ...] = ()


def rebuild_link_graph(conn) -> LinkGraphResult:
    """Resolve every stored link target to a document id, rebuilding ``doc_links``.

    Rebuilt wholesale rather than patched incrementally: a re-chunked document
    is deleted and reinserted under a new id, so any edge pointing at it would
    otherwise be silently dropped by the cascade and never restored. The graph
    is a few hundred nodes, so a full rebuild costs nothing.

    Resolution is tried most-specific first. A ``./`` or ``../`` target is
    path arithmetic against the linking document's own directory
    (:func:`corpusdex.links.resolve_relative_path`) and admits no
    ambiguity, so it is tried before anything else. A bare four-digit target
    is a decision record cited by number and is looked up in the decision
    records' own numbering, which no other resolution step can serve because
    the number appears nowhere in the filename keys. Otherwise the target is
    matched against the keys in :func:`corpusdex.links.resolution_keys`,
    preferring a match inside the linking document's own repo before a
    workspace-wide one.

    A target that matches nothing, matches ambiguously, or points at its own
    document is dropped: an unresolved reference is not an edge. The two
    reasons are counted separately, because they call for opposite responses.
    Returns a :class:`LinkGraphResult`, where unresolved means no document
    matched (a content gap someone could close), unlinkable means the
    resolver declined to guess (working as intended), and ``from_work_logs``
    counts decision citations dropped because their source is an append-only
    work log.

    A ``superseded_by`` target that produces no edge is additionally recorded
    BY NAME. The ranking penalty for a superseded document is derived from
    the resolved edge (see :data:`corpusdex.search.SUPERSEDED_PENALTY`), so
    a value that resolves to nothing means the document is not penalised and
    displays no successor: the record claims it was replaced and the engine
    behaves as though it never was. That is silent in both directions unless
    the rebuild says so out loud.
    """
    conn.execute("DELETE FROM doc_links")
    doc_rows = conn.execute("SELECT id, repo, path, title FROM documents").fetchall()
    repo_keys: dict[tuple[str, str], set[int]] = {}
    global_keys: dict[str, set[int]] = {}
    doc_repo: dict[int, str] = {}
    doc_path: dict[int, str] = {}
    by_path: dict[str, int] = {}
    #: Decision number -> the records carrying it. A set rather than a single
    #: id on purpose: two records can take one number (it has happened twice,
    #: and merges cleanly because the slugs differ), and the resolver must
    #: decline a citation it cannot attribute rather than pick the first.
    adr_keys: dict[str, set[int]] = {}
    for row in doc_rows:
        doc_repo[row["id"]] = row["repo"]
        doc_path[row["id"]] = row["path"]
        by_path[row["path"].strip().lower()] = row["id"]
        number = _decision_number(row["path"])
        if number is not None:
            adr_keys.setdefault(number, set()).add(row["id"])
        for key in resolution_keys(row["repo"], row["path"], row["title"]):
            repo_keys.setdefault((row["repo"], key), set()).add(row["id"])
            global_keys.setdefault(key, set()).add(row["id"])

    def resolve(src: int, target: str) -> tuple[int | None, bool]:
        """Return ``(doc_id, ambiguous)``; ``doc_id`` is None when nothing matched.

        The same-repo bucket is tried before the workspace-wide one, but that
        ordering is INERT and kept for readability rather than effect: the
        repo bucket is a subset of the global one, so a single same-repo
        candidate is always also the single global candidate, and a global
        bucket with two entries is rejected by the ambiguity rule before the
        order could matter. Swapping the two changes no outcome, confirmed by
        mutation. The documented preference is real as behaviour and is
        delivered by the ambiguity rule; do not "repair" it into an actual
        precedence, which would let a same-repo name win a contest the
        resolver currently declines.
        """
        if _ADR_NUMBER_RE.fullmatch(target):
            candidates = adr_keys.get(target)
            if not candidates:
                return None, False
            if len(candidates) > 1:
                return None, True
            return next(iter(candidates)), False
        relative = resolve_relative_path(doc_path.get(src, ""), target)
        if relative is not None:
            # Unambiguous by construction, so a miss here is a miss: falling
            # through to the filename fallback would answer a question the
            # document did not ask, resolving ../a/README.md by the name
            # README.md alone.
            return by_path.get(relative.lower()), False
        key = normalize_target(target)
        src_repo = doc_repo.get(src, "")
        ambiguous = False
        for candidates in (repo_keys.get((src_repo, key)), global_keys.get(key)):
            if not candidates:
                continue
            if len(candidates) == 1:
                return next(iter(candidates)), False
            ambiguous = True
        return None, ambiguous

    edges: set[tuple[int, int, str]] = set()
    unresolved = 0
    unlinkable = 0
    from_work_logs = 0
    dangling_supersedence: set[str] = set()

    def note_if_supersedence(src: int, relation: str) -> None:
        """Record a ``superseded_by`` that produced no edge, by source path."""
        if relation == RELATION_SUPERSEDED_BY:
            dangling_supersedence.add(doc_path.get(src, f"<doc {src}>"))

    target_rows = conn.execute(
        "SELECT doc_id, target, relation FROM doc_link_targets ORDER BY doc_id, target, relation"
    ).fetchall()
    for row in target_rows:
        src = row["doc_id"]
        if (
            _ADR_NUMBER_RE.fullmatch(row["target"])
            and row["relation"] != RELATION_SUPERSEDED_BY
            and _is_work_log(doc_path.get(src, ""))
        ):
            # The relation guard matters: the rule exists because a ledger
            # MENTIONS a decision it worked on, which says nothing about
            # subject. A ledger declaring itself superseded is not a mention,
            # it is a claim about that document's own status, and dropping it
            # here would remove the ranking penalty AND skip the report that
            # exists to catch exactly that (issue #21), leaving no trace at
            # all.
            from_work_logs += 1
            continue
        dst, ambiguous = resolve(src, row["target"])
        if dst == src and dst is not None:
            # A document declaring itself superseded by itself is as much a
            # broken claim as one naming a document that does not exist, and
            # produces the same missing edge, so it is reported the same way.
            note_if_supersedence(src, row["relation"])
            unlinkable += 1
            continue
        if dst is None:
            note_if_supersedence(src, row["relation"])
            if ambiguous:
                unlinkable += 1
            else:
                unresolved += 1
            continue
        edges.add((src, dst, row["relation"]))
    if edges:
        conn.executemany(
            "INSERT INTO doc_links (src_doc_id, dst_doc_id, relation) VALUES (?, ?, ?)",
            sorted(edges),
        )
    return LinkGraphResult(
        edges=len(edges),
        unresolved=unresolved,
        unlinkable=unlinkable,
        from_work_logs=from_work_logs,
        superseded_by_unresolved=tuple(sorted(dangling_supersedence)),
    )


def reindex(
    *,
    db_path: Path | None = None,
    workspace_root: Path | None = None,
    full: bool = False,
    embedder: OllamaEmbedder | None = None,
) -> ReindexStats:
    """Incrementally (re)build the index from the on-disk corpus.

    With ``full=True`` every discovered document is re-chunked regardless of
    its stored fingerprint, useful after a chunker change. Embedding coverage
    is always backfilled to completion when the embedding backend is
    reachable, independent of ``full``: a chunk lacking a vector gets one.
    Holds the single-writer lock for the whole run (acquired before opening
    the database, so no schema write happens outside the lock); raises
    :class:`corpusdex.db.IndexLocked` if another writer already holds it,
    or :class:`RegistryMissing` if no repo registry can be found and no
    explicit ``BRAIN_CORPUS_ROOTS`` was configured.
    """
    start = time.monotonic()
    resolved_db_path = Path(db_path) if db_path is not None else db.default_db_path()
    if workspace_root is not None:
        resolved_workspace_root: Path | None = Path(workspace_root)
    elif configured_corpus_roots():
        # Explicit roots are self-contained scope. Resolving a workspace root
        # anyway would raise NotConfigured on exactly the installation this
        # setting exists to serve: someone outside the workspace it grew up in.
        resolved_workspace_root = None
    else:
        resolved_workspace_root = db.workspace_root()

    stats = ReindexStats()
    corpus = discover_corpus(resolved_workspace_root)
    stats.docs_seen = len(corpus)
    seen_paths = {doc.rel_path for doc in corpus}

    with db.write_lock(resolved_db_path):
        # A rebuild is never done in place. The live index stays untouched
        # until a complete replacement exists, so a concurrent reader cannot
        # observe an empty half-built index and a failed run cannot leave one
        # behind. Incremental runs against a current index write directly.
        rebuilding = full or not _index_is_current(resolved_db_path)
        work_path = Path(f"{resolved_db_path}.rebuild") if rebuilding else resolved_db_path
        if rebuilding:
            db.discard_index(work_path)
        try:
            stats = _run_indexing_pass(
                work_path=work_path,
                corpus=corpus,
                seen_paths=seen_paths,
                stats=stats,
                full=full,
                embedder=embedder,
            )
        except BaseException:
            if rebuilding:
                db.discard_index(work_path)
            raise
        if rebuilding:
            db.swap_index(work_path, resolved_db_path)

    stats.duration_seconds = time.monotonic() - start
    return stats


def _index_is_current(db_path: Path) -> bool:
    """True when ``db_path`` is an index this build can update incrementally.

    A missing, empty, unreadable, or older index needs a full rebuild. A
    *newer* one is refused outright rather than rebuilt: an older checkout
    silently downgrading a healthy index built by newer code loses work that
    the newer checkout would have kept.

    An index whose stored link-extractor version differs from
    :data:`corpusdex.links.EXTRACTOR_VERSION` is also not current, in either
    direction. Link targets are extracted only while chunking, and an
    incremental pass skips every unmoved document, so a changed extractor
    would otherwise leave the whole stored target set as the previous one
    wrote it and the change would appear to do nothing. Handled here rather
    than as a separate code path because "this index cannot be updated
    incrementally by this build" is the question this function already
    answers, and the rebuild-and-swap it triggers is already the right
    response.
    """
    if not db_path.is_file():
        return False
    stored_version = db.stored_schema_version(db_path)
    if stored_version is None:
        return False
    try:
        stored = int(stored_version)
    except ValueError:
        return False
    if stored > db.SCHEMA_VERSION:
        raise db.SchemaVersionMismatch(
            f"index schema version {stored} is newer than this build's version "
            f"{db.SCHEMA_VERSION}: the code writing the index is older than the "
            "index itself. Refusing to rebuild, because an older writer would "
            "silently downgrade an index a newer checkout built. Update the "
            "checkout, or delete the index file deliberately and reindex."
        )
    if stored != db.SCHEMA_VERSION:
        return False
    stored_extractor = db.stored_meta(db_path, db.META_LINK_EXTRACTOR_VERSION)
    # An index predating the stamp reads as None and rebuilds once, which is
    # correct: its targets were written by an extractor whose version is
    # unknown, so it cannot be assumed to match.
    return stored_extractor == str(EXTRACTOR_VERSION)


def _run_indexing_pass(
    *,
    work_path: Path,
    corpus: list[CorpusDoc],
    seen_paths: set[str],
    stats: ReindexStats,
    full: bool,
    embedder: OllamaEmbedder | None,
) -> ReindexStats:
    """Index ``corpus`` into ``work_path``, which may be a scratch rebuild file."""
    conn, vec_ok = db.open_index(work_path, create=True)
    # Resolved once for the run rather than per document: it can read the
    # environment and the filesystem, and a value that changed mid-run would
    # classify two documents in the same repo differently.
    knowledge = knowledge_repo()
    try:
        active_embedder = embedder or default_embedder()
        embed_fn = None
        if vec_ok:
            try:
                active_embedder.probe()
            except EmbeddingUnavailable as exc:
                stats.errors.append(f"embedding backend unavailable: {exc}")
            else:
                embed_fn = active_embedder.embed
                stats.embedding_available = True

        existing = {
            row["path"]: row
            for row in conn.execute(
                "SELECT id, path, mtime, content_hash FROM documents"
            ).fetchall()
        }

        with conn:
            if vec_ok and db.has_vec_table(conn):
                # Width first, because it rebuilds the table outright and so
                # subsumes the model-change delete below. The active width is
                # whatever the probe above already learned from the model; it
                # is None only when that probe failed, in which case nothing
                # will be embedded this run and the table must be left alone.
                # Reads the width the probe above already learned; never
                # triggers a second probe, because a failed probe must leave
                # this None rather than raise here.
                active_dim = getattr(active_embedder, "dim", None)
                table_dim = db.vec_table_dim(conn)
                if active_dim and table_dim and active_dim != table_dim:
                    db.recreate_vec_table(conn, active_dim)
                    stats.errors.append(
                        f"embedding dimension changed ({table_dim} -> {active_dim}); "
                        "vector table rebuilt, re-embedding via backfill"
                    )
                stored_model = db.get_meta(conn, db.META_EMBED_MODEL)
                # Gated on the probe having SUCCEEDED, for the same reason the
                # width rebuild above is. A failed probe means nothing can be
                # re-embedded this run, so deleting the old vectors trades a
                # stale-but-usable vector channel for no vector channel at all
                # and cannot repair it until the backend returns. The width
                # branch had this guard from the start and this one did not,
                # which is the asymmetry the review found.
                if (
                    stats.embedding_available
                    and stored_model
                    and stored_model != active_embedder.model
                ):
                    conn.execute("DELETE FROM vec_chunks")
                    stats.errors.append(
                        f"embedding model changed ({stored_model} -> "
                        f"{active_embedder.model}); all vectors invalidated, "
                        "re-embedding via backfill"
                    )
                _reconcile_orphaned_vectors(conn)

            for doc in corpus:
                prior = existing.get(doc.rel_path)
                try:
                    stat = doc.abs_path.stat()
                except OSError as exc:
                    stats.errors.append(f"{doc.rel_path}: {exc}")
                    continue
                file_mtime = stat.st_mtime

                if (
                    not full
                    and prior is not None
                    and prior["mtime"] == file_mtime
                    and _fingerprint_size(prior["content_hash"]) == stat.st_size
                ):
                    stats.unchanged += 1
                    continue

                try:
                    raw = doc.abs_path.read_bytes()
                except OSError as exc:
                    stats.errors.append(f"{doc.rel_path}: {exc}")
                    continue
                fingerprint = _fingerprint(raw)

                if not full and prior is not None and prior["content_hash"] == fingerprint:
                    conn.execute(
                        "UPDATE documents SET mtime = ? WHERE id = ?",
                        (file_mtime, prior["id"]),
                    )
                    stats.unchanged += 1
                    continue

                text = raw.decode("utf-8", errors="replace")
                fallback_title = doc.rel_path.rsplit(".", 1)[0]
                parsed = chunk_markdown(text, fallback_title=fallback_title)
                doc_type = classify_doc_type(doc.repo, doc.rel_path, knowledge)

                if prior is not None:
                    _delete_doc_vectors(conn, vec_ok, prior["id"])
                    conn.execute("DELETE FROM documents WHERE id = ?", (prior["id"],))
                    stats.changed += 1
                else:
                    stats.added += 1

                cur = conn.execute(
                    "INSERT INTO documents "
                    "(repo, path, title, doc_type, mtime, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc.repo, doc.rel_path, parsed.title, doc_type, file_mtime, fingerprint),
                )
                doc_id = cur.lastrowid

                # Ordinal counts prior chunks sharing this heading path in
                # this document, so a document with two identically titled
                # sections still gets distinct refs, and adding a section
                # elsewhere does not renumber the ones below it.
                heading_seen: dict[str, int] = {}
                for chunk in parsed.chunks:
                    ordinal = heading_seen.get(chunk.heading_path, 0)
                    heading_seen[chunk.heading_path] = ordinal + 1
                    conn.execute(
                        "INSERT INTO chunks "
                        "(ref, doc_id, heading_path, body, decided_on, superseded_by, tags) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            db.chunk_ref(doc.rel_path, chunk.heading_path, ordinal),
                            doc_id,
                            chunk.heading_path,
                            chunk.body,
                            parsed.decided_on,
                            parsed.superseded_by,
                            parsed.tags,
                        ),
                    )
                    stats.chunks_written += 1

                # Body links are read from the post-frontmatter text so a
                # path inside the YAML header cannot also register as a
                # body reference under a second relation.
                frontmatter, link_body = split_frontmatter(text)
                for link in extract_links(frontmatter, link_body):
                    conn.execute(
                        "INSERT OR IGNORE INTO doc_link_targets "
                        "(doc_id, target, relation) VALUES (?, ?, ?)",
                        (doc_id, link.target, link.relation),
                    )

            for path, row in existing.items():
                if path not in seen_paths:
                    _delete_doc_vectors(conn, vec_ok, row["id"])
                    conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
                    stats.removed += 1

            # After every document insert and delete, so resolution sees
            # the final id for each path.
            link_result = rebuild_link_graph(conn)
            stats.link_edges = link_result.edges
            stats.link_targets_unresolved = link_result.unresolved
            stats.link_targets_unlinkable = link_result.unlinkable
            stats.link_targets_from_work_logs = link_result.from_work_logs
            stats.superseded_by_unresolved = link_result.superseded_by_unresolved

            if embed_fn is not None:
                embedded_now, still_available = _backfill_missing_embeddings(conn, embed_fn)
                stats.embedded_chunks = embedded_now
                if not still_available:
                    stats.embedding_available = False
                    stats.errors.append("embedding backend became unavailable during backfill")

            stats.vector_total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            if vec_ok and db.has_vec_table(conn):
                stats.vector_covered = conn.execute(
                    "SELECT COUNT(*) AS n FROM vec_chunks"
                ).fetchone()["n"]

            db.set_meta(conn, db.META_LAST_REINDEX, datetime.now(UTC).isoformat())
            # Stamped only after a pass completes, so an interrupted run
            # leaves the old version in place and the next run rebuilds
            # again rather than trusting a half-written target set.
            db.set_meta(conn, db.META_LINK_EXTRACTOR_VERSION, str(EXTRACTOR_VERSION))
            if not vec_ok:
                embed_status = db.EMBED_STATUS_DISABLED
            elif stats.fully_embedded:
                embed_status = db.EMBED_STATUS_READY
            else:
                embed_status = db.EMBED_STATUS_UNAVAILABLE
            db.set_meta(conn, db.META_EMBED_STATUS, embed_status)
            # Only record a model we actually reached. Storing an
            # unreachable one makes the next run see no change and skip the
            # invalidation that this run deferred.
            if vec_ok and stats.embedding_available:
                db.set_meta(conn, db.META_EMBED_MODEL, active_embedder.model)
            elif not vec_ok:
                db.set_meta(conn, db.META_EMBED_MODEL, "")
            # Recorded for reporting and for a reader that wants the width
            # without opening the vec table. vec_table_dim() stays the
            # authority: this row is a copy, and a copy can go stale.
            live_dim = db.vec_table_dim(conn) if vec_ok else None
            if live_dim:
                db.set_meta(conn, db.META_EMBED_DIM, str(live_dim))
    finally:
        conn.close()

    return stats
