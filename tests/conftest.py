from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corpusdex import db
from corpusdex.embedder import EmbeddingUnavailable


class StubEmbedder:
    """Deterministic in-process stand-in for :class:`OllamaEmbedder`.

    No network I/O; every entry point is either a fixed deterministic vector
    or a raised :class:`EmbeddingUnavailable`, so tests never depend on
    Ollama running.
    """

    def __init__(self, dim: int = db.EMBED_DIM, model: str = "stub-embed"):
        self.dim = dim
        self.model = model
        self.calls: list[list[str]] = []

    def _vector_for(self, text: str) -> list[float]:
        # A cheap deterministic embedding: hash characters into a fixed-size
        # vector. Good enough to exercise storage/query plumbing; no claim of
        # semantic meaning.
        vector = [0.0] * self.dim
        for i, ch in enumerate(text):
            vector[(i + ord(ch)) % self.dim] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return [self._vector_for(t) for t in texts]

    def probe(self) -> None:
        self.embed(["probe"])

    def dimension(self) -> int:
        self.probe()
        return self.dim


class FailingEmbedder:
    """Stand-in embedder that always raises, simulating Ollama being down."""

    model = "unreachable-model"

    def embed(self, texts):
        raise EmbeddingUnavailable("stub: Ollama is not running")

    def probe(self) -> None:
        self.embed(["probe"])

    def dimension(self) -> int:
        self.probe()
        raise AssertionError("unreachable: probe() raises")


def _settings_names() -> list[str]:
    """Every environment name this engine reads, both sides of the aliases."""
    from corpusdex import config

    names = ["BRAIN_MCP_NAME", *config.CORPUS_SETTINGS]
    for canonical, legacy in config.LEGACY_ENV_ALIASES.items():
        names += [canonical, legacy]
    return names


@pytest.fixture(scope="session", autouse=True)
def isolate_settings_for_the_session():
    """Clear every setting once, before any fixture of any scope runs.

    Session scope is load-bearing, not caution. A function-scoped version of
    this passed the suite and still let a hostile environment break it,
    because module-scoped fixtures build their workspaces outside any
    function-scoped patch: the isolation ran, and ran too late to matter.
    """
    with pytest.MonkeyPatch.context() as patch:
        for name in _settings_names():
            patch.delenv(name, raising=False)
        yield


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    """Re-clear per test, so one test's setenv cannot reach the next.

    Both sides of every alias, because deleting only the canonical name
    leaves the legacy one free to supply a value -- which is precisely the
    hole the aliases exist to fill.
    """
    for name in _settings_names():
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def failing_embedder() -> FailingEmbedder:
    return FailingEmbedder()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "var" / "index.db"


@pytest.fixture
def lexical_conn(db_path: Path):
    """A connection with the core schema only (no vector table)."""
    conn = db.connect(db_path)
    db.init_schema(conn, vec=False)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def vec_probe() -> bool:
    """True if the sqlite-vec extension actually loads in this environment."""
    conn = sqlite3.connect(":memory:")
    try:
        return db.load_vec(conn)
    finally:
        conn.close()


@pytest.fixture
def vec_conn(db_path: Path, vec_probe: bool):
    """A connection with sqlite-vec loaded and the vector schema created.

    ``load_vec`` must run on this connection before ``init_schema(vec=True)``:
    it is what actually loads the ``vec0`` module into it (``vec_probe`` only
    checked that the extension loads at all on a throwaway connection).
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    conn = db.connect(db_path)
    assert db.load_vec(conn) is True
    db.init_schema(conn, vec=True)
    try:
        yield conn
    finally:
        conn.close()


def write_registry(workspace_root: Path, repos: list[str]) -> Path:
    """Write a repo-registry fixture, matching the real registry format
    (``name<TAB>origin-url<TAB>branch``, one row per registered repo).

    ``discover_corpus`` requires this file to exist (see
    ``indexer.RegistryMissing``); every test workspace built by hand needs one.
    Written under the CURRENT filename, so the legacy name is exercised only
    by the test that is specifically about the fallback rather than by every
    test incidentally.
    """
    from corpusdex import indexer

    workspace_root.mkdir(parents=True, exist_ok=True)
    registry_path = workspace_root / indexer.REGISTRY_FILENAME
    lines = [f"{name}\thttps://example.invalid/{name}\tmain" for name in repos]
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return registry_path

