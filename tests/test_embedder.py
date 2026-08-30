"""Coverage for corpusdex.embedder: loopback-only local HTTP client behaviour.

These tests run a minimal in-process HTTP server on loopback to stand in for
Ollama, so they never depend on a real Ollama process being up. This also
lets the proxy-bypass test prove something a live Ollama process could not by
itself: that HTTP_PROXY/HTTPS_PROXY cannot divert the call, by pointing those
env vars at a dead port and showing the real loopback target still answers.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from corpusdex import db
from corpusdex.embedder import EmbeddingUnavailable, OllamaEmbedder

# Nothing listens here: if a proxy env var were honoured, the request would
# be routed through this unreachable address and fail even though the real
# fake-Ollama target below is up.
DEAD_PROXY = "http://127.0.0.1:1"


def _make_handler(response_body: dict):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (stdlib handler method name)
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            payload = json.dumps(response_body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:  # silence default stderr logging
            pass

    return Handler


class _FakeOllama:
    """A throwaway loopback HTTP server standing in for Ollama's /api/embed."""

    def __init__(self, response_body: dict) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(response_body))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> _FakeOllama:
        self.thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


@pytest.fixture
def fake_ollama() -> Iterator[type]:
    yield _FakeOllama


def test_embed_ignores_https_proxy_env_var(monkeypatch, fake_ollama):
    # This machine runs proxy-injecting tooling, so a stray HTTP_PROXY /
    # HTTPS_PROXY is a real, not hypothetical, way for a supposedly
    # loopback-only Ollama call to leave the machine.
    monkeypatch.setenv("HTTPS_PROXY", DEAD_PROXY)
    monkeypatch.setenv("HTTP_PROXY", DEAD_PROXY)

    vector = [0.25] * db.EMBED_DIM
    with fake_ollama({"embeddings": [vector]}) as fake:
        embedder = OllamaEmbedder(host=fake.host, model="test-model")
        result = embedder.embed(["hello"])

    # If the proxy env var had been honoured, this call would have failed
    # (connection refused against the dead proxy) instead of returning.
    assert result == [vector]


def test_embed_wraps_nonnumeric_vector_values_into_embedding_unavailable(fake_ollama):
    bad_vector = [0.1] * (db.EMBED_DIM - 1) + ["not-a-number"]
    with fake_ollama({"embeddings": [bad_vector]}) as fake:
        embedder = OllamaEmbedder(host=fake.host, model="test-model")
        with pytest.raises(EmbeddingUnavailable):
            embedder.embed(["hello"])


def test_the_dimension_is_learned_from_the_model_not_from_the_index(fake_ollama):
    """#15: pinning ``dim`` to the index constant made the model setting inert,
    because only 768-wide models could ever validate."""
    narrow = [0.1] * 384
    with fake_ollama({"embeddings": [narrow]}) as fake:
        embedder = OllamaEmbedder(host=fake.host, model="narrow-model")
        assert embedder.dim is None
        assert embedder.embed(["hello"]) == [narrow]
        assert embedder.dim == 384
        assert embedder.dim != db.EMBED_DIM


def test_a_width_that_changes_mid_run_is_rejected(fake_ollama):
    """Within one model this cannot legitimately vary; a disagreement means the
    backend swapped models underneath us."""
    with fake_ollama({"embeddings": [[0.1] * 384, [0.1] * 768]}) as fake:
        embedder = OllamaEmbedder(host=fake.host, model="unstable")
        with pytest.raises(EmbeddingUnavailable):
            embedder.embed(["a", "b"])


def test_dimension_probes_once_and_caches(fake_ollama):
    with fake_ollama({"embeddings": [[0.1] * 512]}) as fake:
        embedder = OllamaEmbedder(host=fake.host, model="m")
        assert embedder.dimension() == 512
        assert embedder.dimension() == 512


def test_an_empty_embedding_is_rejected_rather_than_sized_at_zero(fake_ollama):
    """Width zero is indistinguishable from "width not yet known": every guard
    built on ``dim`` tests truthiness, so accepting 0 silently disables the
    width-change rebuild and the search-side mismatch check at once, and the
    failure surfaces much later as a raw SQLite error."""
    with fake_ollama({"embeddings": [[]]}) as fake:
        embedder = OllamaEmbedder(host=fake.host, model="empty-model")
        with pytest.raises(EmbeddingUnavailable, match="empty embedding"):
            embedder.embed(["anything"])
        assert embedder.dim is None
