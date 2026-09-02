"""Coverage for corpusdex.embedder: loopback-only local HTTP client behaviour.

These tests run a minimal in-process HTTP server on loopback to stand in for
Ollama, so they never depend on a real Ollama process being up. This also
lets the proxy-bypass test prove something a live Ollama process could not by
itself: that HTTP_PROXY/HTTPS_PROXY cannot divert the call, by pointing those
env vars at a dead port and showing the real loopback target still answers.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import httpx
import pytest

from corpusdex import db
from corpusdex.embedder import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_TIMEOUT,
    EmbeddingUnavailable,
    OllamaEmbedder,
    default_embedder,
)

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


class _HungOllama:
    """A loopback socket that accepts a connection and then never answers.

    Deliberately a raw socket rather than another ``ThreadingHTTPServer``: the
    behaviour under test is the absence of a response, and a stdlib HTTP server
    exists to produce one. Building the hang out of a server that wants to
    reply means the test passes only as long as nothing in that server decides
    to send an error page, which is a weaker guarantee than never writing at
    all.

    This is the failure mode issue #16 is about, and the one a timeout is the
    only defence against. A STOPPED backend is refused by the kernel and
    already failed fast; a RUNNING but wedged one produces no event of any
    kind, so the client waits for its whole budget and nothing raises.
    """

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self._accepted: list[socket.socket] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._accept_forever, daemon=True)

    def _accept_forever(self) -> None:
        self.sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except (TimeoutError, OSError):
                continue
            # Held open, never written to. Closing it would hand the client an
            # immediate EOF, which is a connection error rather than a hang.
            self._accepted.append(conn)

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.sock.getsockname()[1]}"

    def __enter__(self) -> _HungOllama:
        self.thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self.thread.join(timeout=5)
        for conn in self._accepted:
            conn.close()
        self.sock.close()


def test_a_hung_backend_fails_the_probe_within_the_probe_budget():
    """The whole point of #16: `brain status` waited two minutes on a wedged server.

    Asserts the elapsed time, not merely that it raised, because the pre-fix
    code raised too. It just did so after ``DEFAULT_TIMEOUT``.
    """
    with _HungOllama() as hung:
        embedder = OllamaEmbedder(host=hung.host, probe_timeout=0.4, timeout=30.0)
        start = time.monotonic()
        with pytest.raises(EmbeddingUnavailable) as caught:
            embedder.probe()
        elapsed = time.monotonic() - start

    # Generous upper bound: the assertion that matters is that this is nowhere
    # near the 30s indexing timeout the same embedder is configured with, so a
    # slow CI machine must not be able to fail it for the wrong reason.
    assert elapsed < 5.0, f"probe took {elapsed:.2f}s"
    message = str(caught.value)
    assert "sent no response" in message
    assert "hung rather than absent" in message


def test_the_probe_budget_does_not_become_the_indexing_budget():
    """The two timeouts must stay independent per call.

    Pinned by capturing what each call hands to httpx rather than by timing,
    because a mutant that set ``self.timeout`` inside ``probe()`` would pass
    every timing assertion here: the probe itself would still be fast, and the
    damage would only appear on the NEXT indexing call through the same object.
    Both ``reindex`` probe paths reuse the embedder they then index with, so
    that object is exactly the one at risk.
    """
    seen: list[httpx.Timeout] = []
    real_client = httpx.Client

    class RecordingClient(real_client):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs["timeout"])
            super().__init__(*args, **kwargs)

    body = {"embeddings": [[0.5] * db.EMBED_DIM]}
    with _FakeOllama(body) as server:
        embedder = OllamaEmbedder(
            host=server.host, timeout=99.0, probe_timeout=7.0, connect_timeout=1.5
        )
        with mock.patch.object(httpx, "Client", RecordingClient):
            embedder.probe()
            embedder.embed(["an ordinary indexing call"])
            embedder.probe()

    assert [t.read for t in seen] == [7.0, 99.0, 7.0]
    # connect is per-request and unrelated to which operation it is.
    assert [t.connect for t in seen] == [1.5, 1.5, 1.5]
    # The attribute was not mutated by either probe, which is the actual defect
    # this test exists to catch.
    assert embedder.timeout == 99.0


def test_the_connect_bound_is_clamped_to_a_shorter_budget():
    """A caller asking for less than the connect bound must still get less.

    Without the clamp the function's one promise, that a request fails within
    the budget, is false in exactly the case a caller asked for explicitly.
    """
    embedder = OllamaEmbedder(host="http://127.0.0.1:11434", connect_timeout=2.0)
    tight = embedder._client_timeout(0.25)
    assert tight.connect == 0.25
    assert tight.read == 0.25
    # ...and an ordinary budget leaves the connect bound alone rather than
    # raising it to match, so the split still does its job.
    roomy = embedder._client_timeout(60.0)
    assert roomy.connect == 2.0
    assert roomy.read == 60.0


def test_a_refused_connection_is_still_reported_as_absent_not_as_a_hang():
    """Control arm. Nothing listens on port 1, so the kernel refuses at once.

    The fix must not relabel a stopped backend as a wedged one: the two need
    opposite operator actions, start it versus restart it. Pre-fix this case
    already worked, and the risk in adding timeout branches is precisely that
    they capture cases that were classified correctly before.
    """
    embedder = OllamaEmbedder(host=DEAD_PROXY, probe_timeout=0.4)
    start = time.monotonic()
    with pytest.raises(EmbeddingUnavailable) as caught:
        embedder.probe()
    elapsed = time.monotonic() - start

    message = str(caught.value)
    assert "cannot reach Ollama" in message
    assert "hung" not in message
    assert "not accepting" not in message
    # A refusal is an event, so it does not consume the budget at all.
    assert elapsed < 0.4


def test_a_connect_timeout_says_the_server_is_running_but_not_accepting():
    """A saturated listen backlog leaves SYN unanswered, so the stall is in
    the CONNECT phase and reads as an absent server unless it is separated.

    Raised directly rather than reproduced with a firewall rule, because the
    behaviour under test is the CLASSIFICATION and its message, and httpx
    deciding which exception to raise is not this module's logic. The read-side
    hang is reproduced for real above, where it can be.
    """
    body = {"embeddings": [[0.5] * db.EMBED_DIM]}
    with _FakeOllama(body) as server:
        embedder = OllamaEmbedder(host=server.host, probe_timeout=5.0, connect_timeout=1.0)

        def refuse_to_connect(*args, **kwargs):
            raise httpx.ConnectTimeout("simulated saturated accept backlog")

        with mock.patch.object(httpx.Client, "post", refuse_to_connect):
            with pytest.raises(EmbeddingUnavailable) as caught:
                embedder.probe()

    message = str(caught.value)
    assert "did not answer the connection within 1.0s" in message
    assert "not accepting" in message
    # Must not be mistaken for the read-side hang, whose remedy differs.
    assert "sent no response" not in message


def test_the_default_embedder_carries_a_short_probe_budget():
    """Pins the DEFAULTS and the factory, not just the plumbing.

    Every test above passes ``probe_timeout`` explicitly, so all of them would
    keep passing if ``DEFAULT_PROBE_TIMEOUT`` were set back to 120, or if
    ``default_embedder()`` were changed to pass the indexing timeout through.
    That is the shape in which this fix goes inert: the mechanism intact, no
    caller reaching it. `brain status`, both `reindex` probe paths, and the MCP
    server all build their embedder through this factory and pass nothing.

    The ceiling is a relationship plus a human-scale bound rather than an equality
    on 10.0, so raising the constant for a slow machine does not require editing a
    test, while restoring it to the indexing timeout does fail.
    """
    embedder = default_embedder()
    assert embedder.probe_timeout == DEFAULT_PROBE_TIMEOUT
    assert embedder.connect_timeout == DEFAULT_CONNECT_TIMEOUT
    assert embedder.timeout == DEFAULT_TIMEOUT

    assert DEFAULT_PROBE_TIMEOUT < DEFAULT_TIMEOUT
    assert DEFAULT_PROBE_TIMEOUT <= 30.0, (
        "a probe budget above 30s makes `brain status` unusable as a health "
        "check, which is what issue #16 was filed about"
    )
    assert DEFAULT_CONNECT_TIMEOUT <= DEFAULT_PROBE_TIMEOUT
