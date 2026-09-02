"""Local embeddings via the Ollama HTTP API.

Repository content is never sent to a third-party embedding API, so the host is
validated to be loopback and the check is not overridable by configuration. When
the server is down or the model is missing, every entry point raises
:class:`EmbeddingUnavailable` and callers degrade to lexical-only behaviour.

The loopback validation is only meaningful if the request actually goes to the
validated host: the HTTP client is built with ``trust_env=False``, so an
``HTTP_PROXY``/``HTTPS_PROXY`` environment variable can never redirect a
supposedly loopback-only call through an external proxy. This matters
concretely in this workspace, which runs proxy-injecting tooling on developer
machines.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit

import httpx

from .config import env

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT = 120.0
#: Budget for a READINESS PROBE, which is a different operation from indexing
#: and so cannot share its timeout. A probe embeds one short string; the only
#: legitimate cost it can carry is Ollama loading the model, measured here at a
#: stable 0.312s cold and 0.05s warm for a 274MB model. 120s is roughly 380x
#: that, and it was being spent by `brain status`, the command whose entire
#: purpose is to answer quickly whether the backend is healthy.
#:
#: 10s is deliberately generous rather than aggressive, about 32x the measured
#: cold load, because the reverse failure is worse than a slow answer: a probe
#: that gives up early reports a slow-but-healthy backend as unavailable, and
#: two callers act on that. `reindex` skips embedding for the whole run, and
#: `_model_change_needs_a_rebuild` declines the rebuild-and-swap path and falls
#: back to invalidating in place, which is the behaviour decision 0045 exists
#: to avoid. A false negative here is not a cosmetic mis-report.
DEFAULT_PROBE_TIMEOUT = 10.0
#: Bound on establishing the TCP connection, applied to EVERY request rather
#: than only to probes, because it measures a different physical quantity from
#: the ones above. Connecting to a loopback port costs sub-millisecond and its
#: cost is bounded by the kernel, not by the model, the batch size, or the
#: machine's load, so no legitimate request needs seconds of it. Splitting it
#: out is what makes a hung backend fail fast in the case that a single scalar
#: cannot help: a live Ollama whose accept backlog is saturated leaves SYN
#: unanswered, so the stall is in the CONNECT phase and a long read budget
#: would otherwise absorb it.
DEFAULT_CONNECT_TIMEOUT = 2.0

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class EmbeddingUnavailable(RuntimeError):
    """The local embedding backend cannot serve a request."""


def _validate_loopback(host: str) -> str:
    parts = urlsplit(host)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"embedding host must be an http(s) URL, got {host!r}")
    if parts.hostname not in LOOPBACK_HOSTS:
        raise ValueError(
            f"embedding host must be loopback (one of {sorted(LOOPBACK_HOSTS)}), "
            f"got {parts.hostname!r}"
        )
    return host.rstrip("/")


class OllamaEmbedder:
    """Batching client for ``POST /api/embed`` on a local Ollama server."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> None:
        self.host = _validate_loopback(
            host or env("BRAIN_OLLAMA_HOST", DEFAULT_HOST)
        )
        self.model = model or env("BRAIN_EMBED_MODEL", DEFAULT_MODEL)
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.probe_timeout = probe_timeout
        self.connect_timeout = connect_timeout
        #: Width of this model's vectors, learned from the first response.
        #: None until then. It is NOT seeded from the index's constant: the
        #: model decides its own width, and pinning it here is what made the
        #: model setting inert (only 768-wide models could be selected).
        self.dim: int | None = None

    def _client_timeout(self, budget: float) -> httpx.Timeout:
        """Split ``budget`` into per-phase deadlines for one httpx client.

        ``connect`` is clamped to ``budget`` rather than used as given. Without
        the clamp a caller asking for a budget SHORTER than the connect bound
        (a test, or a caller wanting a sub-second answer) would still wait the
        full connect time, so the one promise this function makes, that a
        request fails within ``budget``, would be false in exactly the case a
        caller had gone out of their way to ask for.
        """
        return httpx.Timeout(budget, connect=min(self.connect_timeout, budget))

    def embed(self, texts: Sequence[str], *, budget: float | None = None) -> list[list[float]]:
        """Embed ``texts`` in order, batching requests.

        ``budget`` overrides the indexing timeout FOR THIS CALL ONLY. It is a
        parameter rather than an attribute assignment on purpose: a probe that
        set ``self.timeout`` would shorten every subsequent indexing request
        made through the same embedder, and both ``reindex`` paths probe with
        the same object they then index with.
        """
        if not texts:
            return []
        out: list[list[float]] = []
        # trust_env=False: never read HTTP_PROXY/HTTPS_PROXY/NO_PROXY from the
        # environment. This is a loopback-only client by design (see
        # _validate_loopback); honouring an ambient proxy variable would let
        # something outside this process's control silently reroute a call
        # that is supposed to never leave the machine.
        timeout = self._client_timeout(self.timeout if budget is None else budget)
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                out.extend(self._embed_batch(client, batch))
        return out

    def probe(self) -> None:
        """Raise :class:`EmbeddingUnavailable` unless the backend can embed.

        Bounded by ``probe_timeout``, not by the indexing timeout. A stopped
        backend already failed fast (the kernel refuses the connection), so the
        case this bound exists for is a backend that is RUNNING and wedged,
        where nothing fails at all and the caller simply waits.
        """
        self.embed(["embedding backend readiness probe"], budget=self.probe_timeout)

    def dimension(self) -> int:
        """Return this model's vector width, probing once if not yet known."""
        if self.dim is None:
            self.probe()
        if self.dim is None:  # pragma: no cover - probe raises instead
            raise EmbeddingUnavailable(f"model {self.model!r} produced no vector to size")
        return self.dim

    def _embed_batch(self, client: httpx.Client, batch: list[str]) -> list[list[float]]:
        try:
            response = client.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": batch},
            )
        except httpx.ConnectTimeout as exc:
            # Caught before TimeoutException, which is caught before HTTPError,
            # because httpx makes each a subclass of the next and the first
            # matching handler wins. Folded into the generic message these read
            # as "cannot reach Ollama", which points an operator at starting a
            # server that is already running.
            raise EmbeddingUnavailable(
                f"Ollama at {self.host} did not answer the connection within "
                f"{client.timeout.connect}s. A stopped server REFUSES the "
                "connection immediately, so this is a server that is running "
                "but not accepting: check whether the process is wedged or its "
                "listen backlog is saturated, rather than starting it again"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingUnavailable(
                f"Ollama at {self.host} accepted the request but sent no "
                f"response within {client.timeout.read}s. The server is hung "
                "rather than absent; restarting it is the fix, and `ollama ps` "
                "will show whether a model is stuck loading"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailable(f"cannot reach Ollama at {self.host}: {exc}") from exc
        if response.status_code == 404:
            raise EmbeddingUnavailable(
                f"Ollama has no model {self.model!r} (try: ollama pull {self.model})"
            )
        if response.status_code >= 400:
            raise EmbeddingUnavailable(
                f"Ollama returned HTTP {response.status_code} for /api/embed: {response.text[:200]}"
            )
        try:
            vectors = response.json()["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingUnavailable(f"unexpected /api/embed response from {self.host}") from exc
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise EmbeddingUnavailable(
                f"Ollama returned {len(vectors) if isinstance(vectors, list) else '?'} embeddings "
                f"for {len(batch)} inputs"
            )
        for vector in vectors:
            if not isinstance(vector, list):
                raise EmbeddingUnavailable(
                    f"model {self.model!r} returned a non-list embedding"
                )
            if not vector:
                # Guarding here rather than downstream: a zero width would be
                # recorded as this model's dimension, and every later guard
                # that asks "is the width known" tests truthiness, so 0 reads
                # as unknown and silently disables the checks built on it.
                raise EmbeddingUnavailable(
                    f"model {self.model!r} returned an empty embedding"
                )
            if self.dim is None:
                self.dim = len(vector)
            if len(vector) != self.dim:
                # Within one model this cannot legitimately vary, so a
                # disagreement means the backend swapped models underneath us
                # rather than that the index is configured wrongly.
                raise EmbeddingUnavailable(
                    f"model {self.model!r} returned inconsistent dimensions "
                    f"({len(vector)} after {self.dim})"
                )
        try:
            return [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as exc:
            raise EmbeddingUnavailable(
                f"model {self.model!r} returned a non-numeric embedding value: {exc}"
            ) from exc


def default_embedder() -> OllamaEmbedder:
    return OllamaEmbedder()
