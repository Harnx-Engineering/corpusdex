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
    ) -> None:
        self.host = _validate_loopback(
            host or env("BRAIN_OLLAMA_HOST", DEFAULT_HOST)
        )
        self.model = model or env("BRAIN_EMBED_MODEL", DEFAULT_MODEL)
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        #: Width of this model's vectors, learned from the first response.
        #: None until then. It is NOT seeded from the index's constant: the
        #: model decides its own width, and pinning it here is what made the
        #: model setting inert (only 768-wide models could be selected).
        self.dim: int | None = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in order, batching requests."""
        if not texts:
            return []
        out: list[list[float]] = []
        # trust_env=False: never read HTTP_PROXY/HTTPS_PROXY/NO_PROXY from the
        # environment. This is a loopback-only client by design (see
        # _validate_loopback); honouring an ambient proxy variable would let
        # something outside this process's control silently reroute a call
        # that is supposed to never leave the machine.
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                out.extend(self._embed_batch(client, batch))
        return out

    def probe(self) -> None:
        """Raise :class:`EmbeddingUnavailable` unless the backend can embed."""
        self.embed(["embedding backend readiness probe"])

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
