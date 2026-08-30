"""Read-only stdio MCP server exposing the brain's hybrid search core.

Three tools, each a thin wrapper around the exact code paths ``brain``
(the CLI, see :mod:`corpusdex.cli`) already uses: :func:`corpusdex.db.open_index`,
:func:`corpusdex.search.search`, :func:`corpusdex.search.get_chunk`, and
:func:`corpusdex.cli.status_payload`. Nothing here re-implements retrieval,
so the degraded behaviour when Ollama is unreachable (see
:mod:`corpusdex.search`), where the vector channel drops out and lexical
plus graph carry the ranking, applies identically to ``brain_search`` here as
it does to ``brain search`` on the command line.

Deliberately read-only: there is no reindex tool. Indexing stays a CLI
concern (``brain reindex``), since it mutates the shared index and takes the
single-writer lock (see :func:`corpusdex.db.write_lock`); an MCP client
should not be able to trigger that as a side effect of a read request. For
the same reason, every ``open_index`` call here uses the default
``create=False``: a read tool called before the index has ever been built
raises :class:`corpusdex.db.IndexMissing` (surfaced to the MCP client as a
tool error) rather than silently minting an empty index.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from . import db
from . import search as search_mod
from .cli import _hit_dict, status_payload
from .config import env
from .embedder import default_embedder

#: The name this server registers under. It is an integration identity that
#: a client config names BACK, so a deployment whose clients already name
#: something else must pin that name here rather than inherit this default;
#: a rename that silently unhooks every registered client is worse than a
#: stale-looking string, and an unhooked MCP client reports nothing. Hence
#: ``BRAIN_MCP_NAME``, which is where such a deployment states its own.
MCP_SERVER_NAME = env("BRAIN_MCP_NAME") or "corpusdex"

mcp = MCPServer(
    MCP_SERVER_NAME,
    instructions=(
        "Read-only hybrid search over an indexed Markdown corpus: decisions, "
        "context cards, architecture maps, gap ledgers, and the docs of "
        "whichever repositories this index was configured to cover. Use "
        "brain_search to find relevant chunks, "
        "brain_get to read one chunk's full body by the 'ref' a search "
        "result carries, and brain_status to check "
        "index freshness and embedding backend health. Every brain_search "
        "response carries a citation (path#heading) for each result and "
        "reports whether the vector channel dropped out (e.g. because the "
        "local Ollama embedding backend is unreachable), in which case "
        "lexical and link-graph retrieval still produce the results and "
        "'mode' reads 'lexical+graph'. Read 'mode' and 'channels_used' as "
        "the answer to what produced a page: 'channels_used' is the exact "
        "set and 'mode' summarises it, so neither is a claim about backend "
        "health. Each result also "
        "carries an 'assembled' list: a few compact references to the "
        "document's supersedence chain and directly linked documents, so a "
        "replaced decision and its replacement surface together."
    ),
)


@mcp.tool()
def brain_search(query: str, n: int = 8) -> dict[str, Any]:
    """Hybrid (lexical + vector) search over the indexed corpus.

    Runs the same fused BM25 + vector + link-graph retrieval as ``brain
    search`` on the command line. Each result carries a citation
    (``path#heading``), a fused relevance score, a snippet, and an
    ``assembled`` list of compact references (``doc_id``, ``title``,
    ``repo``, ``path``, ``relation``) covering the document's supersedence
    chain in both directions and its directly linked documents. When the
    local Ollama embedding backend is unreachable, the vector channel is
    skipped automatically and lexical plus link-graph retrieval still serve
    results; the response's ``degraded`` flag and ``degraded_reason`` say so
    rather than raising an error, and ``mode`` reads ``lexical+graph``
    instead of overstating the loss as ``lexical-only``. ``channels_used``
    lists exactly which channels contributed, and ``mode`` is derived from
    it, so the two cannot disagree.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    embedder = default_embedder()
    conn, vec_ok = db.open_index()
    try:
        response = search_mod.search(conn, vec_ok, query, limit=n, embedder=embedder)
    finally:
        conn.close()
    return {
        "query": response.query,
        "mode": response.mode,
        "channels_used": sorted(response.channels_used),
        "degraded": response.degraded,
        "degraded_reason": response.degraded_reason,
        "results": [_hit_dict(hit) for hit in response.hits],
    }


@mcp.tool()
def brain_get(ref: str) -> dict[str, Any]:
    """Fetch one chunk's full body plus its document info, by its stable ref.

    ``ref`` comes from a prior ``brain_search`` result's ``ref`` field. It is
    derived from the chunk's position in its document, so it survives
    reindexing and stays valid while the section exists; if the section has
    been renamed or removed this raises rather than returning a different
    chunk. Carries the same ``assembled`` references as a search hit, so a
    chunk read on its own still shows what supersedes it and what it links
    to.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("ref must be a non-empty string from a brain_search result")
    conn, _vec_ok = db.open_index()
    try:
        hit = search_mod.get_chunk(conn, ref)
    finally:
        conn.close()
    if hit is None:
        raise ValueError(
            f"no chunk with ref {ref!r}; the section it named may have been "
            "renamed or removed. Run brain_search again for a current ref."
        )
    return _hit_dict(hit, full_body=True)


@mcp.tool()
def brain_status() -> dict[str, Any]:
    """Index freshness and embedding backend health.

    Reports the same fields as ``brain status`` on the command line:
    document/chunk/embedded-chunk counts, schema version, last reindex
    time, and whether the embedding backend is live right now.
    """
    return status_payload()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
