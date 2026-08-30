"""Coverage for the ``brain-mcp`` stdio server (see corpusdex.mcp_server).

The server is driven exactly the way a real MCP host would: spawned as a
subprocess and talked to over stdio with the official MCP SDK client
(``mcp.stdio_client`` + ``mcp.ClientSession``). The corpus behind it is built
once per module with the deterministic, offline :class:`StubEmbedder` from
``conftest.py`` (no network calls, real vectors), and the server subprocess
itself is pointed at an unreachable loopback Ollama host so it can never
reach the network either: this exercises the same degraded lexical-only path
``brain search`` takes when Ollama is down, over the real MCP protocol.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from conftest import StubEmbedder, write_registry
from mcp import ClientSession, StdioServerParameters, stdio_client

from corpusdex import db, indexer

# Nothing listens on loopback port 1: connection refused is immediate, so the
# server's embedder fails fast instead of waiting out a connect timeout.
UNREACHABLE_OLLAMA_HOST = "http://127.0.0.1:1"

CORPUS_HEADING = "Cache invalidation rules"
CORPUS_BODY = "keep the derived cache consistent. " * 10


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


async def _run_scenario(*, workspace_root: Path, db_path: Path) -> dict:
    """Spawn ``brain-mcp`` over stdio and exercise all three tools once."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "corpusdex.mcp_server"],
        env={
            "BRAIN_WORKSPACE_ROOT": str(workspace_root),
            "BRAIN_DB": str(db_path),
            "BRAIN_OLLAMA_HOST": UNREACHABLE_OLLAMA_HOST,
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()

            listed = await session.list_tools()

            search_result = await session.call_tool(
                "brain_search", {"query": "cache invalidation rules", "n": 5}
            )
            search_payload = search_result.structured_content

            ref = search_payload["results"][0]["ref"]
            get_result = await session.call_tool("brain_get", {"ref": ref})

            missing_get_result = await session.call_tool(
                "brain_get", {"ref": "cdeadbeefdeadbeef"}
            )

            no_query_result = await session.call_tool(
                "brain_search", {"query": "no-such-term-anywhere-xyz"}
            )

            negative_n_result = await session.call_tool(
                "brain_search", {"query": "cache invalidation rules", "n": -1}
            )

            zero_n_result = await session.call_tool(
                "brain_search", {"query": "cache invalidation rules", "n": 0}
            )

            status_result = await session.call_tool("brain_status", {})

    return {
        "server_name": init_result.server_info.name,
        "tools": {t.name: t for t in listed.tools},
        "search": search_result,
        "get": get_result,
        "missing_get": missing_get_result,
        "no_query": no_query_result,
        "negative_n_search": negative_n_result,
        "zero_n_search": zero_n_result,
        "status": status_result,
    }


async def _run_missing_index_scenario(*, workspace_root: Path, db_path: Path) -> dict:
    """Spawn ``brain-mcp`` against a workspace whose index was never built."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "corpusdex.mcp_server"],
        env={
            "BRAIN_WORKSPACE_ROOT": str(workspace_root),
            "BRAIN_DB": str(db_path),
            "BRAIN_OLLAMA_HOST": UNREACHABLE_OLLAMA_HOST,
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            status_result = await session.call_tool("brain_status", {})
            search_result = await session.call_tool("brain_search", {"query": "anything"})
    return {"status": status_result, "search": search_result}


def test_brain_status_reports_cleanly_when_index_was_never_built(tmp_path_factory):
    # Read surfaces (the MCP server included) must never mint an empty index
    # on a fresh checkout: brain_status/brain_search called before the first
    # `brain reindex` must fail cleanly, and must not create a DB file as a
    # side effect of merely being asked a question.
    tmp_path = tmp_path_factory.mktemp("mcp-brain-no-index")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# Repo A\n\n## Section\n\ntext " * 20)
    write_registry(workspace, ["repo-a"])

    result = asyncio.run(_run_missing_index_scenario(workspace_root=workspace, db_path=db_path))

    assert result["status"].is_error
    assert result["search"].is_error
    assert not db_path.exists()


@pytest.fixture(scope="module")
def mcp_scenario(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("mcp-brain")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"

    _write(
        workspace / "repo-a" / "AGENTS.md",
        f"# Repo A agents\n\n## {CORPUS_HEADING}\n\n{CORPUS_BODY}",
    )
    write_registry(workspace, ["repo-a"])

    # Built with the deterministic offline StubEmbedder: real vectors, no
    # network, so the corpus behind the server is genuinely hybrid-indexed
    # even though the live server itself can never reach Ollama (see
    # UNREACHABLE_OLLAMA_HOST above).
    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.added == 1
    assert stats.embedding_available is True

    return asyncio.run(_run_scenario(workspace_root=workspace, db_path=db_path))


def test_server_identifies_itself_as_corpusdex(mcp_scenario):
    """The name is an integration identity a client config names back, so a
    deployment overrides it rather than inheriting; this pins the default and
    the override together, because only the pair is the contract."""
    assert mcp_scenario["server_name"] == "corpusdex"

    from corpusdex import mcp_server

    assert mcp_server.MCP_SERVER_NAME == "corpusdex"


def test_tool_listing_exposes_exactly_the_three_read_only_tools(mcp_scenario):
    names = set(mcp_scenario["tools"])
    assert names == {"brain_search", "brain_get", "brain_status"}
    # Read-only per the design brief: no reindex tool, since indexing mutates
    # the shared index and takes the single-writer lock.
    assert "reindex" not in names
    assert "brain_reindex" not in names


def test_tool_descriptions_are_present_for_client_discovery(mcp_scenario):
    for name, tool in mcp_scenario["tools"].items():
        assert tool.description, f"{name} is missing a description"


def test_brain_search_returns_cited_results(mcp_scenario):
    payload = mcp_scenario["search"].structured_content
    assert payload["query"] == "cache invalidation rules"
    assert len(payload["results"]) == 1
    hit = payload["results"][0]
    assert hit["citation"].startswith("repo-a/AGENTS.md#")
    assert CORPUS_HEADING in hit["citation"] or CORPUS_HEADING.lower() in hit["snippet"].lower()
    assert isinstance(hit["score"], (int, float))
    assert hit["ref"].startswith("c")
    # The unstable rowid is deliberately absent from every published shape
    # (issue #23): a client cannot store a handle it never receives.
    assert "chunk_id" not in hit
    # brain_search's result dicts are the citation-carrying summary shape
    # (corpusdex.cli._hit_dict with full_body=False): no full chunk body.
    assert "body" not in hit


def test_brain_search_reports_degraded_lexical_only_without_ollama(mcp_scenario):
    payload = mcp_scenario["search"].structured_content
    assert payload["degraded"] is True
    assert payload["mode"] == db.MODE_LEXICAL
    assert payload["degraded_reason"]
    assert not mcp_scenario["search"].is_error


def test_brain_search_no_results_is_not_an_error(mcp_scenario):
    result = mcp_scenario["no_query"]
    assert not result.is_error
    assert result.structured_content["results"] == []


def test_brain_get_round_trips_the_full_chunk_body(mcp_scenario):
    search_hit = mcp_scenario["search"].structured_content["results"][0]
    get_payload = mcp_scenario["get"].structured_content
    assert get_payload["ref"] == search_hit["ref"]
    assert get_payload["citation"] == search_hit["citation"]
    assert "body" in get_payload
    assert "derived cache consistent" in get_payload["body"]


def test_brain_get_missing_ref_is_a_tool_error(mcp_scenario):
    result = mcp_scenario["missing_get"]
    assert result.is_error
    text = result.content[0].text
    assert "cdeadbeefdeadbeef" in text
    # The error has to distinguish "gone" from "your handle went stale", or
    # the caller retries the same dead ref instead of searching again.
    assert "brain_search" in text


def test_brain_search_rejects_negative_n(mcp_scenario):
    # The CLI already rejects non-positive -n via argparse; the MCP tool has
    # no argparse layer, so it must reject in-process instead of silently
    # slicing results with a negative or zero limit.
    result = mcp_scenario["negative_n_search"]
    assert result.is_error
    assert "-1" in result.content[0].text


def test_brain_search_rejects_zero_n(mcp_scenario):
    result = mcp_scenario["zero_n_search"]
    assert result.is_error


LINKED_QUERY = "watering"
LINKED_FILLER = "notes about the watering schedule for the season. " * 6


async def _run_linked_scenario(*, workspace_root: Path, db_path: Path) -> dict:
    """Exercise brain_search/brain_get over a corpus that has a link graph."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "corpusdex.mcp_server"],
        env={
            "BRAIN_WORKSPACE_ROOT": str(workspace_root),
            "BRAIN_DB": str(db_path),
            "BRAIN_OLLAMA_HOST": UNREACHABLE_OLLAMA_HOST,
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            search_result = await session.call_tool(
                "brain_search", {"query": LINKED_QUERY, "n": 5}
            )
            payload = search_result.structured_content
            guide = next(
                hit for hit in payload["results"] if hit["path"].endswith("guide.md")
            )
            get_result = await session.call_tool("brain_get", {"ref": guide["ref"]})
    return {"search": search_result, "guide": guide, "get": get_result}


@pytest.fixture(scope="module")
def mcp_linked_scenario(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("mcp-brain-linked")
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"

    _write(
        workspace / "repo-a" / "docs" / "guide.md",
        f"---\nsuperseded_by: revised\n---\n\n# Guide\n\n## Schedule\n\n"
        f"{LINKED_QUERY} {LINKED_FILLER} see [[appendix]].\n",
    )
    _write(
        workspace / "repo-a" / "docs" / "appendix.md",
        f"# Appendix\n\n## Body\n\n{LINKED_FILLER}\n",
    )
    _write(
        workspace / "repo-a" / "docs" / "revised.md",
        f"# Revised\n\n## Body\n\n{LINKED_FILLER}\n",
    )
    write_registry(workspace, ["repo-a"])

    stats = indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())
    assert stats.link_edges == 2

    return asyncio.run(_run_linked_scenario(workspace_root=workspace, db_path=db_path))


def test_brain_search_hits_carry_assembled_context(mcp_linked_scenario):
    relations = {
        ref["relation"]: ref for ref in mcp_linked_scenario["guide"]["assembled"]
    }
    assert relations["superseded_by"]["path"].endswith("revised.md")
    assert relations["links_to"]["path"].endswith("appendix.md")


def test_assembled_references_are_compact(mcp_linked_scenario):
    for ref in mcp_linked_scenario["guide"]["assembled"]:
        assert set(ref) == {"doc_id", "title", "repo", "path", "relation"}
        # References are pointers, not a second copy of the corpus.
        assert "body" not in ref


def test_brain_get_returns_assembled_context(mcp_linked_scenario):
    payload = mcp_linked_scenario["get"].structured_content
    assert "body" in payload
    relations = {ref["relation"] for ref in payload["assembled"]}
    assert {"superseded_by", "links_to"} <= relations


def test_assembled_context_is_additive_to_the_existing_hit_shape(mcp_linked_scenario):
    # Backward compatibility: every field an existing client already reads is
    # still present alongside the new ones.
    hit = mcp_linked_scenario["guide"]
    for existing in (
        "ref",
        "repo",
        "path",
        "title",
        "heading_path",
        "doc_type",
        "snippet",
        "score",
        "decided_on",
        "superseded_by",
        "tags",
        "citation",
    ):
        assert existing in hit, f"missing pre-existing field: {existing}"


def test_brain_status_reports_the_same_fields_as_brain_status_cli(mcp_scenario):
    payload = mcp_scenario["status"].structured_content
    for field in (
        "db_path",
        "documents",
        "chunks",
        "embedded_chunks",
        "fully_embedded",
        "schema_version",
        "code_schema_version",
        "last_reindex_at",
        "vector_extension_loaded",
        "embed_backend_live",
        "embed_backend_detail",
    ):
        assert field in payload, f"missing status field: {field}"
    assert payload["documents"] == 1
    assert payload["chunks"] >= 1
    # The corpus itself was embedded offline via StubEmbedder (real vectors
    # on disk), but the live server process only ever sees the unreachable
    # HARNX_BRAIN_OLLAMA_HOST, so its own embedder probe must report down.
    assert payload["embed_backend_live"] == "unavailable"
    assert payload["embed_backend_detail"]
