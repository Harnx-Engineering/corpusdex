from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import write_registry

from corpusdex import cli, db, indexer


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch, failing_embedder):
    """A workspace + index wired entirely through env var overrides, the same
    knobs the real CLI honours, with the embedder forced offline so no test
    ever depends on Ollama running.

    Patching ``cli.default_embedder`` alone must be sufficient to isolate
    every subcommand from the real network: cli.py resolves its embedder
    exclusively through this one module-level name (never re-resolving it
    through corpusdex.search or corpusdex.indexer's own imports), so this
    single monkeypatch is the whole isolation contract under test here.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(
        workspace / "repo-a" / "AGENTS.md",
        "# Repo A agents\n\n## Cache invalidation rules\n\n"
        + "keep the derived cache consistent. " * 10,
    )
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))
    monkeypatch.setattr(cli, "default_embedder", lambda: failing_embedder)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=failing_embedder)
    return workspace, db_path


def test_build_parser_requires_a_command(capsys):
    assert cli.main([]) == 1
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_cli_search_json_reports_degraded_mode(cli_env, capsys):
    exit_code = cli.main(["search", "cache invalidation rules", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["degraded"] is True
    assert payload["degraded_reason"]
    assert payload["mode"] == db.MODE_LEXICAL
    assert len(payload["results"]) == 1
    assert payload["results"][0]["citation"].startswith("repo-a/AGENTS.md#")


def test_cli_search_never_touches_the_real_embedder_module_reference(cli_env, capsys, monkeypatch):
    # Regression: cmd_search used to resolve its embedder via
    # corpusdex.search's own imported default_embedder, so patching
    # cli.default_embedder alone did not isolate it from the network. Prove
    # isolation the other way: break corpusdex.search.default_embedder so
    # it would raise if cmd_search ever reached it, and confirm search still
    # succeeds (using only the cli.py-level patch already in cli_env).
    from corpusdex import search as search_mod

    def _boom():
        raise AssertionError("cmd_search must not resolve default_embedder via corpusdex.search")

    monkeypatch.setattr(search_mod, "default_embedder", _boom)
    exit_code = cli.main(["search", "cache invalidation rules", "--json"])
    assert exit_code == 0


def test_cli_search_text_output_shows_mode_and_citation(cli_env, capsys):
    exit_code = cli.main(["search", "cache invalidation rules"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "degraded" in out
    assert "repo-a/AGENTS.md#" in out


def test_cli_search_no_results(cli_env, capsys):
    exit_code = cli.main(["search", "nonexistent-term-xyz"])
    assert exit_code == 0
    assert "no results" in capsys.readouterr().out


def test_cli_get_round_trips_a_chunk(cli_env, capsys):
    cli.main(["search", "cache invalidation rules", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ref = payload["results"][0]["ref"]

    exit_code = cli.main(["get", ref, "--json"])
    assert exit_code == 0
    got = json.loads(capsys.readouterr().out)
    assert got["ref"] == ref
    assert "body" in got and "derived cache consistent" in got["body"]


def test_cli_output_never_publishes_the_unstable_rowid(cli_env, capsys):
    """A client cannot store what it is never shown.

    The rowid is real and useful internally, but any surface that emits it
    invites a caller to keep it and hand it back after a reindex has moved
    it, which is the silent-wrong-content path in issue #23.
    """
    cli.main(["search", "cache invalidation rules", "--json"])
    hit = json.loads(capsys.readouterr().out)["results"][0]
    assert "chunk_id" not in hit
    assert hit["ref"].startswith(db.CHUNK_REF_PREFIX)


def test_cli_get_missing_ref_errors(cli_env, capsys):
    exit_code = cli.main(["get", "cdeadbeefdeadbeef"])
    assert exit_code == 1
    assert "no chunk" in capsys.readouterr().err


def test_cli_get_rejects_a_rowid_with_the_reason(cli_env, capsys):
    """A numeric argument fails with why, not with "not found".

    "no chunk with ref 42" would read as "that chunk was deleted" and send
    the caller looking for the wrong problem.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["get", "999999"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "not a stable reference" in err
    assert "brain search" in err


def test_cli_recent_lists_indexed_chunks(cli_env, capsys):
    exit_code = cli.main(["recent", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["repo"] == "repo-a"


def test_cli_reindex_reports_counts(cli_env, capsys):
    exit_code = cli.main(["reindex"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "added" in out
    assert "unchanged" in out


def test_cli_reindex_json(cli_env, capsys):
    exit_code = cli.main(["reindex", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["added"] == 0  # cli_env's fixture already indexed once
    assert payload["unchanged"] == payload["docs_seen"]
    assert "embedding_available" in payload
    assert "fully_embedded" in payload


def test_cli_reindex_never_touches_the_real_embedder_module_reference(cli_env, capsys, monkeypatch):
    # Same regression class as the search test, for the reindex path: it
    # used to resolve its embedder via corpusdex.indexer's own imported
    # default_embedder, bypassing a cli.default_embedder patch entirely.
    from corpusdex import indexer as indexer_mod

    def _boom():
        raise AssertionError(
            "cmd_reindex must not resolve default_embedder via corpusdex.indexer"
        )

    monkeypatch.setattr(indexer_mod, "default_embedder", _boom)
    exit_code = cli.main(["reindex"])
    assert exit_code == 0


def test_cli_search_rejects_non_positive_n(cli_env, capsys):
    with pytest.raises(SystemExit):
        cli.main(["search", "anything", "-n", "0"])
    with pytest.raises(SystemExit):
        cli.main(["search", "anything", "-n", "-3"])


def test_cli_recent_rejects_non_positive_n(cli_env, capsys):
    with pytest.raises(SystemExit):
        cli.main(["recent", "-n", "0"])


def test_cli_status_json_reports_schema_and_counts(cli_env, capsys):
    exit_code = cli.main(["status", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"] == 1
    assert payload["chunks"] >= 1
    assert payload["code_schema_version"] == db.SCHEMA_VERSION
    assert payload["embed_backend_live"] == "unavailable"


def test_cli_reports_non_loopback_ollama_host_as_a_clean_error(cli_env, capsys, monkeypatch):
    # The real default_embedder() (not the cli_env stub) must be exercised
    # here, since the loopback refusal lives in OllamaEmbedder.__init__.
    from corpusdex import cli as cli_mod
    from corpusdex.embedder import default_embedder as real_default_embedder

    monkeypatch.setattr(cli_mod, "default_embedder", real_default_embedder)
    monkeypatch.setenv("BRAIN_OLLAMA_HOST", "http://evil.example.com:11434")

    exit_code = cli.main(["search", "anything"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "loopback" in err
    assert "Traceback" not in err


def test_cli_status_reports_cleanly_when_index_was_never_built(tmp_path, monkeypatch, capsys):
    # A read surface opened on a fresh checkout (no `brain reindex` ever run)
    # must not silently create an empty index: it should fail with a clean
    # one-line error, and must not leave a DB file behind as a side effect.
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# Repo A\n\n" + "text " * 20)
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))

    exit_code = cli.main(["status"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "brain reindex" in err
    assert "Traceback" not in err
    assert not db_path.exists()


def test_cli_search_reports_cleanly_when_index_was_never_built(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# Repo A\n\n" + "text " * 20)
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))

    exit_code = cli.main(["search", "anything"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert not db_path.exists()


def test_cli_reports_schema_version_mismatch_as_a_clean_error(cli_env, capsys):
    from corpusdex import db

    _workspace, db_path = cli_env
    conn, _vec_ok = db.open_index(db_path)
    with conn:
        db.set_meta(conn, db.META_SCHEMA_VERSION, str(db.SCHEMA_VERSION + 1))
    conn.close()

    exit_code = cli.main(["status"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "brain reindex" in err
    assert "Traceback" not in err


@pytest.fixture
def linked_cli_env(tmp_path: Path, monkeypatch, failing_embedder):
    """A CLI environment whose corpus has a supersedence chain and a link."""
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    filler = "notes about the watering schedule for the season. " * 6
    _write(
        workspace / "repo-a" / "docs" / "guide.md",
        f"---\nsuperseded_by: revised\n---\n\n# Guide\n\n## Schedule\n\n"
        f"watering {filler} see [[appendix]].\n",
    )
    _write(workspace / "repo-a" / "docs" / "appendix.md", f"# Appendix\n\n## Body\n\n{filler}\n")
    _write(workspace / "repo-a" / "docs" / "revised.md", f"# Revised\n\n## Body\n\n{filler}\n")
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))
    monkeypatch.setattr(cli, "default_embedder", lambda: failing_embedder)
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=failing_embedder)
    return workspace, db_path


def test_cli_search_degraded_line_does_not_claim_lexical_only(linked_cli_env, capsys):
    # The graph channel still contributes when the vector channel is down, so
    # the human-readable line must not say results are lexical-only. The label
    # itself now carries this, rather than a sentence appended to correct it.
    assert cli.main(["search", "watering"]) == 0
    out = capsys.readouterr().out
    assert "degraded:" in out
    # Assert the whole label, not a substring of it: "lexical" is a prefix of
    # every mode that includes the lexical channel, so a substring check here
    # would pass on the very value this test exists to reject.
    assert out.splitlines()[0].startswith(f"mode: {db.MODE_LEXICAL_GRAPH}  degraded:")


def test_cli_search_json_reports_the_channels_behind_the_mode(linked_cli_env, capsys):
    # mode is a summary; channels_used is the exact set. Asserting both here
    # is what makes a future divergence between them a test failure rather
    # than a silently wrong label in a payload nobody re-reads.
    assert cli.main(["search", "watering", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == db.MODE_LEXICAL_GRAPH
    assert payload["channels_used"] == ["graph", "lexical"]
    assert payload["degraded"] is True


def test_cli_search_shows_related_context(linked_cli_env, capsys):
    assert cli.main(["search", "watering"]) == 0
    out = capsys.readouterr().out
    assert "related:" in out
    assert "superseded_by" in out


def test_cli_get_shows_related_context(linked_cli_env, capsys):
    assert cli.main(["search", "watering", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    guide = next(hit for hit in results if hit["path"].endswith("guide.md"))

    assert cli.main(["get", guide["ref"]]) == 0
    out = capsys.readouterr().out
    assert "related: superseded_by" in out
    assert "revised.md" in out
    assert "related: links_to" in out


def test_cli_recent_shows_related_context(linked_cli_env, capsys):
    assert cli.main(["recent"]) == 0
    out = capsys.readouterr().out
    assert "related:" in out


# ---------------------------------------------------------------------------
# A broken supersedence claim is surfaced at reindex (issue #21)
# ---------------------------------------------------------------------------


def test_reindex_names_a_document_whose_superseded_by_resolves_to_nothing(
    tmp_path: Path, monkeypatch, capsys, failing_embedder
):
    """The warning must carry the PATH, not just a count.

    Issue #21's complaint is that nothing could say which document was
    affected. A count would restate the complaint rather than answer it, so
    the assertion is on the path and would survive no summarisation.
    """
    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    filler = "notes about the watering schedule for the season. " * 6
    _write(
        workspace / "repo-a" / "docs" / "retired.md",
        f"---\nsuperseded_by: nowhere-at-all.md\n---\n\n# Retired\n\n## Body\n\n{filler}\n",
    )
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))
    monkeypatch.setattr(cli, "default_embedder", lambda: failing_embedder)

    assert cli.main(["reindex"]) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "repo-a/docs/retired.md" in out

    payload_env = cli.main(["reindex", "--json"])
    assert payload_env == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["superseded_by_unresolved"] == ["repo-a/docs/retired.md"]


def test_reindex_is_silent_when_every_supersedence_resolves(linked_cli_env, capsys):
    """No warning on a healthy corpus.

    Asserted because a report that always fires teaches its reader to skip it,
    which is the same silence the report was added to break.
    """
    assert cli.main(["reindex"]) == 0
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "superseded_by but it resolves" not in out


def test_status_flags_a_model_the_index_was_not_built_with(
    tmp_path, monkeypatch, vec_probe
):
    """`fully_embedded: true` must not be reported for a configuration the
    index cannot serve.

    Two distinct failures hide behind the same "every chunk has a vector"
    count. A different WIDTH makes search degrade; a different MODEL of the
    same width is worse, because queries are compared against vectors from
    another embedding space and the rankings are quietly meaningless with no
    error anywhere. Status is the surface that has both facts.
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    from conftest import StubEmbedder

    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# A\n\n## S\n\n" + "content here. " * 20)
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))

    built_with = StubEmbedder(model="model-one")
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=built_with)

    monkeypatch.setattr(cli, "default_embedder", lambda: StubEmbedder(model="model-one"))
    payload = cli.status_payload()
    assert payload["embed_dim"] == db.EMBED_DIM
    assert payload["embed_config_stale"] is None
    assert payload["fully_embedded"] is True

    # Same width, different model: no error would ever be raised anywhere, so
    # it has to be compared for. Search now refuses its vector channel on the
    # same predicate (db.stale_embed_model); status is the surface that says
    # so before a query is run, not the only one that notices.
    monkeypatch.setattr(cli, "default_embedder", lambda: StubEmbedder(model="model-two"))
    payload = cli.status_payload()
    assert payload["fully_embedded"] is False
    assert any("model-two" in reason for reason in payload["embed_config_stale"])

    # Different width: the case that makes search degrade.
    monkeypatch.setattr(
        cli, "default_embedder", lambda: StubEmbedder(dim=384, model="model-one")
    )
    payload = cli.status_payload()
    assert payload["fully_embedded"] is False
    assert any("384" in reason for reason in payload["embed_config_stale"])


def test_status_flags_a_stale_model_even_when_the_backend_is_unreachable(
    tmp_path, monkeypatch, vec_probe
):
    """The case that matters most is the one a backend-health guard hid.

    Switching to a model that has not been pulled yet makes the readiness
    probe fail, and the staleness block used to sit entirely behind that
    probe. Status then reported no stale configuration for exactly the setup
    ``brain search`` was already refusing to use its vector channel on. A
    model comparison needs no backend: the name is configuration, not a
    measurement. A width comparison does, and stays behind the probe.
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    from conftest import FailingEmbedder, StubEmbedder

    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    _write(workspace / "repo-a" / "AGENTS.md", "# A\n\n## S\n\n" + "content here. " * 20)
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))

    indexer.reindex(
        db_path=db_path, workspace_root=workspace, embedder=StubEmbedder(model="model-one")
    )

    monkeypatch.setattr(cli, "default_embedder", lambda: FailingEmbedder())
    payload = cli.status_payload()

    assert payload["embed_backend_live"] == "unavailable"
    assert payload["embed_config_stale"] is not None
    assert any("model-one" in reason for reason in payload["embed_config_stale"])
    assert any("unreachable-model" in reason for reason in payload["embed_config_stale"])
    assert payload["fully_embedded"] is False


def test_cli_search_prints_partial_vector_coverage(tmp_path, monkeypatch, capsys, vec_probe):
    """A number nobody renders is the same silence the issue described.

    The half-embedded index answered `mode: lexical+vector` with
    `degraded: false`, so the page carried no sign that the vector channel was
    ranking within a subset. Both surfaces have to say so: the JSON field for
    programmatic callers, the status line for a person reading a terminal.
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    from conftest import StubEmbedder

    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    for i in range(4):
        # Under docs/, not at the repo root: the indexer walks docs/ plus a
        # fixed set of root filenames, so a doc-0.md at the root is not corpus
        # at all and this fixture would index nothing (issue #34).
        _write(
            workspace / "repo-a" / "docs" / f"doc-{i}.md",
            f"# Doc {i}\n\n## Cache invalidation rules\n\n" + "keep the cache consistent. " * 10,
        )
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))
    monkeypatch.setattr(cli, "default_embedder", lambda: StubEmbedder())
    indexer.reindex(db_path=db_path, workspace_root=workspace, embedder=StubEmbedder())

    # Strip vectors from all but one chunk, which is exactly the state an index
    # grown while the backend was down is left in.
    conn, _vec_ok = db.open_index(db_path)
    try:
        keep = conn.execute("SELECT MIN(id) AS id FROM chunks").fetchone()["id"]
        with conn:
            conn.execute("DELETE FROM vec_chunks WHERE chunk_id != ?", (keep,))
        remaining = conn.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()["n"]
        total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    finally:
        conn.close()
    assert remaining == 1
    assert total > 1

    assert cli.main(["search", "cache invalidation rules", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["vector_coverage"] == pytest.approx(remaining / total)
    assert payload["degraded"] is False

    assert cli.main(["search", "cache invalidation rules"]) == 0
    out = capsys.readouterr().out
    assert "partial vectors" in out
    assert "reindex" in out


def test_cli_reindex_says_when_it_rebuilt_and_why(tmp_path, monkeypatch, capsys, vec_probe):
    """The counters are misleading without this line, so it has to be printed.

    On a rebuild every document counts as `added` even when the corpus is
    identical, so `+4 added` is a fact about a fresh index rather than about
    the corpus. A reader who sees only the counters concludes four documents
    appeared.
    """
    if not vec_probe:
        pytest.skip("sqlite-vec extension does not load in this environment")
    from conftest import StubEmbedder

    workspace = tmp_path / "workspace"
    db_path = tmp_path / "var" / "index.db"
    for i in range(4):
        _write(
            workspace / "repo-a" / "docs" / f"doc-{i}.md",
            f"# Doc {i}\n\n## Cache invalidation rules\n\n" + "keep the cache consistent. " * 10,
        )
    write_registry(workspace, ["repo-a"])
    monkeypatch.setenv("BRAIN_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("BRAIN_DB", str(db_path))

    monkeypatch.setattr(cli, "default_embedder", lambda: StubEmbedder(model="model-a"))
    assert cli.main(["reindex"]) == 0
    capsys.readouterr()

    # A second run under a different model name: nothing about the corpus
    # changed, so the rebuild must be stated rather than inferred.
    monkeypatch.setattr(cli, "default_embedder", lambda: StubEmbedder(model="model-b"))
    assert cli.main(["reindex"]) == 0
    out = capsys.readouterr().out
    assert "rebuilt the index" in out
    assert "model-a" in out
    assert "model-b" in out
    # Ordering matters: the explanation has to reach the reader before the
    # counters it explains.
    assert out.index("rebuilt the index") < out.index("added")

    assert cli.main(["reindex", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # Third run, same model, nothing to rebuild.
    assert payload["rebuilt"] is False
    assert payload["rebuild_reason"] is None
