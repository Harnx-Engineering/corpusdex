from __future__ import annotations

from pathlib import Path

import pytest

from corpusdex import db, evaluate, search


def _write_queries(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _insert_doc(conn, *, path: str, body: str, heading: str = "Title > Section") -> None:
    with conn:
        cur = conn.execute(
            "INSERT INTO documents (repo, path, title, doc_type, mtime, content_hash) "
            "VALUES ('repo', ?, 'Title', 'doc', 1.0, ?)",
            (path, f"hash-{path}"),
        )
        conn.execute(
            "INSERT INTO chunks (ref, doc_id, heading_path, body) VALUES (?, ?, ?, ?)",
            (db.chunk_ref(path, heading, 0), cur.lastrowid, heading, body),
        )


# ---------------------------------------------------------------------------
# query set parsing
# ---------------------------------------------------------------------------


def test_load_queries_parses_ids_and_judgements(tmp_path: Path):
    spec = _write_queries(
        tmp_path / "q.yaml",
        "queries:\n"
        "  - id: a\n    query: alpha\n    relevant: [one.md, two.md]\n"
        "  - id: b\n    query: beta\n    relevant: [three.md]\n",
    )
    judgements = evaluate.load_queries(spec)
    assert [j.id for j in judgements] == ["a", "b"]
    assert judgements[0].relevant == frozenset({"one.md", "two.md"})


def test_load_queries_rejects_a_duplicate_id(tmp_path: Path):
    spec = _write_queries(
        tmp_path / "q.yaml",
        "queries:\n"
        "  - id: a\n    query: alpha\n    relevant: [one.md]\n"
        "  - id: a\n    query: other\n    relevant: [two.md]\n",
    )
    with pytest.raises(evaluate.JudgementError, match="duplicate query id"):
        evaluate.load_queries(spec)


def test_load_queries_rejects_a_query_with_no_judgements(tmp_path: Path):
    """An unjudged query scores 0 against everything and drags the mean down."""
    spec = _write_queries(
        tmp_path / "q.yaml", "queries:\n  - id: a\n    query: alpha\n    relevant: []\n"
    )
    with pytest.raises(evaluate.JudgementError, match="judges no relevant documents"):
        evaluate.load_queries(spec)


def test_load_queries_rejects_a_missing_file(tmp_path: Path):
    with pytest.raises(evaluate.JudgementError, match="no query set at"):
        evaluate.load_queries(tmp_path / "absent.yaml")


# ---------------------------------------------------------------------------
# judgement validation against the corpus
# ---------------------------------------------------------------------------


def test_unknown_judged_document_is_an_error_not_a_miss(lexical_conn, tmp_path: Path):
    """A renamed document must not read as a recall regression.

    Scored as a miss it is indistinguishable from the ranker getting worse,
    which sends the reader after the wrong problem entirely.
    """
    _insert_doc(lexical_conn, path="present.md", body="content about widgets")
    spec = _write_queries(
        tmp_path / "q.yaml",
        "queries:\n  - id: a\n    query: widgets\n    relevant: [present.md, renamed-away.md]\n",
    )
    judgements = evaluate.load_queries(spec)
    with pytest.raises(evaluate.JudgementError) as excinfo:
        evaluate.validate_judgements(lexical_conn, judgements)
    message = str(excinfo.value)
    assert "renamed-away.md" in message
    assert "present.md" not in message


def test_validation_passes_when_every_judgement_resolves(lexical_conn, tmp_path: Path):
    _insert_doc(lexical_conn, path="present.md", body="content about widgets")
    spec = _write_queries(
        tmp_path / "q.yaml", "queries:\n  - id: a\n    query: widgets\n    relevant: [present.md]\n"
    )
    evaluate.validate_judgements(lexical_conn, evaluate.load_queries(spec))


# ---------------------------------------------------------------------------
# scoring math
# ---------------------------------------------------------------------------


def test_recall_counts_distinct_judged_documents_within_k():
    judgement = evaluate.Judgement(id="a", query="q", relevant=frozenset({"x.md", "y.md"}))
    score = evaluate._score_query(judgement, ["x.md", "other.md", "z.md"], 10)
    assert score.recall == 0.5
    assert score.found == ("x.md",)
    assert score.missed == ("y.md",)


def test_reciprocal_rank_uses_the_first_relevant_position():
    judgement = evaluate.Judgement(id="a", query="q", relevant=frozenset({"y.md"}))
    score = evaluate._score_query(judgement, ["other.md", "y.md"], 10)
    assert score.reciprocal_rank == pytest.approx(0.5)


def test_a_relevant_document_below_k_does_not_count():
    """k is a cutoff, not a suggestion: rank 3 is a miss when k is 2."""
    judgement = evaluate.Judgement(id="a", query="q", relevant=frozenset({"y.md"}))
    score = evaluate._score_query(judgement, ["a.md", "b.md", "y.md"], 2)
    assert score.recall == 0.0
    assert score.reciprocal_rank == 0.0


def test_duplicate_document_paths_occupy_their_ranks():
    """Several chunks of one document really do take several result slots."""
    judgement = evaluate.Judgement(id="a", query="q", relevant=frozenset({"y.md"}))
    score = evaluate._score_query(judgement, ["a.md", "a.md", "y.md"], 3)
    assert score.reciprocal_rank == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# end to end, including the unavailable-arm contract
# ---------------------------------------------------------------------------


def test_evaluate_scores_the_offline_arms_without_an_embedder(lexical_conn, tmp_path: Path):
    _insert_doc(lexical_conn, path="target.md", body="findable content about widgets")
    _insert_doc(lexical_conn, path="noise.md", body="entirely unrelated prose")
    spec = _write_queries(
        tmp_path / "q.yaml", "queries:\n  - id: a\n    query: widgets\n    relevant: [target.md]\n"
    )

    report = evaluate.evaluate(lexical_conn, False, queries_path=spec, k=5)
    lexical = report.arm("lexical")
    assert lexical.available is True
    assert lexical.recall == 1.0
    assert lexical.mrr == 1.0


def test_an_arm_whose_channel_cannot_run_is_unavailable_not_zero(lexical_conn, tmp_path: Path):
    """"Never ran" and "scored nothing" are different findings.

    Reported as 0.0 the vector arms would read as "semantic retrieval finds
    nothing on this corpus", which is a claim the run did not test.
    """
    _insert_doc(lexical_conn, path="target.md", body="findable content about widgets")
    spec = _write_queries(
        tmp_path / "q.yaml", "queries:\n  - id: a\n    query: widgets\n    relevant: [target.md]\n"
    )

    report = evaluate.evaluate(lexical_conn, False, queries_path=spec, k=5)
    vector_arm = report.arm("lexical+vector")
    assert vector_arm.available is False
    assert vector_arm.unavailable_reason
    assert vector_arm.recall == 0.0
    assert report.notes and "not run" in report.notes[0]


def test_graph_delta_is_none_when_a_comparison_arm_did_not_run(lexical_conn, tmp_path: Path):
    """Without both arms the harness must decline to report a contribution."""
    _insert_doc(lexical_conn, path="target.md", body="findable content about widgets")
    spec = _write_queries(
        tmp_path / "q.yaml", "queries:\n  - id: a\n    query: widgets\n    relevant: [target.md]\n"
    )
    report = evaluate.evaluate(lexical_conn, False, queries_path=spec, k=5)
    assert report.graph_delta() is None
    assert evaluate.report_payload(report)["graph_channel_delta"] is None


def test_evaluate_rejects_a_non_positive_k(lexical_conn, tmp_path: Path):
    spec = _write_queries(
        tmp_path / "q.yaml", "queries:\n  - id: a\n    query: widgets\n    relevant: [t.md]\n"
    )
    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluate.evaluate(lexical_conn, False, queries_path=spec, k=0)


# ---------------------------------------------------------------------------
# the channels parameter the ablation depends on
# ---------------------------------------------------------------------------


def test_disabling_a_channel_is_not_reported_as_degradation(lexical_conn):
    """An ablation run must not look like a broken run.

    Both produce an empty vector candidate list; only one of them means
    something is wrong, and conflating them makes every ablation unreadable.
    """
    _insert_doc(lexical_conn, path="a.md", body="findable content about widgets")
    response = search.search(
        lexical_conn, False, "widgets", channels=frozenset({search.CHANNEL_LEXICAL})
    )
    assert response.degraded is False
    assert response.degraded_reason is None
    assert search.CHANNEL_VECTOR not in response.channels_used


def test_a_requested_channel_that_cannot_run_is_still_degradation(lexical_conn):
    _insert_doc(lexical_conn, path="a.md", body="findable content about widgets")
    response = search.search(
        lexical_conn,
        False,
        "widgets",
        channels=frozenset({search.CHANNEL_LEXICAL, search.CHANNEL_VECTOR}),
    )
    assert response.degraded is True
    assert response.degraded_reason


def test_unknown_channel_name_is_rejected(lexical_conn):
    with pytest.raises(ValueError, match="unknown retrieval channel"):
        search.search(lexical_conn, False, "widgets", channels=frozenset({"telepathy"}))


def test_default_search_still_requests_every_channel(lexical_conn):
    """Production behaviour is unchanged by the ablation parameter existing."""
    _insert_doc(lexical_conn, path="a.md", body="findable content about widgets")
    response = search.search(lexical_conn, False, "widgets")
    assert search.CHANNEL_LEXICAL in response.channels_used
