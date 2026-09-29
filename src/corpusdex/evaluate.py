"""Retrieval evaluation and channel ablation over a judged query set.

Answers the question the unit tests cannot: does a ranking change help. Until
this existed, any edit to fusion, boosts, or chunking was unfalsifiable, and
the specific open question was whether the link-graph channel adds recall or
only re-votes for documents the lexical and vector channels already ranked
highly (issue #17).

Three deliberate choices:

* **Relevance is judged per document, not per chunk.** Chunk boundaries move
  when the chunker changes, so chunk-level judgements would rot on a change
  that has nothing to do with retrieval quality.
* **Judgements are validated against the corpus before scoring.** A judgement
  naming a document that is no longer indexed is an error, not a zero. Scored
  as a zero it is indistinguishable from a genuine recall regression, and the
  wrong thing gets investigated.
* **The ablation runs the real ranking code** through
  :func:`corpusdex.search.search`'s ``channels`` parameter rather than
  reimplementing fusion here. An ablation with its own copy of the ranking
  measures the copy.

Arms whose channel cannot run (typically the vector channel with no local
embedding backend) are reported as unavailable rather than scored. A zero and
"never ran" are different findings and must not share a cell.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import db
from . import search as search_mod
from .embedder import OllamaEmbedder

DEFAULT_K = 10

#: The ablation arms. ``lexical`` and ``lexical+graph`` need no embedding
#: backend, so a fully offline run still measures the graph channel's
#: contribution, which is the question issue #17 was opened for.
ARMS: tuple[tuple[str, frozenset[str]], ...] = (
    ("lexical", frozenset({search_mod.CHANNEL_LEXICAL})),
    ("lexical+graph", frozenset({search_mod.CHANNEL_LEXICAL, search_mod.CHANNEL_GRAPH})),
    ("lexical+vector", frozenset({search_mod.CHANNEL_LEXICAL, search_mod.CHANNEL_VECTOR})),
    ("full", search_mod.ALL_CHANNELS),
)


class JudgementError(ValueError):
    """Raised when the query set is malformed or judges a document not in the index."""


@dataclass(frozen=True)
class Judgement:
    id: str
    query: str
    relevant: frozenset[str]


@dataclass(frozen=True)
class QueryScore:
    id: str
    query: str
    recall: float
    reciprocal_rank: float
    found: tuple[str, ...]
    missed: tuple[str, ...]


@dataclass(frozen=True)
class ArmResult:
    name: str
    channels: frozenset[str]
    available: bool
    unavailable_reason: str | None
    recall: float
    mrr: float
    scores: tuple[QueryScore, ...] = ()


@dataclass(frozen=True)
class EvalReport:
    k: int
    query_count: int
    arms: tuple[ArmResult, ...]
    notes: list[str] = field(default_factory=list)

    def arm(self, name: str) -> ArmResult | None:
        return next((a for a in self.arms if a.name == name), None)

    def graph_delta(self) -> dict[str, float] | None:
        """Recall and MRR the graph channel adds on top of lexical+vector.

        None when either arm did not run, so a missing embedding backend can
        never be reported as "the graph channel contributes nothing".
        """
        base, full = self.arm("lexical+vector"), self.arm("full")
        if base is None or full is None or not base.available or not full.available:
            return None
        return {"recall": full.recall - base.recall, "mrr": full.mrr - base.mrr}


def default_queries_path() -> Path:
    return db.repo_root() / "eval" / "queries.yaml"


def load_queries(path: Path | None = None) -> list[Judgement]:
    """Parse the judged query set, without touching the index."""
    resolved = Path(path) if path is not None else default_queries_path()
    if not resolved.is_file():
        raise JudgementError(f"no query set at {resolved}")
    raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("queries"), list):
        raise JudgementError(f"{resolved} must contain a top-level 'queries' list")

    judgements: list[Judgement] = []
    seen: set[str] = set()
    for entry in raw["queries"]:
        if not isinstance(entry, dict):
            raise JudgementError(f"{resolved}: every query entry must be a mapping")
        qid, query, relevant = entry.get("id"), entry.get("query"), entry.get("relevant")
        if not qid or not query:
            raise JudgementError(f"{resolved}: every query entry needs 'id' and 'query'")
        if qid in seen:
            raise JudgementError(f"{resolved}: duplicate query id {qid!r}")
        if not isinstance(relevant, list) or not relevant:
            raise JudgementError(f"{resolved}: query {qid!r} judges no relevant documents")
        seen.add(qid)
        judgements.append(Judgement(id=qid, query=query, relevant=frozenset(relevant)))
    if not judgements:
        raise JudgementError(f"{resolved} contains no queries")
    return judgements


def validate_judgements(conn: sqlite3.Connection, judgements: list[Judgement]) -> None:
    """Fail when a judgement names a document the index does not have.

    Deliberately fatal. Scoring an unknown path as a miss makes a renamed or
    deleted document look exactly like a ranking regression, so the harness
    would report a number that quietly means something else.
    """
    known = {row["path"] for row in conn.execute("SELECT path FROM documents")}
    unknown = sorted(
        f"{j.id} -> {path}" for j in judgements for path in j.relevant if path not in known
    )
    if unknown:
        listed = "\n  ".join(unknown)
        raise JudgementError(
            "the query set judges documents that are not in the index:\n  "
            f"{listed}\n"
            "Either the corpus moved (update the judgements) or the index is "
            "stale (run `brain reindex`)."
        )


def _score_query(judgement: Judgement, hit_paths: list[str], k: int) -> QueryScore:
    """Recall@k and reciprocal rank for one query.

    ``hit_paths`` is the ranked list of document paths behind the hits, with
    duplicates kept: a document contributing several chunks occupies several
    ranks, exactly as a reader sees it.
    """
    top = hit_paths[:k]
    found = [path for path in judgement.relevant if path in top]
    reciprocal = 0.0
    for rank, path in enumerate(top, start=1):
        if path in judgement.relevant:
            reciprocal = 1.0 / rank
            break
    return QueryScore(
        id=judgement.id,
        query=judgement.query,
        recall=len(found) / len(judgement.relevant),
        reciprocal_rank=reciprocal,
        found=tuple(sorted(found)),
        missed=tuple(sorted(judgement.relevant - set(found))),
    )


def _run_arm(
    conn: sqlite3.Connection,
    vec_ok: bool,
    judgements: list[Judgement],
    *,
    name: str,
    channels: frozenset[str],
    k: int,
    embedder: OllamaEmbedder | None,
) -> ArmResult:
    scores: list[QueryScore] = []
    for judgement in judgements:
        response = search_mod.search(
            conn, vec_ok, judgement.query, limit=k, embedder=embedder, channels=channels
        )
        if response.degraded and search_mod.CHANNEL_VECTOR in channels:
            # The arm asked for a channel that could not run, so its numbers
            # would describe a different system than the one named.
            return ArmResult(
                name=name,
                channels=channels,
                available=False,
                unavailable_reason=response.degraded_reason or "vector channel unavailable",
                recall=0.0,
                mrr=0.0,
            )
        scores.append(_score_query(judgement, [hit.path for hit in response.hits], k))

    return ArmResult(
        name=name,
        channels=channels,
        available=True,
        unavailable_reason=None,
        recall=sum(s.recall for s in scores) / len(scores),
        mrr=sum(s.reciprocal_rank for s in scores) / len(scores),
        scores=tuple(scores),
    )


def evaluate(
    conn: sqlite3.Connection,
    vec_ok: bool,
    *,
    queries_path: Path | None = None,
    k: int = DEFAULT_K,
    embedder: OllamaEmbedder | None = None,
) -> EvalReport:
    """Score every arm over the judged query set against the open index."""
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    judgements = load_queries(queries_path)
    validate_judgements(conn, judgements)

    arms = tuple(
        _run_arm(
            conn, vec_ok, judgements, name=name, channels=channels, k=k, embedder=embedder
        )
        for name, channels in ARMS
    )
    notes: list[str] = []
    unavailable = [a.name for a in arms if not a.available]
    if unavailable:
        notes.append(
            f"arms not run: {', '.join(unavailable)} "
            "(reported as unavailable, not as a zero score)"
        )
    return EvalReport(k=k, query_count=len(judgements), arms=arms, notes=notes)


def report_payload(report: EvalReport, *, per_query: bool = False) -> dict[str, Any]:
    """Serialisable form of an :class:`EvalReport`, for ``--json``."""
    payload: dict[str, Any] = {
        "k": report.k,
        "queries": report.query_count,
        "arms": [
            {
                "name": arm.name,
                "channels": sorted(arm.channels),
                "available": arm.available,
                "unavailable_reason": arm.unavailable_reason,
                "recall_at_k": round(arm.recall, 4) if arm.available else None,
                "mrr": round(arm.mrr, 4) if arm.available else None,
            }
            for arm in report.arms
        ],
        "graph_channel_delta": report.graph_delta(),
        "notes": report.notes,
    }
    delta = payload["graph_channel_delta"]
    if delta is not None:
        payload["graph_channel_delta"] = {key: round(value, 4) for key, value in delta.items()}
    if per_query:
        payload["per_query"] = {
            arm.name: [
                {
                    "id": score.id,
                    "recall": round(score.recall, 4),
                    "reciprocal_rank": round(score.reciprocal_rank, 4),
                    "missed": list(score.missed),
                }
                for score in arm.scores
            ]
            for arm in report.arms
            if arm.available
        }
    return payload
