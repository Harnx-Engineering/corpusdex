"""``brain`` command-line entry point.

Subcommands: ``search``, ``get``, ``recent``, ``reindex``, ``status``. Each
reads/writes the single index database at :func:`corpusdex.db.default_db_path`
(override with ``BRAIN_DB``) over a corpus scoped by
:func:`corpusdex.indexer.discover_corpus` (see ``BRAIN_WORKSPACE_ROOT`` and
``BRAIN_CORPUS_ROOTS``).

Every command resolves its embedder through this module's own
``default_embedder`` name (imported below and never re-resolved through
another module), so tests and callers can substitute an embedder for every
subcommand by monkeypatching a single symbol: ``corpusdex.cli.default_embedder``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import db
from . import evaluate as evaluate_mod
from . import search as search_mod
from .embedder import EmbeddingUnavailable, default_embedder
from .indexer import RegistryInvalid, RegistryMissing
from .indexer import reindex as run_reindex

# Exceptions any subcommand may raise that should print as a clean one-line
# error rather than a Python traceback: a held write lock, a missing or
# unsafe repo registry, an index that has never been built, an index built
# by a different schema version, a path the caller must configure because we
# were installed rather than run from a checkout, or a rejected (non-loopback)
# embedding host configuration.
_CLEAN_ERRORS = (
    db.IndexLocked,
    db.SchemaVersionMismatch,
    db.IndexMissing,
    db.NotConfigured,
    RegistryMissing,
    RegistryInvalid,
    ValueError,
)


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}") from None
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


def _related_line(hit: search_mod.SearchHit, limit: int | None = None) -> str | None:
    """One compact ``related:`` line summarising a hit's assembled context."""
    refs = hit.assembled if limit is None else hit.assembled[:limit]
    if not refs:
        return None
    parts = [f"{ref.relation} {ref.path}" for ref in refs]
    suffix = ""
    if limit is not None and len(hit.assembled) > limit:
        suffix = f" (+{len(hit.assembled) - limit} more)"
    return f"   related: {'; '.join(parts)}{suffix}"


def _hit_dict(hit: search_mod.SearchHit, *, full_body: bool = False) -> dict:
    payload = asdict(hit)
    if not full_body:
        payload.pop("body", None)
    # The rowid is deliberately not published. It is reassigned by every
    # reindex, and a client that can see it will store it and hand it back
    # later, which is precisely the silent-wrong-content path ``ref`` exists
    # to close (issue #23).
    payload.pop("chunk_id", None)
    payload["citation"] = hit.citation
    return payload


def _chunk_ref_arg(value: str) -> str:
    """Accept a stable chunk ref, and reject an old-style rowid explicitly.

    An integer here is almost always a ref copied from a pre-ref search
    result or a habit carried over from the previous interface. Silently
    looking it up is what issue #23 is about, so it fails with the reason
    rather than with "not found", which would read as "that chunk is gone".
    """
    if value.lstrip("+-").isdigit():
        raise argparse.ArgumentTypeError(
            f"{value!r} looks like a chunk rowid, which is not a stable reference: "
            "reindexing reassigns it, so it can resolve to unrelated content. "
            "Use the `ref` value from a `brain search` result instead."
        )
    return value


def cmd_search(args: argparse.Namespace) -> int:
    embedder = default_embedder()
    conn, vec_ok = db.open_index()
    try:
        response = search_mod.search(conn, vec_ok, args.query, limit=args.n, embedder=embedder)
    finally:
        conn.close()

    if args.json:
        print(
            json.dumps(
                {
                    "query": response.query,
                    "mode": response.mode,
                    "channels_used": sorted(response.channels_used),
                    "degraded": response.degraded,
                    "degraded_reason": response.degraded_reason,
                    "results": [_hit_dict(hit) for hit in response.hits],
                },
                indent=2,
            )
        )
        return 0

    status_line = f"mode: {response.mode}"
    if response.degraded:
        reason = response.degraded_reason or "vector search unavailable"
        # `mode` now names the channels that ran, so this no longer has to
        # correct it; the reason is the part the mode label cannot carry.
        status_line += f"  degraded: {reason}"
    print(status_line)
    if not response.hits:
        print("no results")
        return 0
    for i, hit in enumerate(response.hits, start=1):
        flags = []
        if hit.superseded_by:
            flags.append(f"superseded_by={hit.superseded_by}")
        if hit.decided_on:
            flags.append(f"decided_on={hit.decided_on}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"{i}. {hit.citation}  score={hit.score}  ref={hit.ref}{flag_str}")
        print(f"   {hit.snippet}")
        related = _related_line(hit, limit=2)
        if related:
            print(related)
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    conn, _vec_ok = db.open_index()
    try:
        hit = search_mod.get_chunk(conn, args.ref)
    finally:
        conn.close()

    if hit is None:
        print(
            f"no chunk with ref {args.ref}; the section it named may have been "
            "renamed or removed since the search that produced it",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(_hit_dict(hit, full_body=True), indent=2))
        return 0

    print(hit.citation)
    print(f"title: {hit.title}  doc_type: {hit.doc_type}  ref: {hit.ref}")
    if hit.decided_on:
        print(f"decided_on: {hit.decided_on}")
    if hit.superseded_by:
        print(f"superseded_by: {hit.superseded_by}")
    if hit.tags:
        print(f"tags: {hit.tags}")
    for ref in hit.assembled:
        print(f"related: {ref.relation} {ref.path}  (doc {ref.doc_id}: {ref.title})")
    print()
    print(hit.body)
    return 0


def cmd_recent(args: argparse.Namespace) -> int:
    conn, _vec_ok = db.open_index()
    try:
        hits = search_mod.recent(conn, limit=args.n)
    finally:
        conn.close()

    if args.json:
        print(json.dumps([_hit_dict(hit) for hit in hits], indent=2))
        return 0

    if not hits:
        print("no results")
        return 0
    for i, hit in enumerate(hits, start=1):
        print(f"{i}. {hit.citation}  ref={hit.ref}")
        print(f"   {hit.snippet}")
        related = _related_line(hit, limit=2)
        if related:
            print(related)
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    embedder = default_embedder()
    stats = run_reindex(full=args.full, embedder=embedder)

    if args.json:
        print(
            json.dumps(
                {
                    "docs_seen": stats.docs_seen,
                    "added": stats.added,
                    "changed": stats.changed,
                    "removed": stats.removed,
                    "unchanged": stats.unchanged,
                    "chunks_written": stats.chunks_written,
                    "embedded_chunks": stats.embedded_chunks,
                    "vector_total": stats.vector_total,
                    "vector_covered": stats.vector_covered,
                    "embedding_available": stats.embedding_available,
                    "fully_embedded": stats.fully_embedded,
                    "link_edges": stats.link_edges,
                    "link_targets_unresolved": stats.link_targets_unresolved,
                    "link_targets_unlinkable": stats.link_targets_unlinkable,
                    "link_targets_from_work_logs": stats.link_targets_from_work_logs,
                    "superseded_by_unresolved": list(stats.superseded_by_unresolved),
                    "duration_seconds": round(stats.duration_seconds, 3),
                    "errors": stats.errors,
                },
                indent=2,
            )
        )
        return 0

    print(
        f"scanned {stats.docs_seen} docs: "
        f"+{stats.added} added, {stats.changed} changed, "
        f"{stats.removed} removed, {stats.unchanged} unchanged"
    )
    print(
        f"chunks: {stats.vector_total} total, {stats.vector_covered} embedded "
        f"({stats.embedded_chunks} newly backfilled this run)"
    )
    # Two numbers, because they call for opposite responses: "unresolved"
    # names documents the corpus refers to and does not contain, which someone
    # can close, while "declined" is the resolver refusing to guess between
    # candidates or to link a document to itself, which is it working.
    print(
        f"link graph: {stats.link_edges} edges "
        f"({stats.link_targets_unresolved} targets name no document, "
        f"{stats.link_targets_unlinkable} declined as ambiguous or self, "
        f"{stats.link_targets_from_work_logs} declined as work-log citations)"
    )
    if stats.superseded_by_unresolved:
        # Named, not counted. The ranking penalty comes from the resolved
        # edge, so each of these is a document asserting it was replaced while
        # being ranked and displayed as though it never was; a bare number
        # would leave the reader with no way to find out which.
        print(
            f"WARNING: {len(stats.superseded_by_unresolved)} document(s) declare "
            f"superseded_by but it resolves to no document, so no supersedence "
            f"penalty or successor applies:"
        )
        for path in stats.superseded_by_unresolved:
            print(f"  {path}")
    embed_state = "ready" if stats.fully_embedded else "not fully embedded"
    if not stats.embedding_available:
        embed_state = "unavailable (indexed lexical-only)"
    print(f"embedding backend: {embed_state}")
    for error in stats.errors:
        print(f"warning: {error}", file=sys.stderr)
    print(f"took {stats.duration_seconds:.2f}s")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    embedder = default_embedder()
    conn, vec_ok = db.open_index()
    try:
        report = evaluate_mod.evaluate(
            conn,
            vec_ok,
            queries_path=Path(args.queries) if args.queries else None,
            k=args.k,
            embedder=embedder,
        )
    finally:
        conn.close()

    payload = evaluate_mod.report_payload(report, per_query=args.per_query)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{report.query_count} judged queries, k={report.k}")
    print(f"{'arm':<16} {'recall@k':>9} {'mrr':>7}  channels")
    for arm in report.arms:
        if not arm.available:
            print(f"{arm.name:<16} {'-':>9} {'-':>7}  not run: {arm.unavailable_reason}")
            continue
        channels = "+".join(sorted(arm.channels))
        print(f"{arm.name:<16} {arm.recall:>9.3f} {arm.mrr:>7.3f}  {channels}")

    delta = report.graph_delta()
    if delta is None:
        # Saying "no contribution" here would be a claim the run cannot
        # support: the comparison arm never executed.
        print("\ngraph channel contribution: not measurable in this run")
    else:
        print(
            f"\ngraph channel contribution (full vs lexical+vector): "
            f"recall {delta['recall']:+.3f}, mrr {delta['mrr']:+.3f}"
        )
    for note in report.notes:
        print(f"note: {note}")

    if args.per_query:
        for arm in report.arms:
            if not arm.available:
                continue
            print(f"\n[{arm.name}]")
            for score in arm.scores:
                missed = f"  missed: {', '.join(score.missed)}" if score.missed else ""
                print(
                    f"  {score.id:<34} recall={score.recall:.2f} "
                    f"rr={score.reciprocal_rank:.2f}{missed}"
                )
    return 0


def status_payload(db_path: Path | None = None) -> dict:
    """Compute the index-freshness and embedding-backend-health fields.

    Shared by ``cmd_status`` (the ``brain status`` subcommand) and the MCP
    server's ``brain_status`` tool (see :mod:`corpusdex.mcp_server`), so
    both surfaces report exactly the same fields from exactly the same
    queries rather than maintaining two copies that could drift apart.
    """
    resolved_db_path = db_path if db_path is not None else db.default_db_path()
    conn, vec_ok = db.open_index(resolved_db_path)
    try:
        doc_count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        embedded_count = 0
        if vec_ok and db.has_vec_table(conn):
            embedded_count = conn.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()["n"]
        schema_version = db.get_meta(conn, db.META_SCHEMA_VERSION)
        last_reindex = db.get_meta(conn, db.META_LAST_REINDEX)
        stored_embed_status = db.get_meta(conn, db.META_EMBED_STATUS)
        embed_model = db.get_meta(conn, db.META_EMBED_MODEL)
        index_dim = db.vec_table_dim(conn) if vec_ok else None
    finally:
        conn.close()

    embedder = default_embedder()
    live_dim = None
    try:
        # dimension() rather than probe(): status has to answer both "is the
        # backend up" and "is what it produces the width this index holds",
        # and one round trip answers both.
        live_dim = embedder.dimension()
    except EmbeddingUnavailable as exc:
        embed_live = "unavailable"
        embed_live_detail = str(exc)
    else:
        embed_live = "ready"
        embed_live_detail = None

    # A configuration the index cannot serve, reported before it is hit rather
    # than as a failed search. Both halves matter: a different MODEL of the
    # same width silently compares new queries against old document vectors,
    # which produces plausible but meaningless rankings and no error anywhere.
    stale_reasons = []
    if embed_live == "ready":
        if index_dim is not None and live_dim and live_dim != index_dim:
            stale_reasons.append(
                f"index holds {index_dim}-dimension vectors, active model "
                f"{embedder.model!r} produces {live_dim}"
            )
        if embed_model and embed_model != embedder.model:
            stale_reasons.append(
                f"index was embedded with {embed_model!r}, active model is "
                f"{embedder.model!r}"
            )

    fully_embedded = (
        vec_ok
        and embed_live == "ready"
        and embedded_count >= chunk_count
        and not stale_reasons
    )

    return {
        "db_path": str(resolved_db_path),
        "documents": doc_count,
        "chunks": chunk_count,
        "embedded_chunks": embedded_count,
        "fully_embedded": fully_embedded,
        "schema_version": schema_version,
        "code_schema_version": db.SCHEMA_VERSION,
        "last_reindex_at": last_reindex,
        "vector_extension_loaded": vec_ok,
        "embed_model": embed_model or None,
        "embed_dim": index_dim,
        "embed_config_stale": stale_reasons or None,
        "embed_backend_stored_status": stored_embed_status,
        "embed_backend_live": embed_live,
        "embed_backend_detail": embed_live_detail,
    }


def cmd_status(args: argparse.Namespace) -> int:
    payload = status_payload()

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for key, value in payload.items():
        if value is None:
            continue
        print(f"{key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain", description="Hybrid local retrieval over a Markdown corpus"
    )
    sub = parser.add_subparsers(dest="command")

    p_search = sub.add_parser("search", help="hybrid search over the index")
    p_search.add_argument("query", help="free-text query")
    p_search.add_argument(
        "-n", type=_positive_int, default=10, help="max results, must be positive (default 10)"
    )
    p_search.add_argument("--json", action="store_true", help="emit JSON")
    p_search.set_defaults(func=cmd_search)

    p_get = sub.add_parser("get", help="show a chunk's full text by its stable ref")
    p_get.add_argument(
        "ref",
        type=_chunk_ref_arg,
        help="the `ref` value from a search result (stable across reindex)",
    )
    p_get.add_argument("--json", action="store_true", help="emit JSON")
    p_get.set_defaults(func=cmd_get)

    p_recent = sub.add_parser("recent", help="most recently indexed chunks")
    p_recent.add_argument(
        "-n", type=_positive_int, default=10, help="max results, must be positive (default 10)"
    )
    p_recent.add_argument("--json", action="store_true", help="emit JSON")
    p_recent.set_defaults(func=cmd_recent)

    p_reindex = sub.add_parser("reindex", help="incrementally reindex the workspace docs corpus")
    p_reindex.add_argument(
        "--full", action="store_true", help="re-chunk every document regardless of its fingerprint"
    )
    p_reindex.add_argument("--json", action="store_true", help="emit JSON")
    p_reindex.set_defaults(func=cmd_reindex)

    p_eval = sub.add_parser(
        "eval", help="score retrieval against the judged query set, with channel ablation"
    )
    p_eval.add_argument(
        "-k", type=_positive_int, default=evaluate_mod.DEFAULT_K, help="cutoff rank (default 10)"
    )
    p_eval.add_argument(
        "--queries", default=None, help="path to a query set (default eval/queries.yaml)"
    )
    p_eval.add_argument(
        "--per-query", action="store_true", help="also print each query's score and misses"
    )
    p_eval.add_argument("--json", action="store_true", help="emit JSON")
    p_eval.set_defaults(func=cmd_eval)

    p_status = sub.add_parser("status", help="index freshness and embedding backend health")
    p_status.add_argument("--json", action="store_true", help="emit JSON")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except _CLEAN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
