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
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from . import adr, db
from . import evaluate as evaluate_mod
from . import search as search_mod
from .embedder import EmbeddingUnavailable, default_embedder
from .indexer import (
    CorpusRootsCollide,
    CorpusRootsUnusable,
    RegistryInvalid,
    RegistryMissing,
    corpus_scope_resolvable,
    knowledge_repo,
    knowledge_repo_in_scope,
)
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
    db.VectorTableUnreachable,
    RegistryMissing,
    RegistryInvalid,
    CorpusRootsCollide,
    CorpusRootsUnusable,
    adr.AdrError,
    # A schema violation reaching the CLI is a bug rather than a user error,
    # but a traceback is the worst way to say so: the constraint name is the
    # diagnostic and it is already in the message. Kept as a backstop, not as
    # the handling for any known cause. The one known cause, two corpus roots
    # claiming one citation path, is refused at discovery above and no longer
    # reaches an insert.
    sqlite3.IntegrityError,
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
    # A property, so asdict() does not carry it. Published rather than left
    # for each client to recompute from the other two fields: deriving it per
    # consumer is exactly how the resolved boolean came to be computed on
    # every query and read by none of them (issue #73).
    payload["supersedence_unresolved"] = hit.supersedence_unresolved
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
                    "vector_coverage": response.vector_coverage,
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
    # Printed, not merely returned. A partially embedded index answered with
    # `mode: lexical+vector` and `degraded: false`, so nothing on the page
    # looked wrong while the vector channel was ranking within whichever
    # documents happened to be embedded first. A number nobody renders is the
    # same silence.
    if response.vector_coverage is not None and response.vector_coverage < 1.0:
        percent = response.vector_coverage * 100
        status_line += (
            f"  partial vectors: {percent:.1f}% of the corpus is embedded, so the "
            "vector channel votes over that subset and its weight is reduced to "
            "match; run `brain reindex` to backfill"
        )
    print(status_line)
    if not response.hits:
        print("no results")
        return 0
    for i, hit in enumerate(response.hits, start=1):
        flags = []
        # Rendered from the resolved edge, so the line cannot contradict the
        # ranking. A claim that resolved to nothing is shown as a claim: the
        # record is treated as current everywhere else, and printing the bare
        # string here read as "retracted" (issue #73).
        if hit.is_superseded:
            flags.append(f"superseded_by={hit.superseded_by}")
        elif hit.supersedence_unresolved:
            flags.append(f"superseded_by={hit.superseded_by} (UNRESOLVED, treated as current)")
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
    if hit.is_superseded:
        print(f"superseded_by: {hit.superseded_by}")
    elif hit.supersedence_unresolved:
        print(
            f"superseded_by: {hit.superseded_by} (UNRESOLVED: no supersedence "
            "edge resolved for it, so this record is treated as current)"
        )
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
                    "rejected": stats.rejected,
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
                    "knowledge_repo": stats.knowledge_repo,
                    "reclassified": stats.reclassified,
                    "db_path": stats.db_path,
                    "rebuilt": stats.rebuilt,
                    "rebuild_reason": stats.rebuild_reason,
                    "errors": stats.errors,
                },
                indent=2,
            )
        )
        return 0

    # Printed BEFORE the counters, because it changes how they read: on a
    # rebuild every document counts as added even when none of them changed,
    # so "+328 added" is a fact about a fresh index rather than about the
    # corpus, and a reader who sees the counters first has already drawn the
    # wrong conclusion.
    if stats.rebuilt:
        print(f"rebuilt the index and swapped it in: {stats.rebuild_reason}")
    if stats.knowledge_repo is not None:
        print(f"knowledge repo: {stats.knowledge_repo} (whole tree indexed)")
    if stats.reclassified:
        print(
            f"the knowledge repo changed since the last build: reclassified "
            f"{stats.reclassified} retained document(s)"
        )
    # Named because it is not always the index the reader is about to search.
    # The state directory follows the checkout the command RUNS FROM while the
    # corpus follows the checkout that OWNS the content, so a reindex from a
    # worktree reads the canonical repo and writes a database under the
    # worktree, which no other session opens. Both halves of that split are
    # deliberate, but an unnamed path makes the combination silent, and a
    # silent successful run against the wrong file is the shape of defect this
    # command was just fixed for.
    print(f"index: {stats.db_path}")
    print(
        f"scanned {stats.docs_seen} docs: "
        f"+{stats.added} added, {stats.changed} changed, "
        f"{stats.removed} removed, {stats.unchanged} unchanged, "
        f"{stats.rejected} rejected"
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


def _environment_knowledge_repo() -> tuple[str | None, bool]:
    """What THIS process would apply, and whether it could work that out.

    Returns ``(repo in scope or None, resolved)``. The bool is not decoration.
    ``workspace_root`` raises when nothing is configured and there is no
    checkout to infer one from, which is a legitimate state for a wheel
    install, and "I cannot say what the workspace is" is a different answer
    from "the workspace has no repo getting whole-tree treatment". Collapsing
    them into a bare None made an installation with no BRAIN_WORKSPACE_ROOT
    report exactly what a workspace with no knowledge repo reports (#100).
    """
    name = knowledge_repo()
    try:
        root: Path | None = db.workspace_root()
    except db.NotConfigured:
        # Not fatal on its own: explicit corpus roots answer the scope
        # question without a workspace root, and asking below is what tells
        # the two apart.
        root = None
    if not corpus_scope_resolvable(root):
        return None, False
    return (name if knowledge_repo_in_scope(name, root) else None), True


def _render_knowledge_repo(name: str | None) -> str:
    """One phrasing for the absent case, shared by every surface that says it."""
    return name if name else "none in scope"


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
        stored_knowledge = db.get_meta(conn, db.META_KNOWLEDGE_REPO)
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
    # The two halves need different evidence and so cannot share a guard. A
    # WIDTH comparison needs the backend to say what it produces, so it is
    # only meaningful when the probe succeeded. A MODEL comparison needs
    # nothing but configuration, and gating it on backend health silenced it
    # in the one case that matters most: switching to a model that is not
    # pulled yet makes the probe fail, and status then reported no staleness
    # for a configuration search was already refusing to use. Same predicate
    # as search, so the two surfaces cannot disagree.
    if db.stale_embed_model(embed_model, embedder.model) is not None:
        stale_reasons.append(
            f"index was embedded with {embed_model!r}, active model is "
            f"{embedder.model!r}"
        )
    if embed_live == "ready":
        if index_dim is not None and live_dim and live_dim != index_dim:
            stale_reasons.append(
                f"index holds {index_dim}-dimension vectors, active model "
                f"{embedder.model!r} produces {live_dim}"
            )

    fully_embedded = (
        vec_ok
        and embed_live == "ready"
        and embedded_count >= chunk_count
        and not stale_reasons
    )

    # "" means a run resolved the question and found nothing in scope; absence
    # means no run ever recorded it. Only the second is unknown, and only the
    # first can be compared against today.
    knowledge_from_index = stored_knowledge or None
    knowledge_now, knowledge_now_resolved = _environment_knowledge_repo()
    knowledge_stale = None
    if stored_knowledge is not None and knowledge_now_resolved:
        if knowledge_from_index != knowledge_now:
            knowledge_stale = (
                f"index was built with knowledge repo "
                f"{_render_knowledge_repo(knowledge_from_index)}, this "
                f"environment resolves "
                f"{_render_knowledge_repo(knowledge_now)}"
            )
    # Deliberately NOT folded into fully_embedded or stale_reasons: those are
    # about vectors, and a caller checking whether search will work correctly
    # should not have that answer change because a corpus boundary moved.

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
        # Which repo received whole-tree treatment IN THE RUN THAT BUILT THIS
        # INDEX, read from meta rather than recomputed. Reported because the
        # damage from getting it wrong is documents that are simply absent:
        # the index answers, the answers are just missing whatever that repo
        # holds outside docs/, and nothing else on this surface would say so.
        #
        # Every other field here describes the opened database. Deriving this
        # one from the current environment made it the only field describing
        # something else, with no way for a reader to tell which they got
        # (#100). The environment answer is still worth having, so it is a
        # separate field, and the DISAGREEMENT is the useful signal: it means
        # the index is stale with respect to how it would be built today.
        "knowledge_repo": knowledge_from_index,
        "knowledge_repo_recorded": stored_knowledge is not None,
        "knowledge_repo_now": knowledge_now,
        "knowledge_repo_now_resolved": knowledge_now_resolved,
        "knowledge_repo_stale": knowledge_stale,
    }


def cmd_adr_new(args: argparse.Namespace) -> int:
    """Allocate a decision number, write the stub, and claim the number.

    The decisions directory comes from the checkout this command RUNS FROM, not
    from the checkout that owns the corpus. That is the opposite of what
    :func:`corpusdex.indexer.knowledge_repo` wants and it is deliberate: a new
    record is authored on a branch, so the worktree is the right target. See
    decision 0086 for why one function answering both questions was a bug.
    """
    root = db.repo_root()
    if root is None:
        raise adr.AdrError(
            "not running from a source checkout, so there is no decisions "
            "directory to write to; run this from the knowledge repo"
        )
    decisions = root / "decisions"
    title = args.title or args.slug.replace("-", " ").capitalize()
    allocation = adr.create(
        decisions,
        args.slug,
        title,
        _split_list(args.repos),
        _split_list(args.tags),
        repo_root=root,
    )
    print(f"allocated {allocation.number} from: {', '.join(allocation.sources)}")
    if "origin/main unavailable" in allocation.sources:
        # Not a refusal: see adr._numbers_on_remote_main. But an unverified
        # number reads exactly like a verified one, and this is the case where
        # a record that merged while this branch was open is invisible, so say
        # it in its own line rather than leaving it inside a comma list.
        print(
            "warning: origin/main could not be read, so this number does not "
            "account for records that merged while this branch was open; the "
            "index entry is what will catch it at merge time"
        )
    print(f"record: {allocation.path}")
    print(f"claimed in: {decisions / adr.INDEX_NAME}")
    # Said explicitly because the number is not safe until it is pushed and
    # merged. Another branch can be holding the same one right now; the index
    # entry is what turns that into a merge conflict instead of a red main.
    print(
        "commit the record and the index entry TOGETHER: the entry is what "
        "makes a competing claim on this number conflict at merge time"
    )
    return 0


def _split_list(raw: str | None) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


#: Rendered as a block below rather than by the generic loop. The loop drops
#: None, which is right for a field that is simply absent and wrong for these:
#: None here is a real answer ("no repo in scope") and dropping it makes the
#: most important case the one the reader never sees.
_KNOWLEDGE_KEYS = (
    "knowledge_repo",
    "knowledge_repo_recorded",
    "knowledge_repo_now",
    "knowledge_repo_now_resolved",
    "knowledge_repo_stale",
)


def cmd_status(args: argparse.Namespace) -> int:
    payload = status_payload()

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for key, value in payload.items():
        if value is None or key in _KNOWLEDGE_KEYS:
            continue
        print(f"{key}: {value}")

    if payload["knowledge_repo_recorded"]:
        stamped = _render_knowledge_repo(payload["knowledge_repo"])
        print(f"knowledge repo (this index): {stamped}")
    else:
        print("knowledge repo (this index): not recorded; reindex to stamp it")
    if payload["knowledge_repo_now_resolved"]:
        current = _render_knowledge_repo(payload["knowledge_repo_now"])
        print(f"knowledge repo (this environment): {current}")
    else:
        # Deliberately does not name a cause. Resolution fails for an
        # unconfigured workspace AND for a configured one whose repo registry
        # is missing or unsafe to read, and this surface has not established
        # which. Naming the wrong one sends the reader to fix something that
        # is not broken.
        print(
            "knowledge repo (this environment): cannot resolve; "
            "check the workspace root, corpus roots and repo registry"
        )
    if payload["knowledge_repo_stale"]:
        # A disagreement is not an error, so this does not change the exit
        # code. It means the index is stale with respect to how it would be
        # built today, and the remedy is a reindex.
        print(f"warning: {payload['knowledge_repo_stale']}; reindex to apply it")
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

    p_adr = sub.add_parser("adr", help="decision record helpers")
    adr_sub = p_adr.add_subparsers(dest="adr_command")
    p_adr_new = adr_sub.add_parser(
        "new", help="allocate a decision number, write the stub, claim the number"
    )
    p_adr_new.add_argument("slug", help="lowercase-hyphenated slug, no number")
    p_adr_new.add_argument("--title", help="heading text (default: the slug, prettified)")
    p_adr_new.add_argument("--repos", help="comma-separated repos the decision affects")
    p_adr_new.add_argument("--tags", help="comma-separated tags")
    p_adr_new.set_defaults(func=cmd_adr_new)
    # `brain adr` with no subcommand must not fall through to args.func, which
    # would not exist. main() prints help on a missing top-level command; this
    # is the same contract one level down.
    p_adr.set_defaults(func=lambda _args: (p_adr.print_help(), 1)[1])

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
