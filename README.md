# corpusdex

Hybrid local retrieval over a multi-repository Markdown corpus, built for
agents. Git-tracked Markdown is canonical; the SQLite index is a disposable
projection that can always be rebuilt from it. Retrieval fuses three channels
by reciprocal rank: FTS5 full-text, sqlite-vec dense vectors, and a
link-graph walk over resolved document-to-document edges.

Embeddings come from a local, loopback-only Ollama instance by deliberate
design, and the vector channel degrades out of the way (saying so) when that
backend is unreachable, rather than failing the query. Nothing about a
corpus leaves the machine it is indexed on.

Built to serve a workspace of decisions, per-repo context cards, architecture
maps and gap ledgers, with supersedence as a first-class idea: a decision
that replaced another surfaces alongside it, and a record that claims to be
superseded is ranked as such.

## Principles

1. Markdown in this repo is the source of truth. The search index is a derived,
   disposable projection that can always be rebuilt from the files.
2. Embeddings are computed locally (Ollama). No repository content is ever sent
   to a third-party embedding API.
3. Retrieval fuses three channels by reciprocal rank fusion, with a recency
   boost: FTS5 (BM25), vector similarity, and Personalized PageRank over the
   document link graph. When Ollama is unavailable the vector channel drops
   out and says so; lexical and graph still serve results.
4. Decisions carry supersedence metadata (`decided_on`, `superseded_by`).
   Superseded decisions stay in history but rank below their replacements.
   Every result also carries assembled context: compact references to the
   document's supersedence chain and its directly linked documents.
5. Updating the brain is part of the definition of done. See the Second Brain
   Update Rule in the workspace root `AGENTS.md`.

## Layout

- `decisions/` ADR-style decision records, one file per decision
- `context/` one context card per workspace repo
- `architecture/` cross-repo architecture maps and diagrams
- `gaps/` audit ledgers: tracked-vs-untracked gap inventories
- `src/corpusdex/` indexer, hybrid search, CLI, and MCP server
- `var/` (gitignored) the derived SQLite index and runtime state

## Corpus scope

Scope has exactly one source, and which one is a configuration choice. It is
never a union of the two: two mechanisms that have to agree about what is
corpus is how a stale registry quietly decides it.

**Registry mode (default).** Scope comes from the repo registry at the
workspace root, one `name<TAB>origin-url<TAB>branch` row per registered repo.
For each registered repo it indexes: top-level `AGENTS.md` / `CLAUDE.md` /
`README.md`, `tasks/*.md`, and `docs/**/*.md` at any depth inside that repo's
own directory tree (e.g. `some-repo/service/docs/auth-and-roles.md`).
It also indexes the workspace root's own `AGENTS.md` / `CLAUDE.md` /
`README.md`, its own direct `docs/**/*.md`, and its own `tasks/*.md`;
workspace-level `.claude/skills/*/SKILL.md`; and the knowledge repo's own
Markdown in full.

**Explicit roots.** `BRAIN_CORPUS_ROOTS` (an `os.pathsep`-separated list of
directories) indexes those directories directly and replaces registry mode
entirely. Each root contributes documents under a repo named for its own
directory, and each is treated exactly as a registered repo is, so pointing
this at a source tree does not pull in every stray Markdown file it happens
to contain. Paths are relative to each root and prefixed by that root's name,
because configured roots need share no common parent. This mode needs no
workspace root at all, which is the point: it exists for an installation
outside the workspace this engine grew up in.

**The knowledge repo.** One repo in a workspace is the brain itself: all of
its Markdown is corpus rather than only its `docs` subtree, and its
`decisions/`, `context/`, `architecture/` and `gaps/` directories carry a
`doc_type` that the same directory names elsewhere do not. `BRAIN_KNOWLEDGE_REPO`
names it. The default is the directory the engine runs from when that is a
source checkout, which is not the same as a constant: a constant naming this
repo stops matching the day the package is renamed, and every dev checkout
then resolves as "installed". Installed from a wheel there is no such
directory, so no repo gets whole-tree treatment unless one is named.

**Exclusions.** `BRAIN_EXCLUDE` (also `os.pathsep`-separated) drops matching
paths in either mode. A pattern with no `/` matches any single path *segment*,
so `vendor` excludes `a/vendor/b.md`; a pattern containing `/` is matched
against the whole relative path, so `some-repo/docs/*` narrows to one place.

Unregistered sibling directories (stale checkout copies, personal
directories, ad hoc clones) are never scanned, even if they look exactly like
a real repo: with no registry and no explicit roots, `brain reindex` fails
with a clear error rather than silently falling back to scanning the whole
workspace. Claude-managed ephemeral worktrees under `<repo>/.claude/worktrees/`
and workspace-level `.worktrees/` are both skipped as duplicate checkouts, not
indexed as additional corpus.

## Usage

```
uv run brain reindex        # incremental index over the registered-repo corpus
uv run brain reindex --json # same, machine-readable stats
uv run brain reindex --full # re-chunk every document regardless of fingerprint
uv run brain search "..."   # hybrid search, cited results
uv run brain get <ref>      # full chunk view, by the stable `ref` a search result carries
uv run brain eval           # recall@k and MRR over eval/queries.yaml, with channel ablation
uv run brain eval --per-query --json
uv run brain status         # index freshness, embedding backend health
```

### Where paths come from

| Variable | Legacy alias | Default | Meaning |
|---|---|---|---|
| `BRAIN_WORKSPACE_ROOT` | `HARNX_WORKSPACE_ROOT` | parent of the source checkout | Directory holding the repo checkouts to index |
| `BRAIN_STATE_DIR` | `HARNX_BRAIN_STATE_DIR` | `var/` in a checkout, else a per-user state directory | Where derived state (the index, the lock) is written |
| `BRAIN_DB` | `HARNX_BRAIN_DB` | `<state dir>/index.db` | The index file itself |
| `BRAIN_OLLAMA_HOST` | `HARNX_BRAIN_OLLAMA_HOST` | `http://localhost:11434` | Embedding backend; loopback is enforced |
| `BRAIN_EMBED_MODEL` | `HARNX_BRAIN_EMBED_MODEL` | `nomic-embed-text` | Embedding model, any width |

Scope settings, none of which has a legacy alias because none of them existed
before the split:

| Variable | Default | Meaning |
|---|---|---|
| `BRAIN_CORPUS_ROOTS` | unset | Directories to index directly; replaces registry mode |
| `BRAIN_EXCLUDE` | unset | Glob patterns to drop; bare names match any path segment |
| `BRAIN_REGISTRY_FILE` | `.corpus-repos.tsv`, then `.harnx-repos.tsv` | Registry filename; naming one replaces the search rather than extending it |
| `BRAIN_KNOWLEDGE_REPO` | the source checkout's directory name | The repo whose whole tree is corpus |
| `BRAIN_MCP_NAME` | `corpusdex` | The name the MCP server registers under; pin it when clients already name something else |

The registry filename resolves by existence, newest name first, for the same
reason the environment aliases exist: the workspace this engine grew up in has
the old name on disk, and a rename that stops it indexing the moment it lands
gets reverted rather than finished. `BRAIN_MCP_NAME` is deliberately not
derived from the package name: it is an integration identity that a client
config names back, so it can only change in step with those configs.

Every setting is read through `config.env`, which takes the canonical
`BRAIN_*` name and falls back to the workspace-era alias. Precedence is
new-name-first, so a caller that has migrated is never overridden by a stale
legacy value left in the environment. The alias table is explicit rather than
a computed prefix swap, because the old names were not consistent:
`HARNX_WORKSPACE_ROOT` has no `BRAIN` in it while every other setting does, so
a mechanical rewrite would have silently missed the one setting that decides
what gets indexed. `config.legacy_names_in_use()` reports which aliases are
still supplying values; when it is empty across the workspace, the table can
be dropped.

The embedding width is a property of the index, not a constant. It is learned
from the model's first response and stored in the `vec_chunks` declaration,
with a copy in `meta.embed_dim` for reporting. Selecting a model of a
different width (384, 1024) rebuilds the vector table and re-embeds through
the existing backfill; previously the width was pinned at 768 and any other
model failed validation on every vector, which made the model setting nearly
inert. A width change is handled before the model change because it drops the
table outright, which subsumes the delete.

The rebuild is deliberately gated on the width actually being *known*: when
the embedding backend is down the probe fails and the width is `None`, and an
unknown width must not read as a changed one, or an outage would discard every
vector it could not currently recompute.

Running from this checkout, all of these resolve to what they always did and
nothing needs setting. Installed as a package there is no checkout, so the
state directory moves to a per-user location and the workspace root has no
defensible default: `brain reindex` reports that `BRAIN_WORKSPACE_ROOT` must
be set rather than guessing one, or that `BRAIN_CORPUS_ROOTS` can name the
directories to index instead.

Guessing was the old behaviour and it was worse than it sounds. The checkout
root was resolved by walking up for any `pyproject.toml`, so a package
installed into a `.venv` inside *another* project claimed that project's root:
the index was written into their repository, and the corpus root became their
parent directory, so the tool would index an unrelated tree. The manifest's
declared `name` is now what identifies the checkout, and finding none is a
real answer rather than a fallback.

## MCP server

`brain-mcp` is a thin, read-only stdio MCP server over the same core the CLI
uses (`corpusdex.db`, `corpusdex.search`, `corpusdex.cli.status_payload`),
including the same degraded behaviour when Ollama is down (vector channel off,
lexical plus graph still serving). It exposes three tools and nothing else:

- `brain_search(query, n=8)` hybrid search, cited results (`path#heading`,
  score, snippet, `channels_used`/`mode`, `degraded` flag, `assembled`
  references)
- `brain_get(ref)` full chunk body plus document info and `assembled`, keyed
  by the stable `ref` a search result carries (not a rowid; see "Chunk refs")
- `brain_status()` the same fields as `brain status`

### What produced a page

`channels_used` lists the retrieval channels that actually contributed to a
result page, and `mode` is that list rendered in pipeline order
(`lexical+vector+graph`, `lexical+graph`, `lexical`, or `none`). Neither is a
statement about backend health; `degraded` is, and it means one thing only:
the vector channel was asked for and could not run.

The two used to disagree. `mode` was computed from `degraded` as a binary, so
a page the graph channel had re-ranked was reported as `lexical-only`, and a
caller that narrowed `channels` to lexical alone was told `hybrid` because
nothing had failed. Rendering the label from the channel set removes the
summary vocabulary that made either claim expressible (issue #13).

There is deliberately no reindex tool: indexing mutates the shared index and
takes the single-writer lock, so it stays a CLI concern (`brain reindex`), not
something an MCP client can trigger as a side effect of a read request.

Run it directly with `uv run brain-mcp`, or register it as a Claude Code
project MCP server. This workspace has no pre-existing `.mcp.json`
convention (checked the workspace root and the `.claude/` repo's tracked
files), so registration lives in a workspace-root `.mcp.json`:

```json
{
  "mcpServers": {
    "corpusdex": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "${CLAUDE_PROJECT_DIR:-.}/corpusdex", "run", "brain-mcp"],
      "env": {}
    }
  }
}
```

`${CLAUDE_PROJECT_DIR:-.}` is Claude Code's own environment-variable
expansion for manually configured stdio servers, so the path is never
hardcoded to one machine's checkout location. The workspace root is not
itself a git repository, so this `.mcp.json` is not tracked by any repo and
must not be copied into one with a hardcoded absolute path.

## Chunk refs

Every search result carries a `ref` such as `cf5a472f537725666`. That is the
handle to keep and to hand back to `brain get` or `brain_get`. It is derived
from the chunk's position (document path, heading path, and an ordinal that
disambiguates repeated identical headings), so it survives reindexing and keeps
naming the same section as that section's prose is edited.

The database rowid is not a durable handle and is deliberately never published
by any surface. A changed document's chunks are deleted and reinserted under
fresh rowids, and SQLite reuses freed ids, so a held rowid does not fail when it
goes stale: it silently resolves to unrelated content. Because a reindex job can
run between a search and a read, that window is routine rather than exotic
(issue #23). `brain get` rejects a numeric argument with the reason rather than
answering it.

A ref stops resolving when its section is renamed or removed, which is the
intended failure: the caller gets a clean miss and can search again.

## One chunk per document

A result page carries at most one chunk per document: the highest scoring one,
chosen after ranking so deduplication never changes which section the ranking
picked. Relevance is judged, cited and read per document, so several near
identical sections of one document spend the caller's slot budget without adding
an answer.

This is not a cosmetic tidy. Measured on the 30-query judged set before the rule
existed, a 10-slot page carried a mean of **5.93 distinct documents**, and one
query returned ten chunks of a single document. Adding the rule moved
`lexical+vector` recall@10 from 0.700 to **0.767** and MRR from 0.548 to 0.562,
the largest single retrieval improvement measured so far, because roughly 40
percent of every page had been spent restating documents the caller had already
been shown.

It also changes what the link-graph channel is worth: its multi-hop recall
contribution doubles (+0.083 to +0.167) once its finds stop being crowded out by
duplicate sections of documents the base channels already returned.

## Retrieval evaluation

`brain eval` scores retrieval over `eval/queries.yaml`, a judged query set held
in the repo, and reports recall@k and MRR per ablation arm (`lexical`,
`lexical+graph`, `lexical+vector`, `full`). It exists so that any change to
fusion, boosts, or chunking can be shown to help or hurt rather than argued
about (issue #17).

Three properties it is built to have:

- **Deterministic.** Two consecutive runs against an unchanged index produce
  byte-identical JSON. Ranking ties are broken on the stable `ref` rather than
  on rowid, so even a full rebuild of an unchanged corpus produces the same
  ordering.
- **Offline-capable.** The `lexical` and `lexical+graph` arms need no embedding
  backend, so the graph channel's contribution is measurable with Ollama down.
- **Honest about what did not run.** An arm whose channel is unavailable is
  reported as unavailable, never as a zero, and the graph-contribution line
  refuses to print a number when either comparison arm is missing. Switching a
  channel off for an ablation is not reported as degradation, and a requested
  channel that fails still is.

Relevance is judged per document, not per chunk, so chunker changes do not rot
the judgements. A judgement naming a document that is not in the index is a hard
error rather than a miss: scored as a miss, a renamed document is
indistinguishable from a ranking regression.

## Document link graph

Changing what `links.py` extracts requires bumping `links.EXTRACTOR_VERSION`
in the same commit. Link targets are extracted only while a document is
chunked, and an incremental reindex skips every document whose bytes have not
moved, so a smarter extractor shipped against an existing index otherwise
changes nothing at all until somebody happens to run `--full`. Measured, not
assumed: clearing the stored targets and reindexing incrementally left the
graph at zero edges. The stamp makes `brain reindex` rebuild once by itself
when the extractor changes.


At index time each document contributes its links: `[[wikilinks]]`, Markdown
inline links, bare relative `.md` path references, prose citations of a
decision record by number, and the `superseded_by` frontmatter field.
(`repos` and `tags` are deliberately not links; they name repositories and
topics, not documents.) Targets that resolve to no indexed document, or to
more than one ambiguously, are dropped rather than guessed.

Resolved edges drive three things: a Personalized PageRank channel seeded from
the documents behind the top lexical+vector hits, the `assembled` references
on every result, and the supersedence penalty described below. `brain reindex`
reports the graph it built:

```
link graph: 213 edges (92 targets name no document, 5 declined as ambiguous or self, 21 declined as work-log citations)
```

### The graph channel votes at half weight

It is not a peer channel. It is seeded from the fused lexical+vector head, so
its ranking partly re-expresses the base ranking's own opinion instead of
adding independent evidence. At a full RRF vote that opinion is counted twice
and the derived channel can outrank its own source. Issue #17 predicted this
when the eval harness was built ("nothing measures whether the channel adds
recall or only amplifies the head"); the answer turned out to be both.

Measured on the 36-query judged set, seven queries whose answer the base
ranking already had at **rank 1** were pushed down, one of them to rank 5, and
every one of those had a recall delta of exactly zero. The channel was not
finding anything for them, only reshuffling.

A weight sweep against a frozen index, contribution over `lexical+vector`:

| graph vote weight | recall | mrr |
|---|---|---|
| 1.00 (was) | +0.060 | -0.032 |
| 0.75 | +0.060 | -0.009 |
| **0.50** | **+0.060** | **+0.019** |
| 0.35 | +0.051 | +0.015 |
| 0.25 | +0.051 | +0.026 |
| 0.10 | +0.014 | +0.018 |

Recall is flat down to 0.5 and decays below it, so 0.5 is the smallest weight
that gives up no recall at all, and it is where MRR first turns positive. It
was chosen by that criterion rather than by the best MRR cell, which would
have picked 0.25 and paid recall for it. Five of the eight displaced queries
recover their rank-1 answer and the recall gain is preserved exactly; three
still lose rank.

**Read the MRR column as a guardrail, not as an objective.** A 20,000-sample
paired bootstrap over the per-query deltas puts the channel's recall
contribution at +0.060 with a 95% interval of [+0.014, +0.120], which excludes
zero, and its MRR contribution at +0.019 with an interval of [-0.027, +0.063],
which does not. Half of the 36 queries name a single relevant document, and for
those MRR is `1/rank`, so one document slipping from rank 1 to rank 2 moves the
mean by 0.014 -- nearly the whole effect. This set can see that the channel
finds documents; it cannot adjudicate how the channel orders them. That is why
the structural criterion above is load-bearing: the MRR peak at 0.25 was inside
the noise band and picking it would have been fitting to a coin flip. Use MRR
to catch a collapse (see the -0.149 below, which is far outside the band) and
never to justify a single-digit gain. Widening the set is issue #42.

Restricting the channel to pure expansion (voting only for documents the base
ranking missed) was measured and is much worse: recall +0.005. The channel's
value really is in promoting documents the base already found but ranked below
the cut, so it has to keep voting for them, just not loudly enough to outrank
them.

Two further repairs were measured and rejected, recorded here so they are not
retried. **Giving isolated documents a graph vote** looked obviously right --
`graph.load_adjacency` builds its node set only from `doc_links` rows, so a
document with no edge can never receive restart mass, and 170 of 303 documents
(56%) have no edge at all. Adding them costs recall (+0.060 to +0.037) and
turns the MRR contribution negative again, because isolated seeds are dangling,
retain their restart mass, rank high, and spend the channel's limited vote list
on documents that were already at the top. **Refusing to vote for documents the
base ranking already placed in its head** is worse still and is self-defeating:

| protected prefix | recall | mrr contribution |
|---|---|---|
| none | 0.745 | +0.019 |
| top 1 | 0.745 | +0.014 |
| top 3 | 0.745 | **-0.149** |
| top 5 | 0.745 | -0.140 |
| top 10 | 0.745 | -0.082 |

Recall is identical in every row, so the intervention is pure reordering.
Removing an entry from a rank-indexed list does not subtract its vote; it
promotes everything behind it by one rank, handing the head of the graph
channel to exactly the challengers the exclusion was meant to suppress.

All three failures share one mistake: treating the graph list as a set of
independent per-document votes when it is a rank-ordered list in which every
edit is relative. A real fix has to change a document's score rather than its
position, which means using PPR mass directly instead of reciprocal rank -- and
that needs a judged set that can see an ordering change. Closed as issue #38.

Two notes on measuring this. The index is shared, and another session
reindexing mid-run silently split one four-arm comparison across two corpora;
run ablations against a private frozen copy (`HARNX_BRAIN_DB` pointed at a
`sqlite3` backup). And the eval harness reports an arm whose channel could not
run as *unavailable* rather than as a zero, which is what caught a local
embedding backend exhausting ephemeral ports mid-sweep instead of recording it
as a catastrophic regression.

### Supersedence is read from the edge, not from the frontmatter string

A superseded document ranks below its replacement because an edge says so.
The penalty and the successor shown alongside a hit both read the resolved
`superseded_by` edge, so they cannot disagree.

They used to. The penalty keyed on the raw frontmatter string while the
displayed chain walked the edge, so a `superseded_by` naming a document that
does not exist (a typo, a rename, a record not yet written) dropped its
document by 70 percent while showing no successor and raising no error. The
document was buried and nothing anywhere could say why.

Deriving the penalty from the edge fixes that direction and opens the
opposite one: an unresolved value now means no penalty at all, so a record can
claim it was replaced and be ranked as though it never was, equally
invisibly. So `brain reindex` names them:

```
WARNING: 1 document(s) declare superseded_by but it resolves to no document, so no supersedence penalty or successor applies:
  the-brain/decisions/0007-signed-activation-boundary.md
```

By path, not by count. The complaint in issue #21 was that no query could say
which document was affected, and a count restates that complaint rather than
answering it.

### Citing a decision by number

`decision 0006`, `ADR 0005`, and `Related decisions: 0006 and 0011` are links.
This is how the corpus overwhelmingly cites itself, and because the form
matches none of the path patterns, those citations produced no edges at all
until they were recognised. Adding them took the graph from 169 to 223 edges,
and it is the change that made the link-graph channel worth running: measured
on the 36-query judged set with everything else held constant, its
contribution over `lexical+vector` moved from recall -0.005 / MRR -0.107 to
recall **+0.065 / MRR +0.056**, the first configuration in which `full` beats
`lexical+vector` on both metrics.

The forms that count were settled by reading the corpus, not by generalising
from the pattern. `Related decisions: 0006 (why it matters), 0011 (why that
matters), 0016` is what the corpus writes, and the parenthetical used to end
the run after the first number; the unannotated `0006 and 0011` that the
original test asserted appears nowhere in the corpus, so the test passed
against a form nothing produces. An Oxford comma (`0006, 0011, and 0016`) is
also a real form.

`ADR-0006` is deliberately **not** a link, and that is load-bearing rather
than an omission. The hyphen form is how the context cards cite *another*
repo's numbering (`context/some-repo.md` cites some-repo's own ADR-0001,
0003, 0008 and 0009), and those records are not in this corpus, so matching
it would resolve all five onto knowledge-repo decisions carrying the same
numbers and unrelated subjects. Producing no edge is the correct outcome.

The keyword is required. A bare four-digit number is a year, a port, or an
issue number far more often than it is a decision, and requiring the keyword
produced zero matches on a number that names no record. `record` is not among
the keywords for the same reason: "this record 2026" is ordinary English
where "decision 2026" is not. A record is
identified as `<...>/decisions/NNNN-slug.md`: both halves matter, because a
gap ledger named by date has the same shape as a decision named by number,
and only the directory separates them.

A citation from a `tasks/` work log is **not** an edge. An append-only ledger
cites a decision because it was worked on in some session, which is a
statement about scheduling rather than subject, and the two documents it
co-locates usually have nothing to do with each other. Measured with every
other arm held identical, dropping those 21 citations moved the `full` arm's
recall@10 from 0.722 to **0.745** and the graph channel's contribution from
+0.051 to **+0.074**; four queries moved and all four improved. Work logs keep
any link they write as an actual path, because that is a real reference.

A fan-out cap was measured as the alternative and rejected. It generalises
better in principle, but it cannot tell why a document cites widely: at a cap
of 5 it also truncates the curated repo context cards (one card cites seven
decisions, another five) and the paraphrase group's MRR
contribution collapses from +0.151 to +0.068. A cap high enough to spare the
cards scores well only because the threshold currently happens to fall between
the two populations, and would begin truncating a card the moment one grew.
The distinction being drawn is document kind, so that is what the rule tests.

A corpus with no links is a supported state, not a degraded one: the channel
contributes nothing, assembled context is empty, and ranking is exactly what
the two original channels produce.

## Schema versions and rebuilds

The index records the schema version it was written with. A version bump
needs no migration, because the index is a projection of the Markdown: an
ordinary `brain reindex` rebuilds it. Rebuilds are written to a scratch file
and swapped into place atomically, so a reader never sees a half-built index
and a failed rebuild leaves the previous one untouched. Read commands never
create or alter the index; if it is missing, empty, or written by a different
schema version, they say so and stop. An index *newer* than the running code
is refused rather than rebuilt, so an older checkout cannot silently
downgrade it.

## Embedding backfill and recovery

Embedding is decoupled from change detection: every `brain reindex` run (not
only `--full`) finds every chunk currently lacking a vector and embeds it, as
long as the local Ollama backend is reachable. This means an index built
while Ollama was down is fully caught up by the next ordinary reindex once
Ollama recovers; a full `--full` rebuild is never required just to backfill
missing embeddings. `brain status` and `brain reindex --json`'s
`fully_embedded` field reflect real vector coverage (embedded chunks versus
total chunks), not merely whether the vector table exists.

Coverage alone is not health, so `fully_embedded` is also false whenever the
configured model does not match the index. `brain status` reports the index's
own vector width as `embed_dim` and lists any mismatch under
`embed_config_stale`. Two cases hide behind a full-coverage count: a model of
a different width makes search fall back to lexical-only until the next
reindex, and a different model of the *same* width is worse, because queries
are then compared against vectors from another embedding space and the
rankings are quietly meaningless with nothing raising an error. Both are
cleared by `brain reindex`, which re-embeds at the active model's width.

## Known limitation: same-mtime, same-size edits

The incremental reindex fast-path skips re-reading a file when its mtime and
size both match what is stored. If a file is edited in a way that preserves
both its mtime (e.g. restored via `os.utime` or a tool that pins timestamps)
and its exact byte size, that edit will not be detected until something else
changes the file's mtime or size (or `brain reindex --full` is run). This is
a narrow, deliberate trade-off: reading and hashing every file on every
reindex would defeat the purpose of the fast path, and same-mtime/same-size
edits are rare in practice (most editors and git checkouts change at least
one of the two).
