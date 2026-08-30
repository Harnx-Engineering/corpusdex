"""Document link extraction for the doc-level entity graph.

Links are read syntactically from Markdown that the corpus already writes, so
indexing stays LLM-free. Three body forms are recognised, plus one frontmatter
field:

* ``[[wikilink]]`` (with ``|alias`` and ``#anchor`` suffixes stripped): the
  convention decisions/0005 commits the corpus to going forward.
* ``[text](target.md)``: ordinary Markdown inline links.
* a bare relative path ending in ``.md`` (e.g. ``decisions/0001-x.md``): the
  form the corpus actually uses today, and the only one present in it at the
  time this was written. Supporting only wikilinks would have produced an
  empty graph and an inert ranking channel.
* frontmatter ``superseded_by``: the one link-bearing frontmatter field.
  ``repos`` and ``tags`` are deliberately NOT links; they name repositories
  and topics rather than documents, and turning them into edges would wire
  every decision into a handful of giant hubs that dominate any graph walk
  while carrying no document-to-document meaning.

One limit is deliberate. A bare path reference must contain a slash, so a
lone ``AGENTS.md`` is not treated as a link: every registered repo has one,
so the target is ambiguous by construction and resolution would drop it
anyway. (A bare filename written as a Markdown inline link still becomes a
target, and still resolves only when exactly one document in the linking
document's own repo carries that name.)

``./`` and ``../`` targets ARE resolved, by
:func:`resolve_relative_path`, against the linking document's own directory.
This was previously documented as an acceptable non-feature on the grounds
that the corpus writes repo-rooted paths; measurement disagreed. Of 76
relative targets in the corpus, path arithmetic resolves 47, of which 21 were
edges the filename fallback could not produce. It also never disagreed with
the fallback on the other 26, so the change is additive rather than a
correction of existing edges.

A prose citation of a decision record by NUMBER (``decision 0006``,
``ADR 0005``, ``Related decisions: 0006, 0011``) is also a target, emitted as
the bare four-digit number and resolved against the decision records'
numbering by :func:`corpusdex.indexer.rebuild_link_graph`. This is not a
stylistic nicety: it is how the corpus overwhelmingly cites itself, and
because none of the three path forms above matches it, those citations were
invisible to the graph. Counted 2026-08-29, 32 such citations named a real
record and produced no edge, against 169 edges in total, and they were
exactly the cross-record joins that multi-hop queries turn on. The keyword is
required; a bare four-digit number is a year, a port, or a line number far
more often than it is a decision, so ``0006`` alone is not a link.

A target carrying a URL scheme is not a link to a document and is discarded
at extraction. ``[spec](https://host/x/y.md)`` matched the inline-link
pattern and stored the whole URL as a target, which could then never resolve
and inflated the unresolved count with something no corpus change could fix.

Extraction is text only: targets are resolved to document ids later, against
the fully populated ``documents`` table (see
:func:`corpusdex.indexer.rebuild_link_graph`), because a link may point at a
document that has not been indexed yet when its source is read.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass

#: Version of the link-EXTRACTION rules in this module. Bump it in the same
#: commit as any change to what :func:`extract_links` returns for unchanged
#: text: a new pattern, a widened one, a keyword added or removed, a change to
#: :func:`strip_code`.
#:
#: Link targets are extracted only when a document is (re)chunked, and an
#: incremental reindex skips every document whose bytes have not moved. So a
#: smarter extractor shipped against an existing index changes NOTHING until
#: someone happens to run ``brain reindex --full``: the stored targets stay
#: exactly as the previous extractor left them. That was measured, not
#: reasoned about -- clearing the target table and reindexing incrementally
#: left the graph at zero edges, and only ``--full`` restored it -- and it
#: applies to every past and future change here, which is why the remedy is a
#: version the indexer can compare rather than a note asking people to
#: remember.
EXTRACTOR_VERSION = 2

RELATION_LINK = "links_to"
RELATION_SUPERSEDED_BY = "superseded_by"

_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
#: An inline code span, delimited by a RUN of backticks of any length. The
#: naive ``r"`[^`]*`"`` matched the first two backticks of a ``​``double``​``
#: span as an empty span, blanked those two characters, and left the content
#: behind as prose, so a documentation example of a citation became a real
#: edge. Matched per line, which is why no DOTALL is needed.
_INLINE_CODE_RE = re.compile(r"(`+)(?:(?!\1).)*?\1")
_WIKILINK_RE = re.compile(r"\[\[([^\]\[|#]+)(?:[|#][^\]\[]*)?\]\]")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+?\.md)(?:#[^)\s]*)?\)")
_BARE_PATH_RE = re.compile(r"(?<![\w./-])((?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+\.md)\b")

#: A decision record cited by number in prose. The keyword carries all the
#: precision: without it a four-digit number matches years and port numbers,
#: and with it the corpus produces no match on a number that names no record.
#:
#: ``record``/``records`` were accepted as keywords and are not, because
#: "this record 2026" and "the records 1970 through 1999" are ordinary
#: English while "decision 2026" is not, and the corpus cites nothing that
#: way. Dropping the keyword removes a whole false-positive class at no
#: measured cost.
#:
#: The trailing run follows the corpus's own "Related decisions:" idiom,
#: including the ANNOTATED form it actually writes,
#: ``0006 (why it matters), 0011 (why that matters), 0016``. The
#: unannotated form that the first version of this pattern handled appears
#: nowhere in the corpus, so the parenthetical is not an edge case here, it
#: is the normal case. An Oxford comma (``0006, 0007, and 0008``) is
#: likewise a real corpus form.
#:
#: ``ADR-0006`` is deliberately NOT matched, and this is load-bearing rather
#: than an omission. The hyphen form is how the context cards cite ANOTHER
#: repo's numbering (``context/some-repo.md`` cites some-repo's own
#: ADR-0001, 0003, 0008, 0009), whose records are not in this corpus; the
#: space form is how this corpus cites itself. Matching the hyphen form
#: would resolve those five citations onto knowledge-repo decisions carrying
#: the same numbers and entirely unrelated subjects. Producing no edge is
#: the correct outcome and the reason it currently holds.
_ADR_CITATION_RE = re.compile(
    r"\b(?:ADRs?|decisions?)\b\s*:?\s*"
    r"(\d{4}\b(?:\s*(?:\([^()]*\))?\s*(?:,\s*and|,|and|/|&)\s*\d{4}\b)*)",
    re.IGNORECASE,
)
_FOUR_DIGITS_RE = re.compile(r"\d{4}")


@dataclass(frozen=True)
class LinkTarget:
    """One extracted, still-unresolved link: raw target text plus its relation."""

    target: str
    relation: str


def strip_code(text: str) -> str:
    """Blank out fenced blocks and inline code spans.

    A path or bracket inside a code sample is an illustration, not a link.
    Fenced lines are replaced with empty lines rather than removed so nothing
    downstream depends on line numbering staying aligned.
    """
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if fence is None and match is not None:
            fence = match.group(1)[0]
            out.append("")
            continue
        if fence is not None:
            if match is not None and match.group(1)[0] == fence:
                fence = None
            out.append("")
            continue
        out.append(_INLINE_CODE_RE.sub(" ", line))
    return "\n".join(out)


#: Any RFC-3986-shaped scheme, plus the protocol-relative ``//host/...`` form.
#: Matched on the raw target, before normalization strips leading slashes,
#: because that stripping is what would turn ``//host/x.md`` into something
#: that looks like a path.
_URL_LIKE_RE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//)")


def _clean_target(raw: str) -> str | None:
    target = raw.strip()
    if not target:
        return None
    if _URL_LIKE_RE.match(target):
        # A remote URL that happens to end in .md names a file on some host,
        # not a document in this corpus. Kept out of doc_link_targets rather
        # than dropped at resolution time, so it never counts as an
        # unresolved target: nothing anyone writes here could resolve it.
        return None
    return target


def extract_body_links(body: str) -> list[str]:
    """Return the distinct link targets in ``body``, in first-seen order.

    De-duplicated here rather than by the caller because the patterns overlap
    by design: the path inside a Markdown inline link is also a valid bare
    path reference, so ``[notes](garden/notes.md)`` matches twice.
    """
    scannable = strip_code(body)
    targets: list[str] = []
    seen: set[str] = set()
    for pattern in (_WIKILINK_RE, _MD_LINK_RE, _BARE_PATH_RE):
        for match in pattern.finditer(scannable):
            target = _clean_target(match.group(1))
            if target is not None and target not in seen:
                seen.add(target)
                targets.append(target)
    # Handled after the loop rather than inside it because one citation can
    # name several records ("Related decisions: 0006, 0011"), so the match
    # yields a run of targets where every other pattern yields exactly one.
    for match in _ADR_CITATION_RE.finditer(scannable):
        for number in _FOUR_DIGITS_RE.findall(match.group(1)):
            if number not in seen:
                seen.add(number)
                targets.append(number)
    return targets


def extract_links(frontmatter: dict[str, object], body: str) -> list[LinkTarget]:
    """Extract every link from one document, de-duplicated by (target, relation).

    Frontmatter contributes ``superseded_by`` only; the body contributes
    wikilinks, Markdown links, and bare ``.md`` path references.
    """
    found: list[LinkTarget] = []
    seen: set[tuple[str, str]] = set()

    def add(target: str | None, relation: str) -> None:
        if target is None:
            return
        key = (target, relation)
        if key in seen:
            return
        seen.add(key)
        found.append(LinkTarget(target=target, relation=relation))

    raw_superseded = frontmatter.get("superseded_by")
    if raw_superseded is not None:
        text = str(raw_superseded).strip()
        if text and text.lower() not in {"null", "none", "~"}:
            add(text, RELATION_SUPERSEDED_BY)

    for target in extract_body_links(body):
        add(target, RELATION_LINK)

    return found


def resolution_keys(repo: str, rel_path: str, title: str) -> list[str]:
    """Return the lookup keys a document can be addressed by, most specific first.

    Mirrors how the corpus actually writes references: a workspace-relative
    path, the same path as seen from inside its own repo, the bare filename,
    the filename stem, and the document title.
    """
    keys = [rel_path]
    prefix = f"{repo}/"
    if rel_path.startswith(prefix):
        keys.append(rel_path[len(prefix) :])
    name = rel_path.rsplit("/", 1)[-1]
    keys.append(name)
    if name.endswith(".md"):
        keys.append(name[: -len(".md")])
    if title:
        keys.append(title)
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        normalized = key.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def normalize_target(target: str) -> str:
    """Normalize a raw link target for lookup against :func:`resolution_keys`."""
    cleaned = target.strip()
    for prefix in ("./", "/"):
        while cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned.lower()


def resolve_relative_path(source_path: str, target: str) -> str | None:
    """Resolve a ``./`` or ``../`` target against its linking document's directory.

    ``source_path`` is the linking document's workspace-relative path. Returns
    the workspace-relative path the target names, or ``None`` if the target is
    not relative or walks outside the workspace root.

    Path arithmetic, not a guess: ``../context/some-repo.md`` in
    ``the-brain/decisions/0012-x.md`` can only mean
    ``the-brain/context/some-repo.md``. That makes it strictly more
    specific than the filename fallback, which is why the caller tries this
    first and falls back only when it yields nothing.

    An escape above the workspace root is rejected rather than clamped.
    Clamping would silently retarget the link at some other document, and a
    reference that points outside the corpus is a fact about the document
    worth reporting, not a lookup to be repaired.
    """
    if not (target.startswith("./") or target.startswith("../")):
        return None
    base = posixpath.dirname(source_path)
    joined = posixpath.normpath(posixpath.join(base, target))
    if joined == ".." or joined.startswith("../") or posixpath.isabs(joined):
        return None
    return joined
