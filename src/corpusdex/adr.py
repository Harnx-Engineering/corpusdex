"""Allocate a decision-record number and claim it in one step.

A decision number is a shared resource, and until ``decisions/INDEX.md`` existed
it had no shared file: two branches could each take the same number, and the
merge was clean because the filenames differ after the number, so git saw two
unrelated new files. Nine collisions came from that, two of them caused by
repairs to earlier ones.

The index makes a second claim conflict at merge time. This module makes the
first claim correct without anyone having to remember how, which is the other
half: a mechanism that depends on every author appending a line by hand is a
mechanism that gets skipped.
"""

from __future__ import annotations

import datetime as _dt
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: A record filename. The number is the citable identity, so it, not the slug,
#: is what has to be unique.
RECORD_NAME = re.compile(r"^(\d{4})-([a-z0-9-]+)\.md$")
INDEX_NAME = "INDEX.md"
#: Two spaces, and no link. An index that linked every record would attach
#: itself to every decision's link graph and surface beside every decision hit.
INDEX_LINE = re.compile(r"^- (\d{4})  ([a-z0-9-]+)$")
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class AdrError(RuntimeError):
    """A refusal the CLI should print without a traceback."""


@dataclass(frozen=True)
class Allocation:
    number: str
    path: Path
    #: Where the highest seen number came from. Reported because "0099" is not
    #: useful on its own: a reader needs to know whether the answer accounted
    #: for records that exist only on the remote.
    sources: tuple[str, ...]


def _numbers_on_disk(decisions: Path) -> set[int]:
    return {
        int(m.group(1))
        for p in decisions.glob("*.md")
        if (m := RECORD_NAME.match(p.name))
    }


def _numbers_in_index(decisions: Path) -> set[int]:
    index = decisions / INDEX_NAME
    if not index.is_file():
        return set()
    found = set()
    for line in index.read_text(encoding="utf-8").splitlines():
        match = INDEX_LINE.match(line)
        if match:
            found.add(int(match.group(1)))
    return found


def _numbers_on_remote_main(repo_root: Path, pathspec: str) -> tuple[set[int], bool]:
    """Numbers in ``origin/main``'s ``decisions/``, read WITHOUT fetching.

    Deliberately no network. A fetch here would make an allocation hang behind
    whatever is wrong with the network that day, and this workspace has a proxy
    that turns a stall into a silent wrong answer. Reading the existing
    remote-tracking ref is free and still catches the common case: a record
    that merged to main while this branch was being written.

    Returns ``(numbers, consulted)``. ``consulted`` is False when there is no
    ref to read, so the caller can say the answer is local-only rather than
    implying it checked.

    Two environment caveats, neither worth machinery here. On NFS, O_APPEND is
    emulated as seek-then-write, so the append below is not atomic there and
    the later writer can overwrite; this corpus lives in a local git checkout.
    And a readable but months-stale ref reports as consulted, because that is
    what it is: the last fetched state, not the remote. The CLI says the number
    is not safe until it is merged for that reason.

    An unreadable ref is reported, not refused, and that is a decision rather
    than an oversight. Refusing would mean no record can be written without
    git and the network, and the number this returns is not the guarantee
    anyway: ``decisions/INDEX.md`` is, because a colliding claim conflicts at
    merge time whether or not this read succeeded. The remote read only moves
    the same catch earlier. Callers should surface the unavailable case loudly
    so the author knows the number is unverified, and the CLI does.
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-tree",
                "--name-only",
                "origin/main",
                # Derived from the caller's directory, never hardcoded. A fixed
                # "decisions/" resolves against repo_root, so a nested or
                # differently named directory made this read a DIFFERENT path
                # and still report that origin/main was consulted: empty output
                # exits 0, which is indistinguishable from a repo with no
                # records.
                pathspec,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return set(), False
    if proc.returncode != 0:
        return set(), False
    found = set()
    for line in proc.stdout.splitlines():
        match = RECORD_NAME.match(line.rsplit("/", 1)[-1])
        if match:
            found.add(int(match.group(1)))
    return found, True


def allocate(decisions: Path, slug: str, repo_root: Path | None = None) -> Allocation:
    """Pick the next free number, considering disk, the index, and origin/main.

    The union of three sources rather than the highest file on disk. Disk alone
    misses a number reserved by an index entry whose record is not written yet,
    and both miss a record that merged to main while this branch was open, which
    is precisely the case every collision came from.
    """
    if not SLUG.match(slug):
        raise AdrError(
            f"slug {slug!r} is not lowercase-hyphenated (a-z, 0-9, single "
            "hyphens between words), and the filename convention is what the "
            "guard test and the index parser both key on"
        )
    if not decisions.is_dir():
        raise AdrError(f"no decisions directory at {decisions}")
    if repo_root is not None:
        # Otherwise the origin/main read answers about a different repository
        # and the report still says it consulted the remote. Every repo in this
        # workspace numbers its decisions from 0001, so a mismatch is not a
        # far-fetched input: it silently borrows another repo's highest number,
        # or worse, reports a check that never covered this corpus.
        root = repo_root.resolve()
        target = decisions.resolve()
        if root == target or root not in target.parents:
            raise AdrError(
                f"repo_root {repo_root} does not contain {decisions} as a "
                "subdirectory, so its origin/main holds a different "
                "repository's numbering"
            )
        pathspec = target.relative_to(root).as_posix() + "/"

    # Index entries count too. An entry reserves its number, and it names a
    # slug; taking that slug because the file is not written yet produces two
    # claims on one slug, which every wikilink and the `name:` frontmatter
    # assume is unique.
    existing_slugs = {
        m.group(2) for p in decisions.glob("*.md") if (m := RECORD_NAME.match(p.name))
    }
    index = decisions / INDEX_NAME
    if index.is_file():
        for line in index.read_text(encoding="utf-8").splitlines():
            if match := INDEX_LINE.match(line):
                existing_slugs.add(match.group(2))
    if slug in existing_slugs:
        raise AdrError(
            f"a record with the slug {slug!r} already exists; decision records "
            "are never replaced, so either supersede it (set superseded_by on "
            "the old one) or choose a distinct slug"
        )

    on_disk = _numbers_on_disk(decisions)
    in_index = _numbers_in_index(decisions)
    sources = ["disk", "index"]
    on_remote: set[int] = set()
    if repo_root is not None:
        on_remote, consulted = _numbers_on_remote_main(repo_root, pathspec)
        sources.append("origin/main" if consulted else "origin/main unavailable")

    highest = max(on_disk | in_index | on_remote, default=0)
    if highest >= 9999:
        # RECORD_NAME and INDEX_LINE both require exactly four digits, so a
        # 10000 record would be invisible to allocation and to the guard test
        # at once: the next allocation would reuse 10000 and neither the index
        # correspondence nor the duplicate check would see either of them.
        raise AdrError(
            f"the highest decision number in use is {highest:04d}; the record "
            "filename and index formats are both fixed at four digits, so "
            "widening them is a change to every citation and cannot be done "
            "here"
        )
    number = f"{highest + 1:04d}"
    return Allocation(number, decisions / f"{number}-{slug}.md", tuple(sources))


#: A repo or tag as it can appear inside a YAML flow sequence unquoted. A
#: value containing ": " turns the list element into a mapping, which the
#: frontmatter guard does not type-check and the indexer then mis-reads.
YAML_SAFE_ITEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _stub(number: str, slug: str, title: str, repos: list[str], tags: list[str]) -> str:
    for label, values in (("repo", repos), ("tag", tags)):
        for value in values:
            if not YAML_SAFE_ITEM.match(value):
                raise AdrError(
                    f"{label} {value!r} is not a plain identifier; it would be "
                    "written into the frontmatter list unquoted and change the "
                    "record's shape rather than its contents"
                )
    today = _dt.date.today().isoformat()
    repos_line = ", ".join(repos)
    tags_line = ", ".join(tags)
    return (
        "---\n"
        f"name: {slug}\n"
        f"decided_on: {today}\n"
        # Written as an explicit null rather than omitted: present-and-null
        # asserts that nothing replaces this record, and the guard test
        # requires the key for exactly that reason.
        "superseded_by: null\n"
        f"repos: [{repos_line}]\n"
        f"tags: [{tags_line}]\n"
        "---\n"
        "\n"
        f"# {number}: {title}\n"
        "\n"
        "## Decision\n"
        "\n"
        "## Why\n"
        "\n"
        "## Consequences\n"
    )


def _append_index_entry(decisions: Path, number: str, slug: str) -> None:
    """Append the claim. At the END, because the end is where merges collide.

    An entry inserted in the middle merges cleanly against a competing branch's
    append, which silently removes the only reason this file exists.
    """
    index = decisions / INDEX_NAME
    if not index.is_file():
        raise AdrError(
            f"{index} is missing; it is the file a competing claim conflicts "
            "with, so a record claimed without it is claimed without any "
            "collision check"
        )
    # One O_APPEND write, not read-modify-write. Two sessions share a checkout
    # in this workspace, and rewriting the whole file means the later writer
    # overwrites the earlier one's claim while both report success. Appending
    # also cannot truncate what is already there, so a failed write leaves the
    # existing claims intact rather than needing a rollback that this function
    # is not in a position to perform.
    entry = f"- {number}  {slug}\n".encode()
    tail = index.read_bytes()[-1:]
    if tail not in (b"", b"\n"):
        entry = b"\n" + entry
    with index.open("ab") as handle:
        handle.write(entry)


def create(
    decisions: Path,
    slug: str,
    title: str,
    repos: list[str],
    tags: list[str],
    repo_root: Path | None = None,
) -> Allocation:
    """Write the record and claim its number, or leave nothing behind."""
    # Checked before allocate() so the message can tell an interrupted run
    # apart from a genuine slug reuse. allocate() refuses a known slug with
    # "supersede it or choose another", which is the wrong instruction when the
    # record is sitting right there missing only its index line.
    for existing in decisions.glob(f"*-{slug}.md"):
        if (match := RECORD_NAME.match(existing.name)) and _numbers_in_index(
            decisions
        ).isdisjoint({int(match.group(1))}):
            raise AdrError(
                f"{existing.name} exists but has no {INDEX_NAME} entry, which "
                "is what an interrupted run leaves behind. Append "
                f"'- {match.group(1)}  {slug}' to {INDEX_NAME} rather than "
                "allocating a second number for the same decision"
            )
    allocation = allocate(decisions, slug, repo_root)
    if allocation.path.exists():
        raise AdrError(f"{allocation.path.name} already exists")
    allocation.path.write_text(
        _stub(allocation.number, slug, title, repos, tags), encoding="utf-8"
    )
    try:
        _append_index_entry(decisions, allocation.number, slug)
    except Exception:
        # The stub is ours and was empty a moment ago, so removing it is safe
        # and is not the destructive kind of error-path cleanup. Leaving it
        # would be a record with no index entry: a number claimed without
        # anything for a second claim to conflict with, which is the exact
        # state this command exists to prevent.
        allocation.path.unlink(missing_ok=True)
        raise
    # Two runs in the SAME checkout never reach a merge, so the index conflict
    # cannot fire and both would report success on one number. Re-reading after
    # the append is what turns that into a refusal: the loser sees its own
    # number twice. Not a lock, so a narrow window remains where both read a
    # clean index, but the window is now the append itself rather than the whole
    # allocate-write-append sequence.
    claims = [
        line
        for line in (decisions / INDEX_NAME).read_text(encoding="utf-8").splitlines()
        if (m := INDEX_LINE.match(line)) and m.group(1) == allocation.number
    ]
    if len(claims) > 1:
        allocation.path.unlink(missing_ok=True)
        raise AdrError(
            f"{allocation.number} was claimed twice while this ran ({claims}); "
            "another session in this checkout allocated the same number. The "
            "record has been removed. Remove the duplicate line and run again"
        )
    return allocation
