"""Allocation and claiming of decision numbers.

Unlike ``test_decision_records.py``, which reads this repo's real records
because the defects it catches are properties of the corpus, everything here is
synthetic: allocation is code, and the cases that matter (a number that exists
only on the remote, a claim that half-succeeds) cannot be staged in the real
tree without writing to it.
"""

from __future__ import annotations

import argparse
import functools
import shutil
import subprocess
from pathlib import Path

import pytest

from corpusdex import adr

INDEX_HEADER = "# Decision number index\n\nheader prose\n\n"


def _decisions(tmp_path: Path, numbers: dict[str, str], *, with_index: bool = True) -> Path:
    """Build a decisions dir holding ``{number: slug}``."""
    d = tmp_path / "decisions"
    d.mkdir()
    for number, slug in sorted(numbers.items()):
        (d / f"{number}-{slug}.md").write_text(f"# {number}\n", encoding="utf-8")
    if with_index:
        body = "".join(f"- {n}  {s}\n" for n, s in sorted(numbers.items()))
        (d / adr.INDEX_NAME).write_text(INDEX_HEADER + body, encoding="utf-8")
    return d


def test_the_next_number_follows_the_highest_in_use(tmp_path: Path):
    d = _decisions(tmp_path, {"0001": "first", "0007": "seventh"})
    assert adr.allocate(d, "eighth").number == "0008"


def test_an_empty_decisions_dir_starts_at_one(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    (d / adr.INDEX_NAME).write_text(INDEX_HEADER, encoding="utf-8")
    assert adr.allocate(d, "first").number == "0001"


def test_an_index_entry_with_no_record_still_reserves_its_number(tmp_path: Path):
    """Disk alone would hand out a number the index has already promised.

    The entry is the claim; the file may simply not be written yet. Allocating
    over it would put two records on one number with the index asserting one of
    them, which is worse than the collision it is meant to prevent.
    """
    d = _decisions(tmp_path, {"0001": "first"})
    index = d / adr.INDEX_NAME
    index.write_text(index.read_text() + "- 0009  reserved-but-unwritten\n", encoding="utf-8")
    assert adr.allocate(d, "next").number == "0010"


def test_a_slug_that_already_exists_is_refused(tmp_path: Path):
    d = _decisions(tmp_path, {"0001": "already-here"})
    with pytest.raises(adr.AdrError, match="already exists"):
        adr.allocate(d, "already-here")


@pytest.mark.parametrize(
    "slug",
    ["Capitalised", "under_scored", "trailing-", "-leading", "double--hyphen", ""],
    ids=["capitalised", "underscore", "trailing", "leading", "double", "empty"],
)
def test_a_slug_outside_the_filename_convention_is_refused(tmp_path: Path, slug: str):
    """The convention is not cosmetic: ``RECORD_NAME`` and the index parser both
    key on it, so a record named outside it is invisible to allocation and to
    the guard test at once."""
    d = _decisions(tmp_path, {"0001": "first"})
    with pytest.raises(adr.AdrError, match="lowercase-hyphenated"):
        adr.allocate(d, slug)


def test_create_writes_the_record_and_claims_the_number_together(tmp_path: Path):
    d = _decisions(tmp_path, {"0001": "first"})
    allocation = adr.create(d, "second", "Second", ["example-repo"], ["numbering"])
    assert allocation.path.name == "0002-second.md"
    assert allocation.path.is_file()
    text = (d / adr.INDEX_NAME).read_text()
    assert text.endswith("- 0002  second\n"), "the claim must be APPENDED, not inserted"
    body = allocation.path.read_text()
    assert "name: second" in body
    assert "superseded_by: null" in body, (
        "present-and-null asserts that nothing replaces this record; the guard "
        "test requires the key"
    )
    assert "# 0002: Second" in body


def test_a_failed_claim_leaves_no_record_behind(tmp_path: Path):
    """A record with no index entry is a number claimed with nothing for a
    competing claim to conflict with, which is the exact state this command
    exists to prevent. Half-done is worse than not done."""
    d = _decisions(tmp_path, {"0001": "first"}, with_index=False)
    with pytest.raises(adr.AdrError, match="missing"):
        adr.create(d, "second", "Second", [], [])
    assert not (d / "0002-second.md").exists()
    assert sorted(p.name for p in d.glob("*.md")) == ["0001-first.md"]


def _git_repo_with_remote(
    tmp_path: Path, remote_numbers: dict[str, str], subdir: str = "decisions"
) -> Path:
    """A checkout whose ``origin/main`` holds records the checkout does not.

    Real git rather than a stubbed subprocess: the thing under test is that
    ``git ls-tree origin/main`` is read correctly, and a stub would confirm the
    parser against my own belief about git's output format.
    """
    git = shutil.which("git")
    if git is None:  # pragma: no cover - git is present wherever this repo is
        pytest.skip("git is not available")
    origin = tmp_path / "origin"
    origin.mkdir()
    run = functools.partial(
        subprocess.run, cwd=origin, check=True, capture_output=True, text=True
    )
    run([git, "init", "-q", "-b", "main"])
    run([git, "config", "user.email", "tests@example.invalid"])
    run([git, "config", "user.name", "tests"])
    (origin / subdir).mkdir(parents=True)
    for number, slug in remote_numbers.items():
        (origin / subdir / f"{number}-{slug}.md").write_text("x\n", encoding="utf-8")
    run([git, "add", "-A"])
    run([git, "commit", "-qm", "remote records"])

    clone = tmp_path / "clone"
    subprocess.run(
        [git, "clone", "-q", str(origin), str(clone)], check=True, capture_output=True
    )
    return clone


def test_a_number_that_exists_only_on_origin_main_is_not_handed_out(tmp_path: Path):
    """The case every collision came from.

    A branch cut before another record merged sees a lower highest-number on
    disk than main actually holds. Allocating from disk alone hands out a number
    that is already taken, and the merge is clean because the filenames differ.
    """
    clone = _git_repo_with_remote(tmp_path, {"0001": "first", "0042": "merged-while-i-worked"})
    decisions = clone / "decisions"
    # Simulate a branch that predates 0042: remove it locally, keep it on the
    # remote-tracking ref.
    (decisions / "0042-merged-while-i-worked.md").unlink()
    (decisions / adr.INDEX_NAME).write_text(INDEX_HEADER + "- 0001  first\n", encoding="utf-8")

    local_only = adr.allocate(decisions, "next")
    assert local_only.number == "0002", "without the remote, disk says 0002"

    with_remote = adr.allocate(decisions, "next", repo_root=clone)
    assert with_remote.number == "0043", (
        "origin/main holds 0042, so 0002 would collide with a record that is "
        "already merged"
    )
    assert "origin/main" in with_remote.sources


def test_an_unreadable_remote_allocates_anyway_and_says_so(tmp_path: Path):
    """Fail-open, deliberately, and this test pins the choice rather than
    blessing it by omission.

    The number is not the guarantee. ``decisions/INDEX.md`` is: a colliding
    claim conflicts at merge whether or not this read succeeded, so refusing
    here would only mean no record can be written without git and the network.
    What the caller must not do is present an unverified number as a verified
    one, so the source list has to name the gap.
    """
    d = _decisions(tmp_path, {"0001": "first"})
    allocation = adr.allocate(d, "second", repo_root=tmp_path)
    assert allocation.number == "0002"
    assert "origin/main unavailable" in allocation.sources
    assert "origin/main" not in allocation.sources, (
        "the bare source name would claim a check that did not happen"
    )


def test_the_cli_warns_on_its_own_line_when_the_remote_was_unreadable(
    tmp_path: Path, capsys, monkeypatch
):
    """An unverified number reads exactly like a verified one.

    The annotation inside a comma-separated source list is easy to miss, and
    the case it marks is precisely the one every collision came from.
    """
    from corpusdex import cli, db

    _decisions(tmp_path, {"0001": "first"})
    monkeypatch.setattr(db, "repo_root", lambda: tmp_path)
    args = argparse.Namespace(slug="second", title=None, repos=None, tags=None)
    assert cli.cmd_adr_new(args) == 0
    out = capsys.readouterr().out
    assert "warning: origin/main could not be read" in out


def test_a_number_past_four_digits_is_refused(tmp_path: Path):
    """Both parsers require exactly four digits.

    A 10000 record is invisible to allocation AND to the guard test, so the
    next call reuses 10000 and neither the correspondence check nor the
    duplicate check sees either record. Widening the format is a change to
    every citation, so it cannot be done as a side effect of one allocation.
    """
    d = _decisions(tmp_path, {"9999": "the-last-one"})
    with pytest.raises(adr.AdrError, match="four digits"):
        adr.allocate(d, "one-too-many")


def test_a_repo_root_equal_to_the_decisions_dir_is_refused(tmp_path: Path):
    """Containment alone was not enough, because the pathspec was hardcoded.

    With ``repo_root == decisions``, ``git ls-tree origin/main decisions/``
    looked for ``decisions/decisions/``, which exits 0 with no output. That is
    indistinguishable from a repository holding no records, so allocation
    reported the bare ``origin/main`` source for a read that covered nothing.
    """
    d = _decisions(tmp_path, {"0001": "first"})
    with pytest.raises(adr.AdrError, match="subdirectory"):
        adr.allocate(d, "second", repo_root=d)


def test_a_nested_decisions_dir_reads_its_own_path_on_the_remote(tmp_path: Path):
    """The pathspec is derived from the caller's directory, not assumed.

    A hardcoded ``decisions/`` resolves against ``repo_root``, so a corpus at
    ``docs/decisions`` had its remote numbers read from a sibling path (or from
    nothing) while the report still said origin/main was consulted.
    """
    clone = _git_repo_with_remote(
        tmp_path, {"0001": "first", "0042": "merged-while-i-worked"}, subdir="docs/decisions"
    )
    decisions = clone / "docs" / "decisions"
    (decisions / "0042-merged-while-i-worked.md").unlink()
    (decisions / adr.INDEX_NAME).write_text(INDEX_HEADER + "- 0001  first\n", encoding="utf-8")

    allocation = adr.allocate(decisions, "next", repo_root=clone)
    assert "origin/main" in allocation.sources
    assert allocation.number == "0043", (
        "the remote read must cover docs/decisions, not a hardcoded decisions/"
    )


def test_a_repo_root_that_does_not_own_the_decisions_dir_is_refused(tmp_path: Path):
    """Otherwise the remote read answers about a different repository.

    Every repo in this workspace numbers from 0001, so a mismatch borrows
    another repo's highest number, and the report still says origin/main was
    consulted. A check that covered the wrong corpus is worse than no check.
    """
    (tmp_path / "repo-a").mkdir()
    (tmp_path / "repo-b").mkdir()
    d = _decisions(tmp_path / "repo-a", {"0001": "first"})
    with pytest.raises(adr.AdrError, match="does not contain"):
        adr.allocate(d, "second", repo_root=tmp_path / "repo-b")


def test_a_claim_that_lands_after_our_read_is_not_lost(tmp_path: Path, monkeypatch):
    """The append must not be a read-modify-write.

    Two sessions share a checkout in this workspace, so the interleaving is
    reachable: A reads the index, B appends its claim and commits, then A
    writes back the text it read. A read-modify-write drops B's line while
    both calls report success, leaving two records and one claim, and the index
    then asserts that one of two colliding numbers is unique.

    Two sequential appends do NOT test this: every implementation passes that.
    The defect only appears when a write lands between our read and our write,
    so this hooks the implementation's own read of the index and injects one
    there.
    """
    d = _decisions(tmp_path, {"0001": "first"})
    index = d / adr.INDEX_NAME
    competitor = "- 0002  raced-us-here\n"
    fired: list[str] = []

    def interpose(original):
        def wrapper(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if self == index and not fired:
                fired.append(original.__name__)
                with index.open("a", encoding="utf-8") as handle:
                    handle.write(competitor)
            return result

        return wrapper

    monkeypatch.setattr(Path, "read_bytes", interpose(Path.read_bytes))
    monkeypatch.setattr(Path, "read_text", interpose(Path.read_text))

    adr._append_index_entry(d, "0003", "third")

    assert fired, (
        "the implementation no longer reads the index before writing it, so "
        "this test's hook never fired and proves nothing; rewrite it against "
        "whatever the new write path is"
    )
    text = index.read_text(encoding="utf-8")
    assert competitor in text, (
        f"a claim written after our read was overwritten (read hook: {fired}); "
        "the write must extend the file rather than reproduce what it read"
    )
    assert text.endswith("- 0003  third\n")


def test_an_index_with_no_trailing_newline_still_gets_a_whole_line(tmp_path: Path):
    d = _decisions(tmp_path, {"0001": "first"})
    index = d / adr.INDEX_NAME
    index.write_text(index.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    adr._append_index_entry(d, "0002", "second")
    lines = index.read_text(encoding="utf-8").splitlines()
    assert lines[-2:] == ["- 0001  first", "- 0002  second"]


def test_two_branches_appending_a_claim_actually_conflict(tmp_path: Path):
    """The central claim of the whole design, tested against real git.

    Every other test here checks a checked-out corpus. None of them prove that
    two competing appends stop a merge, which is the only reason the index
    exists. A regression from append-at-end to a merge-clean edit would
    otherwise ship silently.
    """
    git = shutil.which("git")
    if git is None:  # pragma: no cover
        pytest.skip("git is not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    run = functools.partial(subprocess.run, cwd=repo, capture_output=True, text=True)

    def ok(*argv: str):
        proc = run([git, *argv])
        assert proc.returncode == 0, proc.stderr
        return proc

    ok("init", "-q", "-b", "main")
    ok("config", "user.email", "tests@example.invalid")
    ok("config", "user.name", "tests")
    decisions = repo / "decisions"
    decisions.mkdir()
    (decisions / adr.INDEX_NAME).write_text(
        INDEX_HEADER + "- 0001  first\n", encoding="utf-8"
    )
    ok("add", "-A")
    ok("commit", "-qm", "base")

    # Two branches, each unaware of the other, each taking the next number.
    ok("checkout", "-q", "-b", "branch-a")
    adr.create(decisions, "from-a", "From A", [], [])
    ok("add", "-A")
    ok("commit", "-qm", "a")

    ok("checkout", "-q", "main")
    ok("checkout", "-q", "-b", "branch-b")
    adr.create(decisions, "from-b", "From B", [], [])
    ok("add", "-A")
    ok("commit", "-qm", "b")

    both = sorted(p.name for p in decisions.glob("0002-*.md"))
    assert both == ["0002-from-b.md"], "each branch allocated 0002 in isolation"

    merge = run([git, "merge", "--no-commit", "branch-a"])
    assert merge.returncode != 0, (
        "two claims on 0002 merged cleanly, which is the exact defect the "
        f"index exists to prevent: {merge.stdout} {merge.stderr}"
    )
    conflicted = run([git, "diff", "--name-only", "--diff-filter=U"]).stdout.split()
    assert "decisions/INDEX.md" in conflicted, (
        f"the merge stopped, but not on the index: {conflicted}"
    )


def test_two_different_placement_conventions_merge_clean(tmp_path: Path):
    """What the number-order rule is actually buying.

    The conflict does not come from appending as such: two claims on one
    number land in the same sorted position too, so either rule works on its
    own. What breaks the mechanism is two authors using DIFFERENT rules, which
    puts the same number in two distant places in the file and merges cleanly.
    That is why the guard test checks order rather than trusting everyone to
    append, and it is the case the post-merge duplicate check still has to
    catch.
    """
    git = shutil.which("git")
    if git is None:  # pragma: no cover
        pytest.skip("git is not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    run = functools.partial(subprocess.run, cwd=repo, capture_output=True, text=True)

    def ok(*argv: str):
        proc = run([git, *argv])
        assert proc.returncode == 0, proc.stderr
        return proc

    ok("init", "-q", "-b", "main")
    ok("config", "user.email", "tests@example.invalid")
    ok("config", "user.name", "tests")
    decisions = repo / "decisions"
    decisions.mkdir()
    index = decisions / adr.INDEX_NAME
    # Enough distance between the two placements that git's three-way merge
    # sees two independent hunks.
    body = "".join(f"- {n:04d}  filler-{n}\n" for n in range(1, 41))
    index.write_text(INDEX_HEADER + body, encoding="utf-8")
    ok("add", "-A")
    ok("commit", "-qm", "base")

    ok("checkout", "-q", "-b", "appends-at-the-end")
    (decisions / "0041-from-a.md").write_text("a\n", encoding="utf-8")
    with index.open("a", encoding="utf-8") as handle:
        handle.write("- 0041  from-a\n")
    ok("add", "-A")
    ok("commit", "-qm", "a")

    ok("checkout", "-q", "main")
    ok("checkout", "-q", "-b", "inserts-near-the-top")
    (decisions / "0041-from-b.md").write_text("b\n", encoding="utf-8")
    text = index.read_text(encoding="utf-8")
    index.write_text(
        text.replace("- 0002  filler-2\n", "- 0002  filler-2\n- 0041  from-b\n"),
        encoding="utf-8",
    )
    ok("add", "-A")
    ok("commit", "-qm", "b")

    merge = run([git, "merge", "--no-commit", "appends-at-the-end"])
    assert merge.returncode == 0, (
        "if this now conflicts, git is separating the hunks less aggressively "
        "than when this was written and the order rule matters less"
    )
    merged = index.read_text(encoding="utf-8")
    assert merged.count("- 0041  ") == 2, (
        "both claims on 0041 survived the merge, which is the degradation the "
        "order test exists to prevent"
    )
    # And this is what the retained post-merge duplicate check is for.
    numbers = [
        line.split()[1] for line in merged.splitlines() if line.startswith("- ")
    ]
    assert numbers.count("0041") == 2


def test_the_record_files_alone_would_have_merged_clean(tmp_path: Path):
    """The control arm: without the index there is no conflict to have.

    This is the defect, reproduced. It is what makes the test above meaningful
    rather than a demonstration that git conflicts on something.
    """
    git = shutil.which("git")
    if git is None:  # pragma: no cover
        pytest.skip("git is not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    run = functools.partial(subprocess.run, cwd=repo, capture_output=True, text=True)

    def ok(*argv: str):
        proc = run([git, *argv])
        assert proc.returncode == 0, proc.stderr
        return proc

    ok("init", "-q", "-b", "main")
    ok("config", "user.email", "tests@example.invalid")
    ok("config", "user.name", "tests")
    decisions = repo / "decisions"
    decisions.mkdir()
    (decisions / "0001-first.md").write_text("x\n", encoding="utf-8")
    ok("add", "-A")
    ok("commit", "-qm", "base")

    ok("checkout", "-q", "-b", "branch-a")
    (decisions / "0002-from-a.md").write_text("a\n", encoding="utf-8")
    ok("add", "-A")
    ok("commit", "-qm", "a")

    ok("checkout", "-q", "main")
    ok("checkout", "-q", "-b", "branch-b")
    (decisions / "0002-from-b.md").write_text("b\n", encoding="utf-8")
    ok("add", "-A")
    ok("commit", "-qm", "b")

    merge = run([git, "merge", "--no-commit", "branch-a"])
    assert merge.returncode == 0, (
        "if this now conflicts, git has learned something about our filenames "
        "and the index may no longer be necessary"
    )
    assert sorted(p.name for p in decisions.glob("0002-*.md")) == [
        "0002-from-a.md",
        "0002-from-b.md",
    ], "both records land on 0002 with nothing having objected"


def test_a_slug_reserved_by_an_index_entry_is_refused(tmp_path: Path):
    """The entry reserves the number AND names a slug.

    Taking the slug because its file is not written yet produces two claims on
    one slug, which the `name:` frontmatter and every wikilink assume is
    unique, and which no corpus check saw before this.
    """
    d = _decisions(tmp_path, {"0001": "first"})
    index = d / adr.INDEX_NAME
    index.write_text(
        index.read_text(encoding="utf-8") + "- 0009  reserved-but-unwritten\n",
        encoding="utf-8",
    )
    with pytest.raises(adr.AdrError, match="already exists"):
        adr.allocate(d, "reserved-but-unwritten")


def test_a_number_claimed_twice_in_one_checkout_is_refused(tmp_path: Path):
    """Two runs in ONE checkout never reach a merge, so nothing conflicts.

    The index mechanism only fires across branches. Same worktree means both
    runs would report success on one number and the duplicate could be
    committed and pushed before any test ran, since CI is disabled here. The
    post-append re-read is what makes the loser refuse.
    """
    d = _decisions(tmp_path, {"0001": "first"})
    index = d / adr.INDEX_NAME
    real_append = adr._append_index_entry

    def racing_append(decisions: Path, number: str, slug: str) -> None:
        # The competing session appends the same number first.
        with (decisions / adr.INDEX_NAME).open("a", encoding="utf-8") as handle:
            handle.write(f"- {number}  someone-elses-record\n")
        real_append(decisions, number, slug)

    original = adr._append_index_entry
    adr._append_index_entry = racing_append
    try:
        with pytest.raises(adr.AdrError, match="claimed twice"):
            adr.create(d, "mine", "Mine", [], [])
    finally:
        adr._append_index_entry = original

    assert not (d / "0002-mine.md").exists(), (
        "the losing run must not leave its record behind"
    )
    assert index.read_text(encoding="utf-8").count("- 0002  ") == 2, (
        "the two competing lines are left for a human; the refusal says so"
    )


def test_an_unindexed_record_gets_the_repair_instruction_not_a_slug_refusal(
    tmp_path: Path,
):
    """The state an interrupted run leaves behind has a specific remedy.

    "supersede it or choose a distinct slug" is wrong when the record is
    sitting there missing only its index line, and following it would allocate
    a second number for one decision.
    """
    d = _decisions(tmp_path, {"0001": "first"})
    (d / "0002-half-done.md").write_text("x\n", encoding="utf-8")
    with pytest.raises(adr.AdrError, match="has no INDEX.md entry") as caught:
        adr.create(d, "half-done", "Half done", [], [])
    assert "- 0002  half-done" in str(caught.value), (
        "the message must name the exact line to append"
    )


@pytest.mark.parametrize("bad", ["a: b", "[x]", "- y", "", "with space"])
def test_a_repo_or_tag_that_would_change_the_frontmatter_shape_is_refused(
    tmp_path: Path, bad: str
):
    """These go into an unquoted YAML flow sequence.

    `--repos "a: b"` yields `repos: [a: b]`, a list containing a mapping. The
    frontmatter guard checks for the key, not its type, so the record passes
    and the indexer mis-reads it silently.
    """
    d = _decisions(tmp_path, {"0001": "first"})
    with pytest.raises(adr.AdrError, match="plain identifier"):
        adr.create(d, "second", "Second", [bad], [])
    assert not (d / "0002-second.md").exists()
