"""Repo-hygiene checks over every tracked text file in this repository.

The no-em-dash rule is stated in this repository's agent instructions, but
prose is only followed by whoever reads it: by
the time issue #90 was worked, one file with 45 em-dashes had become seven
files with 180. This test is the part that runs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# Built from its code point so this file never contains the character it bans.
EM_DASH = chr(0x2014)


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    paths = [REPO / name for name in out.decode("utf-8").split("\0") if name]
    text = []
    for path in paths:
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        text.append(path)
    return text


def test_the_scan_reads_the_corpus():
    # A scan that reads no files prints what a clean scan prints, so pin that
    # it saw the directories the rule is about. The corpus directories are
    # pinned only where they exist: the public engine snapshot ships src and
    # tests without them (decision 0040), and this file ships with it.
    rels = {p.relative_to(REPO).parts[0] for p in _tracked_text_files()}
    expected = {"src", "tests"} | {d for d in ("context", "decisions") if (REPO / d).is_dir()}
    assert expected <= rels


def test_no_tracked_file_contains_an_em_dash():
    hits = []
    for path in _tracked_text_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if EM_DASH in line:
                hits.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not hits, (
        "em-dash (U+2014) found; use a spaced hyphen, colon or comma:\n  "
        + "\n  ".join(hits)
    )


# Ask the canonical exporter which current-index paths survive its projection;
# scan their working content so staged and unstaged edits are still checked.
# A public engine checkout has no exporter and every tracked file already ships.
# Built from parts so this file does not match its own pattern. The one allowed
# form is the legacy registry filename, which is a supported setting.
_PRIVATE_REPO_NAME = re.compile("harnx" + r"-(?!repos\.tsv)[a-z]")
# Further private names live in a terms file that is never published, so the
# names themselves do not ship in this test. A list written here, even
# assembled from string pieces, IS the leak: 0.2.0 shipped one that way. In a
# checkout without the file only the repository-name check runs.
_PRIVATE_TERMS_FILE = REPO / "scripts" / "public-audit-terms.txt"


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _private_terms() -> list[str]:
    if not _PRIVATE_TERMS_FILE.is_file():
        return []
    lines = _PRIVATE_TERMS_FILE.read_text(encoding="utf-8").splitlines()
    return [_squash(term) for line in lines if (term := line.split("#", 1)[0].strip())]


def _shipped_text_files() -> list[Path]:
    builder = REPO / "scripts" / "build-public-snapshot.sh"
    if not builder.is_file():
        assert not any((REPO / name).exists() for name in (
            "decisions", "context", "gaps", "research", "architecture", "eval",
        )), (
            "private corpus checkout is missing its snapshot membership authority"
        )
        return _tracked_text_files()
    result = subprocess.run(
        ["bash", str(builder), "--list-shipped"],
        cwd=REPO, check=True, capture_output=True,
    )
    shipped = {Path(name) for name in result.stdout.decode("utf-8").split("\0") if name}
    assert shipped, "snapshot membership authority returned no paths"
    return [path for path in _tracked_text_files() if path.relative_to(REPO) in shipped]


def test_shipped_files_name_no_private_repository():
    terms = _private_terms()
    hits = []
    for path in _shipped_text_files():
        rel = path.relative_to(REPO)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            squashed = _squash(line)
            if _PRIVATE_REPO_NAME.search(line) or any(term in squashed for term in terms):
                hits.append(f"{rel}:{lineno}")
    assert not hits, (
        "a private repository or product component is named in a file the public "
        "snapshot ships:\n  "
        + "\n  ".join(hits)
    )


def test_the_private_terms_are_read_where_the_file_exists():
    """A terms check that read no terms passes every file, so it must prove it
    read some. Skipped in a checkout without the file, by design."""
    if not _PRIVATE_TERMS_FILE.is_file():
        pytest.skip("no private terms file in this checkout")
    terms = _private_terms()
    assert terms and all(len(term) >= 5 for term in terms)


def test_membership_matches_actual_export_and_checks_new_index_paths(tmp_path, monkeypatch):
    builder = REPO / "scripts" / "build-public-snapshot.sh"
    if not builder.is_file():
        pytest.skip("public snapshot has no private exporter")
    fixture = tmp_path / "source"
    fixture.mkdir()
    script = fixture / "scripts" / builder.name
    script.parent.mkdir()
    script.write_text(builder.read_text(encoding="utf-8"), encoding="utf-8")
    private_name = "harnx" + "-privatefixture"
    for name, text in {
        "src/engine.py": "VALUE = 1\n",
        "tests/test_engine.py": "assert True\n",
        "tests/with space.py": "assert True\n",
        "tests/with\nnewline.py": "assert True\n",
        "tests/test_audit_public_snapshot.py": private_name,
        "tests/test_decision_records.py": private_name,
        "tests/cache.db": private_name,
        "decisions/private.md": private_name,
        "README.md": "Synthetic engine\n",
    }.items():
        path = fixture / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(fixture)], check=True)
    subprocess.run(["git", "-C", str(fixture), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(fixture), "-c", "user.name=Fixture", "-c",
         "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
         "-c", "core.hooksPath=/dev/null", "commit", "-qm", "synthetic snapshot"],
        check=True,
    )
    output = tmp_path / "snapshot"
    subprocess.run(["bash", str(script), str(output)], cwd=fixture, check=True, capture_output=True)
    actual = {path.relative_to(output) for path in output.rglob("*") if path.is_file()}
    monkeypatch.setitem(globals(), "REPO", fixture)
    selected = {path.relative_to(fixture) for path in _shipped_text_files()}
    assert selected == actual
    test_shipped_files_name_no_private_repository()

    # A new staged path is absent from HEAD but must still be checked against
    # its current content. Otherwise a new source file can bypass the guard.
    added = fixture / "tests" / "new_engine.py"
    added.write_text(private_name, encoding="utf-8")
    subprocess.run(["git", "-C", str(fixture), "add", str(added)], check=True)
    assert added in _shipped_text_files()
    with pytest.raises(AssertionError, match="private repository or product"):
        test_shipped_files_name_no_private_repository()
    added.write_text("assert True\n", encoding="utf-8")
    test_shipped_files_name_no_private_repository()

    # Removing the exporter's real exclusion changes membership immediately;
    # a duplicated exclusion table in this test would hide the leaked fixture.
    removal = 'rm -f "$out/tests/test_audit_public_snapshot.py"'
    source = script.read_text(encoding="utf-8")
    assert source.count(removal) == 1
    script.write_text(source.replace(removal, ""), encoding="utf-8")
    with pytest.raises(AssertionError, match="private repository or product"):
        test_shipped_files_name_no_private_repository()


@pytest.mark.parametrize("corpus_root", [
    "decisions", "context", "gaps", "research", "architecture", "eval",
])
def test_missing_private_membership_authority_fails_closed(tmp_path, monkeypatch, corpus_root):
    (tmp_path / corpus_root).mkdir()
    monkeypatch.setitem(globals(), "REPO", tmp_path)
    with pytest.raises(AssertionError, match="missing its snapshot membership authority"):
        _shipped_text_files()


@pytest.mark.parametrize("listing", [False, True])
def test_snapshot_projection_refuses_symlink_removal_ancestor(tmp_path, listing):
    builder = REPO / "scripts" / "build-public-snapshot.sh"
    if not builder.is_file():
        pytest.skip("public snapshot has no private exporter")
    fixture = tmp_path / "source"
    fixture.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinels = [outside / name for name in (
        "test_decision_records.py", "test_audit_public_snapshot.py",
    )]
    for sentinel in sentinels:
        sentinel.write_text("outside sentinel", encoding="utf-8")
    (fixture / "tests").symlink_to(outside, target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(fixture)], check=True)
    subprocess.run(["git", "-C", str(fixture), "add", "tests"], check=True)
    subprocess.run(
        ["git", "-C", str(fixture), "-c", "user.name=Fixture", "-c",
         "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
         "-c", "core.hooksPath=/dev/null", "commit", "-qm", "synthetic symlink"],
        check=True,
    )
    argument = "--list-shipped" if listing else str(tmp_path / "snapshot")
    run = subprocess.run(["bash", str(builder), argument], cwd=fixture, capture_output=True)
    assert all(path.is_file() and path.read_text(encoding="utf-8") == "outside sentinel"
               for path in sentinels), "snapshot projection modified files outside its archive"
    assert run.returncode != 0
    assert b"must not be a symbolic link" in run.stderr
