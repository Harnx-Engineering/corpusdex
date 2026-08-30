"""Coverage for link extraction (see corpusdex.links).

Extraction is pure text handling: no database, no resolution. Resolving a
target to a document id, and dropping the ones that resolve to nothing, is
exercised in test_graph.py against a real index.
"""

from __future__ import annotations

from corpusdex.links import (
    RELATION_LINK,
    RELATION_SUPERSEDED_BY,
    extract_body_links,
    extract_links,
    normalize_target,
    resolution_keys,
    strip_code,
)


def test_wikilinks_are_extracted():
    assert extract_body_links("see [[garden-notes]] for more") == ["garden-notes"]


def test_wikilink_alias_and_anchor_are_stripped_to_the_target():
    # Both forms name the same document, so they collapse to one target.
    body = "[[garden-notes|the notes]] and [[garden-notes#watering]]"
    assert extract_body_links(body) == ["garden-notes"]


def test_markdown_inline_links_are_extracted():
    assert extract_body_links("see [the notes](garden/notes.md)") == ["garden/notes.md"]


def test_markdown_link_anchor_is_stripped():
    assert extract_body_links("[x](garden/notes.md#watering)") == ["garden/notes.md"]


def test_bare_relative_paths_are_extracted():
    # The form the corpus actually writes today; without it the graph would
    # be empty on the real corpus.
    assert extract_body_links("described in garden/notes.md fully") == ["garden/notes.md"]


def test_links_inside_fenced_code_are_ignored():
    body = "real [[alpha]]\n\n```\nsample [[beta]] and garden/notes.md\n```\n\ntail"
    assert extract_body_links(body) == ["alpha"]


def test_links_inside_inline_code_are_ignored():
    assert extract_body_links("use `[[beta]]` literally") == []


def test_body_link_order_is_first_seen_and_deduplicated_by_relation():
    links = extract_links({}, "[[alpha]] then [[beta]] then [[alpha]]")
    assert [link.target for link in links] == ["alpha", "beta"]
    assert {link.relation for link in links} == {RELATION_LINK}


def test_superseded_by_frontmatter_is_a_link():
    links = extract_links({"superseded_by": "0002-newer"}, "")
    assert links == [type(links[0])(target="0002-newer", relation=RELATION_SUPERSEDED_BY)]


def test_null_superseded_by_is_not_a_link():
    for value in (None, "", "null", "none", "~"):
        assert extract_links({"superseded_by": value}, "") == []


def test_repos_and_tags_frontmatter_are_not_links():
    # They name repositories and topics, not documents. Wiring them as edges
    # would collapse the graph into a few giant hubs carrying no
    # document-to-document meaning.
    links = extract_links({"repos": ["repo-a", "repo-b"], "tags": ["alpha", "beta"]}, "")
    assert links == []


def test_frontmatter_and_body_links_coexist_under_distinct_relations():
    links = extract_links({"superseded_by": "0002-newer"}, "see [[garden-notes]]")
    assert [(link.target, link.relation) for link in links] == [
        ("0002-newer", RELATION_SUPERSEDED_BY),
        ("garden-notes", RELATION_LINK),
    ]


def test_strip_code_preserves_line_count():
    text = "a\n```\nb\nc\n```\nd"
    assert len(strip_code(text).splitlines()) == len(text.splitlines())


def test_resolution_keys_cover_path_repo_relative_name_stem_and_title():
    keys = resolution_keys("repo-a", "repo-a/garden/notes.md", "Garden Notes")
    assert "repo-a/garden/notes.md" in keys
    assert "garden/notes.md" in keys
    assert "notes.md" in keys
    assert "notes" in keys
    assert "garden notes" in keys


def test_resolution_keys_are_lowercased_and_deduplicated():
    keys = resolution_keys("repo-a", "repo-a/NOTES.md", "NOTES.md")
    assert keys == [k.lower() for k in keys]
    assert len(keys) == len(set(keys))


def test_normalize_target_strips_leading_path_noise_and_case():
    assert normalize_target("./Garden/Notes.md") == "garden/notes.md"
    assert normalize_target("/garden/notes.md") == "garden/notes.md"
