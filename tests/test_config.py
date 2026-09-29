from __future__ import annotations

import pytest

from corpusdex import config


def test_the_canonical_name_is_read(monkeypatch):
    monkeypatch.setenv("BRAIN_EMBED_MODEL", "new-model")
    assert config.env("BRAIN_EMBED_MODEL") == "new-model"
    assert config.env_source("BRAIN_EMBED_MODEL") == "BRAIN_EMBED_MODEL"


def test_the_legacy_alias_is_honoured_when_the_canonical_name_is_unset(monkeypatch):
    monkeypatch.delenv("BRAIN_EMBED_MODEL", raising=False)
    monkeypatch.setenv("HARNX_BRAIN_EMBED_MODEL", "old-model")
    assert config.env("BRAIN_EMBED_MODEL") == "old-model"
    assert config.env_source("BRAIN_EMBED_MODEL") == "HARNX_BRAIN_EMBED_MODEL"


def test_the_canonical_name_wins_over_a_stale_legacy_value(monkeypatch):
    """Precedence direction matters: the migrated caller is the one making a
    deliberate choice, so a leftover legacy value must never override it."""
    monkeypatch.setenv("BRAIN_EMBED_MODEL", "new-model")
    monkeypatch.setenv("HARNX_BRAIN_EMBED_MODEL", "old-model")
    assert config.env("BRAIN_EMBED_MODEL") == "new-model"


def test_an_empty_canonical_value_is_set_not_absent(monkeypatch):
    """``BRAIN_EMBED_MODEL=`` is a caller clearing the value. Falling through
    to the alias would resurrect the setting they just cleared."""
    monkeypatch.setenv("BRAIN_EMBED_MODEL", "")
    monkeypatch.setenv("HARNX_BRAIN_EMBED_MODEL", "old-model")
    assert config.env("BRAIN_EMBED_MODEL") == ""


def test_the_default_is_returned_when_neither_name_is_set(monkeypatch):
    monkeypatch.delenv("BRAIN_EMBED_MODEL", raising=False)
    monkeypatch.delenv("HARNX_BRAIN_EMBED_MODEL", raising=False)
    assert config.env("BRAIN_EMBED_MODEL", "fallback") == "fallback"
    assert config.env("BRAIN_EMBED_MODEL") is None
    assert config.env_source("BRAIN_EMBED_MODEL") is None


def test_a_setting_with_no_alias_still_resolves(monkeypatch):
    """Not every setting has a legacy name; those must not raise."""
    monkeypatch.setenv("BRAIN_CORPUS_ROOTS", "/tmp/a")
    assert config.env("BRAIN_CORPUS_ROOTS") == "/tmp/a"
    monkeypatch.delenv("BRAIN_CORPUS_ROOTS")
    assert config.env("BRAIN_CORPUS_ROOTS") is None


@pytest.mark.parametrize("name", ["HARNX_BRAIN_DB", "DB", "brain_db", ""])
def test_a_name_outside_the_prefix_is_rejected(name):
    """Guards against reintroducing a raw legacy read through this helper,
    which would make the alias table look complete while bypassing it."""
    with pytest.raises(ValueError):
        config.env(name)
    with pytest.raises(ValueError):
        config.env_source(name)


def test_the_workspace_root_alias_is_not_a_mechanical_prefix_swap():
    """The one setting that decides what gets indexed is also the one whose
    legacy name breaks the pattern, so a computed rewrite would miss it."""
    assert config.LEGACY_ENV_ALIASES["BRAIN_WORKSPACE_ROOT"] == "HARNX_WORKSPACE_ROOT"
    assert "HARNX_BRAIN_WORKSPACE_ROOT" not in config.LEGACY_ENV_ALIASES.values()


def test_legacy_names_in_use_reports_only_aliases_actually_supplying_a_value(monkeypatch):
    for canonical, legacy in config.LEGACY_ENV_ALIASES.items():
        monkeypatch.delenv(canonical, raising=False)
        monkeypatch.delenv(legacy, raising=False)
    assert config.legacy_names_in_use() == {}

    monkeypatch.setenv("HARNX_BRAIN_DB", "/tmp/x.db")
    assert config.legacy_names_in_use() == {"BRAIN_DB": "HARNX_BRAIN_DB"}

    # Once the caller migrates, the alias stops being reported even though it
    # is still set -- otherwise the report can never reach empty and cannot be
    # used as the signal to drop the table.
    monkeypatch.setenv("BRAIN_DB", "/tmp/y.db")
    assert config.legacy_names_in_use() == {}


def test_every_alias_maps_a_distinct_legacy_name():
    legacy = list(config.LEGACY_ENV_ALIASES.values())
    assert len(legacy) == len(set(legacy))
