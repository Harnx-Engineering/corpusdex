"""Environment configuration for the retrieval engine.

Every setting is read through :func:`env` rather than from ``os.environ``
directly, so the engine has exactly one place that knows what a setting is
called. That matters because the names are moving: the engine is being split
out as a generic package, and ``HARNX_*`` is this workspace's branding, not a
property of the tool.

The new names are ``BRAIN_*``. The workspace-era names keep working as
aliases, because the split is not a flag day: a running LaunchAgent, the MCP
registration, several documents, and other agents' shell environments all set
the old names today, and a rename that breaks them the moment it lands would
be reverted rather than finished. The alias table is deliberately explicit
rather than a computed prefix swap, because the old names were not
consistent -- ``HARNX_WORKSPACE_ROOT`` has no ``BRAIN`` in it while every
other setting does, so a mechanical ``HARNX_`` to ``BRAIN_`` rewrite would
silently fail to find exactly the one that decides what gets indexed.

Precedence is new-name-first. A caller that has migrated is never overridden
by a stale legacy value left in the environment, which is the direction that
matters: the migrated caller is the one making a deliberate choice.
"""

from __future__ import annotations

import os

#: Prefix for this engine's settings. Public so callers and tests can build
#: names without hardcoding the string in several places.
ENV_PREFIX = "BRAIN_"

#: Canonical name -> the workspace-era name it replaced. Every entry is a
#: name that shipped and may still be set somewhere; nothing may be added
#: here that was never released, since an alias for a name nobody used is
#: just a second way to get the setting wrong.
LEGACY_ENV_ALIASES: dict[str, str] = {
    "BRAIN_WORKSPACE_ROOT": "HARNX_WORKSPACE_ROOT",
    "BRAIN_DB": "HARNX_BRAIN_DB",
    "BRAIN_STATE_DIR": "HARNX_BRAIN_STATE_DIR",
    "BRAIN_OLLAMA_HOST": "HARNX_BRAIN_OLLAMA_HOST",
    "BRAIN_EMBED_MODEL": "HARNX_BRAIN_EMBED_MODEL",
}


#: Settings that name corpus scope rather than plumbing. Listed here so the
#: set is discoverable from one place; each is read through :func:`env` like
#: any other. None of them has a legacy alias, because none of them existed
#: before the split.
CORPUS_SETTINGS = (
    "BRAIN_CORPUS_ROOTS",
    "BRAIN_EXCLUDE",
    "BRAIN_REGISTRY_FILE",
    "BRAIN_KNOWLEDGE_REPO",
)


def env(name: str, default: str | None = None) -> str | None:
    """Return the value of setting ``name``, honouring its legacy alias.

    ``name`` is the canonical ``BRAIN_*`` name. The legacy alias is consulted
    only when the canonical name is unset, so a migrated caller always wins.

    An empty string is treated as set, not as absent: ``BRAIN_EMBED_MODEL=``
    is a caller deliberately clearing a value, and silently falling through to
    a legacy alias would resurrect the setting they just cleared.
    """
    if not name.startswith(ENV_PREFIX):
        raise ValueError(f"setting names must start with {ENV_PREFIX!r}, got {name!r}")
    value = os.environ.get(name)
    if value is not None:
        return value
    legacy = LEGACY_ENV_ALIASES.get(name)
    if legacy is not None:
        value = os.environ.get(legacy)
        if value is not None:
            return value
    return default


def env_source(name: str) -> str | None:
    """Return which environment variable actually supplied ``name``.

    For diagnostics: ``brain status`` and a support request both need to be
    able to say *which* name was read, because "the setting is not taking
    effect" is usually a caller setting the alias while something else sets
    the canonical name.
    """
    if not name.startswith(ENV_PREFIX):
        raise ValueError(f"setting names must start with {ENV_PREFIX!r}, got {name!r}")
    if os.environ.get(name) is not None:
        return name
    legacy = LEGACY_ENV_ALIASES.get(name)
    if legacy is not None and os.environ.get(legacy) is not None:
        return legacy
    return None


def legacy_names_in_use() -> dict[str, str]:
    """Return ``{canonical: legacy}`` for every setting supplied by an alias.

    Reported rather than warned on, because these are still supported and a
    warning on every invocation would train callers to ignore it. It exists so
    the split can be finished on evidence: when this returns empty across the
    workspace, the alias table can go.
    """
    return {
        canonical: legacy
        for canonical, legacy in LEGACY_ENV_ALIASES.items()
        if os.environ.get(canonical) is None and os.environ.get(legacy) is not None
    }


def env_list(name: str) -> list[str]:
    """Return a list-valued setting, split on :data:`os.pathsep`.

    ``os.pathsep`` rather than a comma, because two of these settings hold
    filesystem paths and a comma is a legal character in one. Empty segments
    are dropped, so a trailing separator and a value of ``""`` both mean "not
    set" rather than "one empty entry" -- the latter would silently become the
    current directory for a path setting and a match-everything pattern for a
    glob one.
    """
    raw = env(name)
    if not raw:
        return []
    return [part.strip() for part in raw.split(os.pathsep) if part.strip()]
