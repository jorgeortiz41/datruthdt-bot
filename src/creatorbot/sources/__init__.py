"""Knowledge sources.

A source does one or both of:

  * **ingest** — pull documents into the local corpus (`ingest()`), and/or
  * **answer** — expose a tool Claude can call at question time (`tool_spec()`
    + `run_tool()`).

Adding a source for a new site means writing one class here and adding an entry
to the persona YAML. Nothing else changes.
"""

from __future__ import annotations

from typing import Any

from ..config import PersonaConfig, SourceConfig
from .base import Source, ToolResult
from .cardsite import CardSiteSource
from .mediawiki import MediaWikiSource
from .youtube import YouTubeTranscriptSource

# Persona `type:` value -> implementation.
REGISTRY: dict[str, type[Source]] = {
    "youtube_transcripts": YouTubeTranscriptSource,
    "mediawiki": MediaWikiSource,
    # Both community card databases share one implementation; they differ only
    # in URL shapes and access policy, which come from the persona file.
    "dokkaninfo": CardSiteSource,
    "dokkandb": CardSiteSource,
    "cardsite": CardSiteSource,
}

# `web_search` is handled by the Anthropic server-side tool in engine.py rather
# than by a Source class, so it is deliberately absent from REGISTRY.
SERVER_SIDE_TYPES = {"web_search"}


def build_sources(persona: PersonaConfig, store: Any) -> list[Source]:
    """Instantiate every enabled, client-side source in the persona."""
    sources: list[Source] = []
    for cfg in persona.enabled_sources():
        if cfg.type in SERVER_SIDE_TYPES:
            continue
        impl = REGISTRY.get(cfg.type)
        if impl is None:
            known = sorted(set(REGISTRY) | SERVER_SIDE_TYPES)
            raise ValueError(
                f"Unknown source type {cfg.type!r} in {persona.path.name}. "
                f"Known types: {', '.join(known)}"
            )
        sources.append(impl(cfg, persona, store))
    return sources


def web_search_config(persona: PersonaConfig) -> SourceConfig | None:
    for cfg in persona.enabled_sources():
        if cfg.type == "web_search":
            return cfg
    return None


__all__ = [
    "REGISTRY",
    "Source",
    "ToolResult",
    "build_sources",
    "web_search_config",
]
