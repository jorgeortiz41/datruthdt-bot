"""The Source contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import PersonaConfig, SourceConfig
from ..store import CorpusStore


@dataclass
class ToolResult:
    """What a source hands back to the model."""

    text: str
    # Human-readable citations surfaced in the Discord footer.
    citations: list[str] = field(default_factory=list)
    is_error: bool = False


class Source:
    """Base class. Subclasses override what they support."""

    #: Set True if this source contributes documents to the local corpus.
    ingests: bool = False

    def __init__(self, cfg: SourceConfig, persona: PersonaConfig, store: CorpusStore):
        self.cfg = cfg
        self.persona = persona
        self.store = store

    # -- identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        """Corpus `source` column value / log label."""
        return self.cfg.type

    @property
    def tool_name(self) -> str:
        return self.cfg.tool_name or f"search_{self.cfg.type}"

    # -- ingestion ------------------------------------------------------------

    def ingest(self, *, limit: int | None = None, refresh: bool = False) -> dict[str, Any]:
        """Pull documents into the corpus. Returns a summary dict."""
        raise NotImplementedError(f"{self.name} does not support ingestion")

    # -- question-time tool ---------------------------------------------------

    def tool_spec(self) -> dict[str, Any] | None:
        """An Anthropic tool definition, or None if this source has no tool."""
        return None

    def run_tool(self, tool_input: dict[str, Any]) -> ToolResult:
        raise NotImplementedError(f"{self.name} exposes no tool")

    # -- diagnostics ----------------------------------------------------------

    def health(self) -> str:
        """One-line status shown by `creatorbot doctor`."""
        return "ok"
