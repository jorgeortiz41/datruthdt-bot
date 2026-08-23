"""Persona config loading.

One YAML file describes the creator, the subject matter and the knowledge
sources. Everything downstream reads this, so swapping personas never means
touching Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# repo_root/src/creatorbot/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_DIR = REPO_ROOT / "personas"
DATA_DIR = REPO_ROOT / "data"


class ConfigError(RuntimeError):
    """Raised when a persona file is missing or malformed."""


@dataclass(frozen=True)
class SourceConfig:
    """One knowledge source entry from the persona file."""

    type: str
    enabled: bool = True
    tool_name: str | None = None
    description: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceConfig":
        if "type" not in raw:
            raise ConfigError(f"source entry is missing 'type': {raw!r}")
        known = {"type", "enabled", "tool_name", "description"}
        return cls(
            type=raw["type"],
            enabled=bool(raw.get("enabled", True)),
            tool_name=raw.get("tool_name"),
            description=(raw.get("description") or "").strip(),
            options={k: v for k, v in raw.items() if k not in known},
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)


@dataclass(frozen=True)
class PersonaConfig:
    """A fully-loaded persona."""

    id: str
    display_name: str
    tagline: str
    raw: dict[str, Any]
    path: Path

    # -- sections -------------------------------------------------------------

    @property
    def disclosure(self) -> dict[str, Any]:
        return self.raw.get("disclosure", {}) or {}

    @property
    def creator(self) -> dict[str, Any]:
        return self.raw.get("creator", {}) or {}

    @property
    def voice(self) -> dict[str, Any]:
        return self.raw.get("voice", {}) or {}

    @property
    def domain(self) -> dict[str, Any]:
        return self.raw.get("domain", {}) or {}

    @property
    def answering(self) -> dict[str, Any]:
        return self.raw.get("answering", {}) or {}

    @property
    def discord(self) -> dict[str, Any]:
        return self.raw.get("discord", {}) or {}

    @property
    def sources(self) -> list[SourceConfig]:
        entries = self.raw.get("sources", []) or []
        return [SourceConfig.from_dict(e) for e in entries]

    def enabled_sources(self) -> list[SourceConfig]:
        return [s for s in self.sources if s.enabled]

    def source_of_type(self, type_: str) -> SourceConfig | None:
        return next((s for s in self.sources if s.type == type_), None)

    # -- resolved settings ----------------------------------------------------

    @property
    def model(self) -> str:
        return os.getenv("CREATORBOT_MODEL") or self.answering.get(
            "model", "claude-opus-5"
        )

    @property
    def effort(self) -> str:
        return os.getenv("CREATORBOT_EFFORT") or self.answering.get("effort", "medium")

    @property
    def max_tokens(self) -> int:
        return int(self.answering.get("max_tokens", 4000))

    @property
    def max_tool_iterations(self) -> int:
        return int(self.answering.get("max_tool_iterations", 6))

    # -- paths ----------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """Per-persona corpus, index and generated artefacts."""
        d = DATA_DIR / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def db_path(self) -> Path:
        return self.data_dir / "corpus.sqlite3"

    @property
    def style_profile_path(self) -> Path:
        return self.data_dir / "style_profile.md"

    @property
    def exemplars_path(self) -> Path:
        return self.data_dir / "style_exemplars.json"


def load_persona(name: str | None = None) -> PersonaConfig:
    """Load a persona by name (defaults to $CREATORBOT_PERSONA, then 'datruthdt')."""
    name = name or os.getenv("CREATORBOT_PERSONA") or "datruthdt"

    # Accept a bare name, a filename, or a full path.
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        path = candidate
    else:
        path = PERSONA_DIR / f"{name}.yaml"

    if not path.exists():
        available = sorted(p.stem for p in PERSONA_DIR.glob("*.yaml"))
        raise ConfigError(
            f"No persona named {name!r} (looked for {path}). "
            f"Available: {', '.join(available) or 'none'}"
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    for required in ("id", "display_name"):
        if not raw.get(required):
            raise ConfigError(f"{path} is missing required key {required!r}")

    return PersonaConfig(
        id=raw["id"],
        display_name=raw["display_name"],
        tagline=raw.get("tagline", ""),
        raw=raw,
        path=path,
    )


def list_personas() -> list[str]:
    return sorted(p.stem for p in PERSONA_DIR.glob("*.yaml"))
