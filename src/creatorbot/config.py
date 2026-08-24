@property
def model(self) -> str:
    return os.getenv("CREATORBOT_MODEL") or self.answering.get("model", "grok-4.6")


@property
def effort(self) -> str:
    # Grok uses reasoning effort differently; we keep the field for compatibility
    return os.getenv("CREATORBOT_EFFORT") or self.answering.get("effort", "high")


@property
def max_tokens(self) -> int:
    return int(self.answering.get("max_tokens", 4000))


@property
def max_tool_iterations(self) -> int:
    return int(self.answering.get("max_tool_iterations", 8))


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
