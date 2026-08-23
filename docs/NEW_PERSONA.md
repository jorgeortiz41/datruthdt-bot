# Making this a bot for a different creator

The codebase has no knowledge of DaTruthDT or Dokkan in it. Both live entirely
in `personas/datruthdt.yaml`. This doc covers the two levels of customisation:

1. **Same shape, different creator** — YAML only, no code.
2. **A different knowledge source** — one new class.

---

## 1. A different creator (YAML only)

```bash
cp personas/datruthdt.yaml personas/yourcreator.yaml
$EDITOR personas/yourcreator.yaml
```

### What to change

**Identity**

```yaml
id: yourcreator            # must match the filename
display_name: Your Creator
tagline: "Unofficial fan bot — X takes in Y's voice"
```

**Disclosure** — keep this section. It's what makes the bot honest about being
an impression, and `persona.py` enforces the break-character rule from it.

```yaml
disclosure:
  is_parody: true
  statement: >-
    I'm an unofficial fan-made bot that imitates ...
  break_character_on_identity_question: true
```

**The channel**

```yaml
creator:
  youtube:
    handle: "@TheirHandle"
    channel_id: UCxxxxxxxxxxxxxxxxxxxxxx   # most reliable
    channel_url: https://www.youtube.com/@TheirHandle
    max_videos: 300
    transcript_languages: [en]
    min_duration_seconds: 120   # skip Shorts
```

Find the channel ID by opening any video, **View Source**, and searching for
`"channelId"`. `channel_url` alone works too, but IDs never break on a rename.

**Voice** — write a *conservative* seed. Don't invent catchphrases; the style
profile will find the real ones. Describe:

- `summary` — one paragraph on how they come across
- `traits` — sentence shape, energy, how they address the audience
- `vocabulary` — words you're confident about
- `avoid` — hard limits (topics, claims they'd never make)
- `profanity` — `none` | `light` | `uncensored`

**Domain** — the subject matter and its jargon:

```yaml
domain:
  name: "Magic: The Gathering"
  short_name: MTG
  glossary:
    ETB: "Enters the battlefield"
    "mana curve": "Distribution of spell costs across a deck"
```

The glossary goes straight into the system prompt, so the bot speaks the dialect
without spending a tool call on vocabulary.

**Sources** — swap the wiki for whatever database fits:

```yaml
sources:
  - type: youtube_transcripts
    enabled: true
    tool_name: search_videos
    description: >-
      Search their video transcripts for opinions, takes and advice.

  - type: mediawiki
    enabled: true
    tool_name: search_wiki
    api_url: https://mtg.fandom.com/api.php
    site_name: "MTG Wiki"
    license: "CC BY-SA"
    description: >-
      Look up exact card rules and mechanics.

  - type: web_search
    enabled: true
    max_uses: 4
    allowed_domains: [scryfall.com, mtg.fandom.com, reddit.com]
```

`type: mediawiki` works against **any** MediaWiki site — every Fandom wiki,
Wikipedia, and most game wikis. That covers a large share of "any database"
without writing code.

Delete the `dokkaninfo` entry if it doesn't apply.

### Build it

```bash
export CREATORBOT_PERSONA=yourcreator
creatorbot doctor
creatorbot ingest
creatorbot style-profile
creatorbot ask "test question"
```

Read `data/yourcreator/style_profile.md` afterwards. It's the highest-leverage
thing to eyeball — if it's wrong, the voice will be wrong. Ingest more videos
and re-run with a bigger `--sample` if it's thin.

---

## 2. A different knowledge source (one class)

Sources live in `src/creatorbot/sources/`. The contract is in `base.py`:

```python
class Source:
    ingests: bool = False              # does it write to the corpus?

    def ingest(self, *, limit, refresh) -> dict: ...      # optional
    def tool_spec(self) -> dict | None: ...               # Anthropic tool def
    def run_tool(self, tool_input: dict) -> ToolResult: ...
    def health(self) -> str: ...                          # for `doctor`
```

### A read-through source (queried live)

Most cases. Implement `tool_spec()` + `run_tool()`:

```python
# src/creatorbot/sources/scryfall.py
from typing import Any
import httpx
from .base import Source, ToolResult


class ScryfallSource(Source):
    @property
    def name(self) -> str:
        return "scryfall"

    def tool_spec(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": self.cfg.description or "Look up an MTG card on Scryfall.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "card_name": {"type": "string", "description": "Exact or fuzzy card name"},
                },
                "required": ["card_name"],
            },
        }

    def run_tool(self, tool_input: dict[str, Any]) -> ToolResult:
        name = (tool_input.get("card_name") or "").strip()
        if not name:
            return ToolResult("No card name supplied.", is_error=True)
        try:
            r = httpx.get(
                "https://api.scryfall.com/cards/named",
                params={"fuzzy": name},
                timeout=20.0,
                headers={"User-Agent": "creatorbot/0.1", "Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            return ToolResult(f"Scryfall unreachable: {exc}", is_error=True)
        if r.status_code == 404:
            return ToolResult(f"No card named {name!r}.")
        card = r.json()
        return ToolResult(
            f"{card['name']} — {card.get('mana_cost','')}\n"
            f"{card.get('type_line','')}\n{card.get('oracle_text','')}\n"
            f"source: {card['scryfall_uri']}",
            citations=[f"[{card['name']}]({card['scryfall_uri']})"],
        )
```

Register it in `sources/__init__.py`:

```python
from .scryfall import ScryfallSource

REGISTRY = {
    ...,
    "scryfall": ScryfallSource,
}
```

And enable it in the persona:

```yaml
  - type: scryfall
    enabled: true
    tool_name: lookup_card
    description: Look up exact MTG card text, cost and rulings.
```

That's it — `engine.py` picks it up, builds the tool list and routes calls.

### An ingesting source (bulk into the corpus)

Set `ingests = True` and implement `ingest()`. Write `Document` + `Chunk` objects
via `self.store.upsert_document(doc, chunks)`; use the helpers in `chunking.py`.
Give every document a stable `id` so re-running is idempotent. Follow
`youtube.py` as the reference implementation.

Then `run_tool()` typically just calls `self.store.hybrid_search(query,
source=self.name, ...)`.

---

## Config knobs worth knowing

| Key | Effect |
|---|---|
| `answering.model` | `claude-opus-5` by default; override per-run with `CREATORBOT_MODEL` |
| `answering.effort` | `low`→`max`. `medium` is a good chat default; raise for harder domains |
| `answering.style_exemplars` | How many verbatim passages get pinned into the prompt |
| `answering.max_tool_iterations` | Safety valve on the tool loop |
| `answering.rules` | Free-text rules appended to the prompt — the fastest tuning lever |
| `discord.history_turns` | Per-channel conversation memory depth |
| `discord.allowed_channel_ids` | Restrict where it responds |

Inspect the result of any change with:

```bash
creatorbot prompt | less
```

---

## A checklist before you ship someone else's voice

- The bot identifies itself as unofficial when asked.
- It breaks character on a sincere "are you really them?".
- It never claims to have personally played/pulled/streamed anything.
- It doesn't speak for the creator outside their subject area.
- Your sources permit what you're doing with them — see
  [DATA_SOURCES.md](DATA_SOURCES.md).
- If the creator asks you to take it down, take it down.
