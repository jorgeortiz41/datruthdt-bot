# creatorbot — DaTruthDT edition

A Discord bot that answers **Dragon Ball Z: Dokkan Battle** questions in the
voice of [DaTruthDT](https://www.youtube.com/@DaTruthDT), grounded in what he
actually said on video plus live game data.

> **This is an unofficial fan project.** It is not DaTruthDT, not affiliated with
> him, and not endorsed by him. Its opinions are generated, not his. It says so
> itself when asked, and it will break character if someone sincerely asks
> whether they're talking to the real person. Go watch
> [the real channel](https://www.youtube.com/@DaTruthDT).

Everything creator-specific and subject-specific lives in **one YAML file**
(`personas/datruthdt.yaml`), so the same codebase becomes a bot for a different
YouTuber and a different database without touching Python. See
[docs/NEW_PERSONA.md](docs/NEW_PERSONA.md).

---

## How it works

```
                 ┌──────────────────────────────────────────┐
   Discord ─────▶│  engine.py — Claude tool-use loop        │
   question      │  (claude-opus-5, adaptive thinking)      │
                 └───────────────┬──────────────────────────┘
                                 │ picks tools per question
        ┌────────────────────────┼────────────────────────┬─────────────────┐
        ▼                        ▼                        ▼                 ▼
  search_videos            search_wiki            dokkaninfo_lookup    web_search
  BM25 + optional          Fandom MediaWiki       deep links to        Anthropic
  semantic search          API, live, rendered    dokkaninfo.com       server-side
  over transcripts         card tables                                 (live news)
        │                        │                        │                 │
        └──── his opinions ──────┴──── exact mechanics ────┴──── current meta ┘
                                 │
                                 ▼
                    persona.py builds the voice:
                    seed traits + a style guide *learned*
                    from real transcripts + verbatim
                    exemplars, all in a cached prefix
```

**The split that keeps it honest:** mechanics come from the wiki, opinions come
from his transcripts. The bot is told never to invent a multiplier and never to
attribute an opinion to him that isn't in a retrieved passage.

**The voice is learned, not hand-written.** `creatorbot style-profile` samples
your ingested transcripts and has Claude write a concrete style guide — real
filler words, real recurring phrases, real cadence. That's why this generalises
to any creator: you never have to guess at someone's catchphrases.

---

## Quick start

```bash
git clone <your-repo-url> datruthdt-bot
cd datruthdt-bot

conda env create -f environment.yml
conda activate datruthdt-bot        # <- do this in every new shell

cp .env.example .env                # then fill in the two required keys
```

Or without the environment file:

```bash
conda create -n datruthdt-bot python=3.12 -y
conda activate datruthdt-bot
pip install -e .
```

> **`zsh: command not found: creatorbot`** means the environment isn't active in
> this shell. Run `conda activate datruthdt-bot`. Being in the project directory
> is not enough — every `creatorbot ...` command below assumes the env is active.

<details>
<summary>Prefer venv?</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

On macOS, check your interpreter first — a broken `pyexpat` in some Homebrew
Python builds makes `python3 -m venv` produce a half-built environment with no
`activate` script and no pip. See
[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#pip-install-fails-on-macos).
</details>

You need two things in `.env`:

| Key | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> |
| `DISCORD_BOT_TOKEN` | See [Creating the Discord bot](#creating-the-discord-bot) below |

Then build the brain and start it:

```bash
creatorbot doctor          # check keys and source reachability
creatorbot ingest          # pull his videos into a local corpus
creatorbot style-profile   # learn his voice from those transcripts
creatorbot ask "is the new LR Gogeta worth summoning for?"
creatorbot run             # start the Discord bot
```

`creatorbot ingest` needs YouTube access that isn't IP-blocked — see
[Troubleshooting](docs/TROUBLESHOOTING.md) if you get `IpBlocked`. Everything
else works without it; the bot just won't have his voice or takes yet.

---

## Creating the Discord bot

### 1. Make the application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. **New Application** → name it (e.g. `DaTruthDT Bot`) → **Create**.
3. **Bot** in the left sidebar → **Reset Token** → **Copy**.
   Paste it into `.env` as `DISCORD_BOT_TOKEN=...`.
   You only see this once; reset it again if you lose it. Never commit it.

### 2. Turn on the message intent

Still on the **Bot** page, scroll to **Privileged Gateway Intents** and enable:

- ✅ **Message Content Intent**

Without this the bot can't read the text of messages that @mention it, and it
will only work via `/ask`. Click **Save Changes**.

### 3. Invite it to your server

**Just build the URL yourself.** A bot invite is a callback-less OAuth2 flow — it
needs no redirect URI, no client secret and no web server. Take your
**Application ID** (Developer Portal → **General Information** → *Application
ID*, also shown as *Client ID* on the OAuth2 page) and paste it in here:

```
https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=2147568640&scope=bot%20applications.commands
```

Open it, pick your server, **Authorize**. You need **Manage Server** permission
on the target server.

`2147568640` is the sum of the permissions the bot actually uses:

| Permission | Bit | Value |
|---|---|---|
| View Channel | `1 << 10` | 1024 |
| Send Messages | `1 << 11` | 2048 |
| Embed Links | `1 << 14` | 16384 |
| Read Message History | `1 << 16` | 65536 |
| Use Application Commands | `1 << 31` | 2147483648 |

> **"Please enter a redirect URI"** — that's the portal's **URL Generator**
> refusing to build a link until the app has a redirect registered, even though
> a bot invite doesn't use one. Ignore the generator and use the URL above. If
> you'd rather use the generator anyway, add any placeholder under **OAuth2 →
> Redirects** (e.g. `http://localhost`), **Save Changes**, and the dropdown
> unblocks. The bot never receives a callback either way.

<details>
<summary>Alternative: the Installation tab</summary>

Newer apps can skip OAuth2 entirely — **Installation** → set *Install Link* to
**Discord Provided Link**, then under **Default Install Settings → Guild
Install** add the `bot` and `applications.commands` scopes plus the permissions
above. Discord generates and hosts the invite link for you.
</details>

### 4. Run it

```bash
creatorbot run
```

You should see `logged in as YourBot#1234` and the bot appear online.

**Slash commands taking an hour to appear?** That's Discord's global
registration delay. For instant updates while developing, put your server's ID
in `.env`:

```bash
DISCORD_DEV_GUILD_ID=123456789012345678
```

(Enable Developer Mode in Discord → right-click your server → **Copy Server ID**.)

### Using it

| How | What happens |
|---|---|
| `@YourBot is the new Gogeta good?` | Replies in-thread |
| Reply to one of its messages | Continues the conversation |
| `/ask question:...` | Works in any channel |
| `/about` | What the bot is, and corpus stats |
| `/forget` | Clears this channel's memory |
| DM the bot | Works too |

Restrict it to specific channels by listing IDs under `discord.allowed_channel_ids`
in the persona file.

---

## Commands

| Command | Does |
|---|---|
| `creatorbot doctor` | Check keys, config and source reachability. Start here. |
| `creatorbot ingest [--limit N] [--refresh]` | Pull the creator's transcripts into `data/<persona>/corpus.sqlite3` |
| `creatorbot style-profile [--sample N]` | Derive the style guide + verbatim exemplars from the corpus |
| `creatorbot embed` | Add semantic vectors (needs `VOYAGE_API_KEY`) |
| `creatorbot ask "..."` | One question, in the terminal |
| `creatorbot chat` | Interactive REPL |
| `creatorbot stats` | Corpus size and composition |
| `creatorbot prompt` | Print the assembled system prompt (useful for tuning) |
| `creatorbot personas` | List available personas |
| `creatorbot run` | Start the Discord bot |

All commands take `-p/--persona <name>` to target a different persona.

---

## Retrieval

Out of the box, search is **SQLite FTS5 / BM25** — no extra dependency, no model
download, no vector database. It's genuinely good at what this bot does most:
matching proper nouns like "Ultra Instinct Goku".

Add semantic search by setting `VOYAGE_API_KEY` and installing the extra:

```bash
pip install -e '.[dense]'
creatorbot embed
```

Results are then fused with Reciprocal Rank Fusion. This mostly helps with
paraphrased questions that share no keywords with the transcript.

---

## Data sources, licensing and etiquette

This bot reads from other people's work. That is worth being careful about, and
the design reflects it. Full detail in **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)**.

| Source | Status | How it's used |
|---|---|---|
| DaTruthDT's transcripts | Public captions | Stored locally, quoted with attribution + timestamped link back to the video |
| Dokkan Fandom wiki | CC BY-SA 3.0, open API | Queried live, cited with a link |
| dokkaninfo.com | **Reference mode only** | Deep links only — never fetched |
| dokkandb.com | **Reference mode only** | Deep links only — never fetched |
| Web search | Anthropic server-side | Live news, cited |

### About the card databases

Both dokkaninfo.com and dokkandb.com explicitly disallow AI crawlers in
`robots.txt` (`ClaudeBot`, `GPTBot`, `CCBot`, `Google-Extended` and others get
`Disallow: /`). dokkaninfo additionally sets
`Content-Signal: search=yes, ai-train=no, use=reference` and firewalls
programmatic requests at the Cloudflare edge. dokkandb serves its data from a
private Supabase backend with no public API.

So this bot **links to them instead of scraping them**, which is exactly the
`use=reference` case they permit, and sends traffic their way. The actual card
mechanics come from the Fandom wiki, which is openly licensed for reuse.

If you get an operator's permission, flip `mode: fetch` in the persona file. That
path checks `robots.txt` against the User-Agent it actually sends, rate-limits,
caches, and identifies itself honestly. **It does not try to defeat Cloudflare,
rotate identities, or disguise itself** — if you're blocked, ask the operator
rather than hiding. There is no supported configuration that bypasses a block.

### If you're the creator or a site operator

Open an issue and this will be changed or taken down, no argument.

---

## Configuration

Everything lives in `personas/datruthdt.yaml`, which is heavily commented. The
sections you'll actually touch:

- `voice` — seed traits, vocabulary, hard limits, `profanity: none|light|uncensored`
- `domain.glossary` — jargon injected into the prompt so the model speaks the dialect
- `sources` — which knowledge sources are on, and their tool descriptions
- `answering` — model, effort, answer rules
- `discord` — channel restrictions, memory depth, message length

Model and effort can be overridden per-run with `CREATORBOT_MODEL` /
`CREATORBOT_EFFORT`.

---

## Making it a bot for someone else

That's the point of the layout. Copy the persona file, change the channel and
the domain, re-run `ingest` and `style-profile`:

```bash
cp personas/datruthdt.yaml personas/yourcreator.yaml
$EDITOR personas/yourcreator.yaml
CREATORBOT_PERSONA=yourcreator creatorbot ingest
CREATORBOT_PERSONA=yourcreator creatorbot style-profile
CREATORBOT_PERSONA=yourcreator creatorbot run
```

Adding a knowledge source for a *different* database means writing one class in
`src/creatorbot/sources/` implementing `tool_spec()` and `run_tool()`, then
registering it. Walkthrough: **[docs/NEW_PERSONA.md](docs/NEW_PERSONA.md)**.

---

## Project layout

```
personas/datruthdt.yaml       everything creator- and domain-specific
src/creatorbot/
  config.py                   persona loading
  store.py                    SQLite FTS5 corpus + hybrid search
  chunking.py                 transcript -> timestamped chunks
  embeddings.py               optional Voyage vectors
  persona.py                  system prompt assembly + style learning
  engine.py                   Claude tool-use loop
  bot.py                      Discord client
  cli.py                      command line
  sources/
    base.py                   the Source contract
    youtube.py                transcript ingestion + search tool
    mediawiki.py              live wiki lookups
    cardsite.py               dokkaninfo / dokkandb (reference or fetch)
tests/test_pipeline.py        offline end-to-end tests
data/<persona>/               corpus, style profile, caches (gitignored)
```

---

## Testing

```bash
pip install -e '.[dev]'
python -m pytest tests/ -q
```

The suite runs offline — no API keys, no network.

---

## Troubleshooting

See **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**. The two you're most
likely to hit:

- **`IpBlocked` during ingest** — YouTube blocks transcript requests from many
  IPs. Set `YOUTUBE_COOKIES_FROM_BROWSER=chrome` in `.env`.
- **Slash commands not appearing** — set `DISCORD_DEV_GUILD_ID`.

---

## License

MIT for the code — see [LICENSE](LICENSE). It does **not** cover third-party
content the bot retrieves at runtime; transcripts, wiki articles and game data
belong to their respective owners.
