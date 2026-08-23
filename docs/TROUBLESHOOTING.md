# Troubleshooting

Run this first — it checks keys, config and every source's reachability:

```bash
creatorbot doctor
```

---

## `IpBlocked` / "Sign in to confirm you're not a bot" during ingest

**Symptom**

```
no_captions: 3
ERROR: YouTube is blocking transcript requests from this IP ...
```

**Cause.** YouTube rate-limits and IP-blocks unauthenticated caption requests.
It hits datacenter and VPN IPs hardest, but residential connections get caught
too after a burst of requests. Nothing is wrong with your config.

**Fix — use your own logged-in session.** In `.env`:

```bash
YOUTUBE_COOKIES_FROM_BROWSER=chrome
```

Accepts `chrome`, `firefox`, `safari`, `edge`, `brave`, `chromium`, `opera`,
`vivaldi`. Close the browser first — Chrome locks its cookie DB while running.

On macOS you may be prompted for your keychain password, since Chrome encrypts
cookies with a keychain-stored key.

**Or export a cookies file** (browser extension "Get cookies.txt LOCALLY", or
`yt-dlp --cookies-from-browser chrome --cookies cookies.txt`):

```bash
YOUTUBE_COOKIES_FILE=/absolute/path/to/cookies.txt
```

**Or route through a proxy:**

```bash
YOUTUBE_PROXY=http://user:pass@host:port
# or, for Webshare residential proxies:
WEBSHARE_PROXY_USERNAME=...
WEBSHARE_PROXY_PASSWORD=...
```

**Then re-run.** Ingest is incremental — already-stored videos are skipped, so
re-running is cheap:

```bash
creatorbot ingest
```

**Also:** ingest gently. `--limit 50` a few times beats 300 in one burst.

### A note on using your own cookies

This authenticates as *you* against content you can already watch. It's what
yt-dlp documents for exactly this situation. It is not a way into private or
paywalled content, and it doesn't make you anonymous — if you hammer the API
while authenticated, you're doing it under your own account.

---

## Nothing was ingested but there's no error

Some videos genuinely have no captions (very old uploads, or captions disabled).
Check the ratio:

```bash
creatorbot stats
```

If `no_captions` is high but not everything, that's normal. If it's *everything*,
it's the IP block above.

---

## Slash commands don't appear in Discord

Global registration takes up to an hour. For instant sync during development,
put your server ID in `.env`:

```bash
DISCORD_DEV_GUILD_ID=123456789012345678
```

Get it via Discord → Settings → Advanced → **Developer Mode** on, then
right-click your server → **Copy Server ID**.

Also confirm you invited the bot with the `applications.commands` scope, not
just `bot`. If you didn't, re-run the OAuth2 URL with both ticked.

---

## The bot ignores @mentions but `/ask` works

**Message Content Intent** is off. Discord Developer Portal → your app → **Bot**
→ **Privileged Gateway Intents** → enable **Message Content Intent** → **Save
Changes** → restart the bot.

---

## `PrivilegedIntentsRequired` on startup

Same fix as above. `bot.py` requests `message_content`, and Discord refuses the
connection outright if the app doesn't have it enabled.

---

## The bot replies but sounds generic

Check that the voice layer actually built:

```bash
creatorbot stats     # 'profile: yes'?
creatorbot prompt | head -100
```

If `profile: NOT GENERATED`, run:

```bash
creatorbot style-profile
```

If the profile exists but is thin, you probably ingested too few videos. Ingest
more, then regenerate with a larger sample:

```bash
creatorbot ingest
creatorbot style-profile --sample 100
```

Read `data/<persona>/style_profile.md` directly. It's the single highest-leverage
artefact — if it's vague, the impression will be vague. You can hand-edit it;
it's just markdown and it's loaded verbatim.

---

## Answers are accurate but boring / hype but wrong

Tune `answering.rules` in the persona file — it's free text appended to the
prompt and the fastest lever you have. Also try raising `answering.effort` from
`medium` to `high` for harder questions.

---

## The bot invents card numbers

It's instructed not to, and to prefer the wiki tool. If it still does:

1. Confirm the wiki tool is reachable: `creatorbot doctor`.
2. Raise `answering.effort`.
3. Add a sharper rule, e.g. *"Never state a percentage or multiplier that did not
   appear verbatim in a `search_wiki` result."*

---

## Dead links to dokkaninfo / dokkandb

The wiki's "ID" field is a different numbering scheme from the 7-digit card IDs
those sites use in URLs. The tool description warns the model about this, but if
you see dead links, the fix is to stop passing `card_id` and let it use `query`
so the site's own search resolves the name.

---

## `403` from dokkaninfo in fetch mode

Expected — dokkaninfo firewalls programmatic requests at the Cloudflare edge and
disallows AI agents in `robots.txt`. Leave it in `mode: reference`. See
[DATA_SOURCES.md](DATA_SOURCES.md). There is no supported way around this, by
design; if you need the data, contact the site operator.

---

## `pip install` fails on macOS with `ValueError: invalid literal for int()`

Not this project — a broken Python install. `platform.mac_ver()` returns `''`
because `pyexpat` can't load, which breaks pip's wheel-tag computation:

```
ImportError: dlopen(.../pyexpat...so): Symbol not found:
  _XML_SetAllocTrackerActivationThreshold
```

Check it:

```bash
python3 -c "import platform, plistlib; print(platform.mac_ver())"
```

If it prints `('', ('', '', ''), '')`, reinstall the interpreter:

```bash
brew reinstall expat python@3.14
```

Or build the venv from a different working Python:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python3 -m venv .venv
```

---

## `chunks_fts` / FTS5 errors

Your SQLite lacks FTS5. Check:

```bash
python3 -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(b)'); print('ok')"
```

macOS system Python and Homebrew Python both ship it. If yours doesn't, use a
`python.org` build or conda.

---

## `400 invalid_request_error: The following domains are not accessible to our user agent`

**Symptom.** Every question fails instantly with a 400 naming a domain, e.g.
`['reddit.com']`. No answer at all, not a degraded one.

**Cause.** `sources[web_search].allowed_domains` in the persona file lists a site
that blocks Anthropic's crawler. The API rejects the **entire request** if even
one entry is unreachable — it does not silently drop the bad domain.

**Fix.** Remove it. Known-blocked and not worth re-adding: `reddit.com`,
`x.com`, `dokkaninfo.com`, `dokkandb.com`. To search the open web instead, delete
the `allowed_domains` key entirely.

---

## Rate limits from Anthropic

The SDK retries 429s automatically. If you're hitting them constantly, lower
`answering.effort`, or reduce `max_tool_iterations` so each question costs fewer
round trips.
