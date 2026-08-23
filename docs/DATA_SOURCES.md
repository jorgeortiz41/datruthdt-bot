# Data sources — what this bot reads, and on what terms

This is the file to read before pointing the bot at a new site.

---

## The rule this project follows

**Read what you're allowed to read, cite it, and link back.** When a site says
no, the answer is to ask them or to use a different source — not to find a way
around the block. There is deliberately no configuration in this codebase that
disguises the client, rotates identities, solves challenges, or ignores a 403.

---

## 1. DaTruthDT's video transcripts

**What:** public caption tracks from the channel's videos.
**How:** `yt-dlp` lists the channel; `youtube-transcript-api` (falling back to
yt-dlp subtitle download) pulls the captions. Only text — no video or audio is
downloaded.
**Stored:** yes, locally in `data/<persona>/corpus.sqlite3`, gitignored.

**Why this is the honest version of "memorize the videos":** every retrieved
passage keeps its video title and start timestamp, and the bot cites a
`?t=` deep link back to the moment in the video. The corpus is a search index
that points at his content, not a replacement for watching it.

**Constraint you will hit:** YouTube IP-blocks unauthenticated transcript
requests from a lot of networks. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## 2. Dragon Ball Z Dokkan Battle Fandom wiki

**What:** the structured card data — leader skills, passives, super attacks,
links, categories, EZA details, event requirements.
**How:** the public MediaWiki API at `dbz-dokkanbattle.fandom.com/api.php`,
queried **live** at question time.
**License:** CC BY-SA 3.0 — reuse is permitted with attribution, which the bot
provides as a link on every answer.
**Stored:** no. Read-through, so answers reflect the current article.

### Implementation note worth knowing

Card pages on this wiki are almost entirely `{{template}}` calls. The leader
skill, super attack, passive and links all live *inside* those templates, so
fetching raw wikitext and stripping templates returns an empty page.

The source therefore uses `action=parse` to have the wiki render the page, then
flattens the resulting HTML while preserving table cell boundaries. That's what
turns a card page into usable text like:

```
"Fusion" Category Ki +3 and HP, ATK & DEF +170%; or Type Ki +3 and HP, ATK & DEF +100%
| 100x Big Bang Kamehameha (Extreme) | Greatly raises ATK & DEF for 1 turn ...
```

`prop=extracts` is not available — Fandom doesn't enable the TextExtracts
extension.

---

## 3. dokkaninfo.com — reference mode

**Status: linked, never fetched.**

What the site says:

```
# robots.txt
User-agent: *
Content-Signal: search=yes, ai-train=no, use=reference
Allow: /

User-agent: ClaudeBot
Disallow: /
User-agent: GPTBot
Disallow: /
User-agent: CCBot
Disallow: /
User-agent: Google-Extended
Disallow: /
...
```

And what the server does to any programmatic request:

```
HTTP/2 403
server: cloudflare
<title>Attention Required! | Cloudflare</title>
"Sorry, you have been blocked — You are unable to access dokkaninfo.com"
```

That's a firewall block, not a JS challenge; no amount of header-setting gets
through it, and getting through it is not a goal.

`use=reference` explicitly permits AI systems to consume the content **as a
reference** — cite it, link it — while `ai-train=no` reserves rights against
ingestion. So the source builds deep links (`/cards/{id}`, `/categories`,
`/events`, `/tools/links`) and tells the model, in the tool result, that it has
*not* read the page and must not claim otherwise.

---

## 4. dokkandb.com — reference mode

**Status: linked, never fetched.**

`robots.txt` allows `User-agent: *` but names the AI crawlers individually:

```
User-agent: ClaudeBot     Disallow: /
User-agent: Claude-Web    Disallow: /
User-agent: anthropic-ai  Disallow: /
User-agent: GPTBot        Disallow: /
User-agent: CCBot         Disallow: /
...
User-agent: *
Allow: /
Disallow: /login /signup /box /share /bugreports
```

The site is reachable (HTTP 200), but it's an Angular SPA: every route returns
the same ~60 KB shell with no server-rendered content, so there is nothing to
read from the HTML. The data comes from a **private Supabase backend**
(`enaskhebnjtktdfszdcb.supabase.co`) reached with an anon key embedded in the JS
bundle, via endpoints like `/api/cards-by-ids`, `/api/recent-cards`,
`/api/summons`, `/api/news`.

**That is not a public API.** Extracting the key from their bundle to bulk-pull
their database would be using someone's private credential against a site that
has said no to AI crawlers — so this project doesn't do it, and won't add an
option to.

What it does instead: link to `/cards/{id}` and the browse sections. Their
`sitemap.xml` is published for crawlers and lists valid 7-digit card IDs, which
is a legitimate way to resolve links if you want to build an ID index.

**Note:** the "ID" field shown on the Fandom wiki is a *different, shorter*
numbering scheme than the 7-digit IDs these sites use. The tool description
tells the model not to confuse them, because doing so produces dead links.

---

## 5. Web search

Anthropic's server-side `web_search_20260209` tool, domain-scoped in the persona
file. This covers everything the wiki hasn't caught up on: new banner
announcements, celebration dates, datamines, current meta discussion. Results
arrive with source URLs, which the bot surfaces in its footer.

---

## Enabling fetch mode (only with permission)

If you obtain an operator's permission:

```yaml
- type: dokkaninfo
  mode: fetch
  fetch:
    respect_robots: true      # keep this true
    rate_limit_seconds: 2.0
    cache_ttl_hours: 24
```

and set `CREATORBOT_CONTACT_EMAIL` so the User-Agent carries a way to reach you.

What fetch mode does:

- checks `robots.txt` against the User-Agent it actually sends, and **refuses**
  if disallowed — `respect_robots: false` exists but is on you;
- sends `creatorbot/0.1 (personal Discord bot; +you@example.com)` — a truthful
  identity, not a browser impersonation;
- serialises requests with a delay and caches to disk;
- on a 403, returns an explicit "do not retry or disguise the request" result
  and falls back to linking.

---

## Adding a source for a different site

Check, in order:

1. **Is there a public, documented API?** Use it. Best case.
2. **What does `robots.txt` say** for the User-Agent you'll actually send —
   including `Content-Signal` if present?
3. **Is the content openly licensed** (CC BY-SA, public domain)? If so,
   attribute it properly.
4. **Is it server-rendered?** An SPA has nothing to scrape; you'd be reverse-
   engineering a private backend, which is a different and worse thing.
5. **If it's ambiguous, ask the operator.** Most hobbyist sites are run by one
   person who is reachable and often happy to help a fan project.

If the answer is "no", link to them instead. A cited link is worth more to a
small site than a scrape, and it's worth more to your users than a hallucination.
