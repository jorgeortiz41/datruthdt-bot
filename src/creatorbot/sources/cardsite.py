"""Community card databases (dokkaninfo.com, dokkandb.com, ...).

These sites are the best Dokkan data on the internet, and they are also
privately run, ad-supported and explicit about not wanting AI crawlers. Both
name ClaudeBot / GPTBot / CCBot and friends as `Disallow: /` in robots.txt, and
dokkaninfo additionally sets `Content-Signal: ai-train=no, use=reference` and
firewalls programmatic requests at the Cloudflare edge.

So this source ships in **reference mode**: it constructs deep links for the
user to click and never fetches a byte. That is exactly the `use=reference`
case the sites allow, and it sends them traffic instead of taking it.

**fetch mode** exists for when you have the operator's permission. It:
  * checks robots.txt for the User-Agent we actually send, and refuses if
    disallowed,
  * sends an honest, identifying User-Agent with a contact address,
  * rate-limits and caches on disk.

It deliberately does *not* impersonate a browser, rotate identities, solve
challenges, or otherwise work around a block. If you're blocked, the answer is
to ask the operator, not to hide. See docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from .base import Source, ToolResult

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_ANY_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Sensible defaults per known site; a persona can override any of these.
SITE_DEFAULTS: dict[str, dict[str, Any]] = {
    "dokkaninfo": {
        "base_url": "https://dokkaninfo.com",
        "site_name": "Dokkan Info",
        "card_url": "{base}/cards/{card_id}",
        "search_url": "{base}/cards?name={query}",
        "browse_urls": {
            "cards": "{base}/cards",
            "categories": "{base}/categories",
            "events": "{base}/events",
            "links": "{base}/tools/links",
        },
    },
    "dokkandb": {
        "base_url": "https://www.dokkandb.com",
        "site_name": "DokkanDB",
        "card_url": "{base}/cards/{card_id}",
        "search_url": "{base}/cards?search={query}",
        "browse_urls": {
            "cards": "{base}/cards",
            "news": "{base}/news",
            "summons": "{base}/summons",
            "awakening medals": "{base}/awakeningmedals",
        },
        "sitemap_url": "{base}/sitemap.xml",
    },
}


class CardSiteSource(Source):
    """A community database exposed as links (and optionally as fetched text)."""

    ingests = False

    # -- config ---------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.cfg.type

    @property
    def _defaults(self) -> dict[str, Any]:
        return SITE_DEFAULTS.get(self.cfg.type, {})

    def _opt(self, key: str, fallback: Any = None) -> Any:
        return self.cfg.get(key, self._defaults.get(key, fallback))

    @property
    def base_url(self) -> str:
        return str(self._opt("base_url", "")).rstrip("/")

    @property
    def site_name(self) -> str:
        return self._opt("site_name", self.cfg.type)

    @property
    def mode(self) -> str:
        return str(self.cfg.get("mode", "reference")).lower()

    @property
    def _fetch_opts(self) -> dict[str, Any]:
        return self.cfg.get("fetch", {}) or {}

    @property
    def user_agent(self) -> str:
        product = self._fetch_opts.get("user_agent_product", "creatorbot")
        contact = os.getenv("CREATORBOT_CONTACT_EMAIL", "").strip()
        suffix = f"; +{contact}" if contact else ""
        return f"{product}/0.1 (personal Discord bot{suffix})"

    # -- URL building (always available, never touches the network) -----------

    def card_url(self, card_id: str | int) -> str:
        tmpl = self._opt("card_url", "{base}/cards/{card_id}")
        return tmpl.format(base=self.base_url, card_id=card_id)

    def search_url(self, query: str) -> str:
        tmpl = self._opt("search_url", "{base}/cards?name={query}")
        return tmpl.format(base=self.base_url, query=quote_plus(query))

    def browse_url(self, what: str) -> str | None:
        urls = self._opt("browse_urls", {}) or {}
        tmpl = urls.get(what.lower())
        return tmpl.format(base=self.base_url) if tmpl else None

    # -- robots ---------------------------------------------------------------

    _robots: RobotFileParser | None = None

    def robots_allows(self, url: str) -> bool:
        """True if robots.txt permits *our* User-Agent to fetch `url`."""
        if not self._fetch_opts.get("respect_robots", True):
            return True
        if self._robots is None:
            rp = RobotFileParser()
            robots_url = urljoin(self.base_url + "/", "robots.txt")
            try:
                resp = httpx.get(
                    robots_url,
                    timeout=15.0,
                    headers={"User-Agent": self.user_agent},
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    # No readable robots.txt — treat as permissive, per RFC 9309.
                    rp.parse([])
            except httpx.HTTPError as exc:
                log.warning("could not read %s (%s); assuming disallowed", robots_url, exc)
                rp.parse(["User-agent: *", "Disallow: /"])
            self._robots = rp
        product = self.user_agent.split("/", 1)[0]
        return self._robots.can_fetch(product, url)

    # -- fetching (opt-in) ----------------------------------------------------

    @property
    def _cache_dir(self) -> Path:
        d = self.persona.data_dir / "cache" / self.cfg.type
        d.mkdir(parents=True, exist_ok=True)
        return d

    _last_request: float = 0.0

    def fetch_page(self, url: str) -> tuple[str, str | None]:
        """Return (text, error). Only ever called in fetch mode."""
        ttl_hours = float(self._fetch_opts.get("cache_ttl_hours", 24))
        key = re.sub(r"[^a-zA-Z0-9]+", "_", url)[:120]
        cache_file = self._cache_dir / f"{key}.json"

        if cache_file.exists():
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_h < ttl_hours:
                try:
                    return json.loads(cache_file.read_text())["text"], None
                except (json.JSONDecodeError, KeyError):
                    pass

        if not self.robots_allows(url):
            return "", (
                f"robots.txt at {self.base_url} disallows {self.user_agent!r} from "
                f"fetching {url}. Not fetching. Link the user to the page instead."
            )

        delay = float(self._fetch_opts.get("rate_limit_seconds", 2.0))
        elapsed = time.time() - self._last_request
        if elapsed < delay:
            time.sleep(delay - elapsed)

        try:
            resp = httpx.get(
                url,
                timeout=25.0,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent, "Accept": "text/html,application/json"},
            )
            self._last_request = time.time()
        except httpx.HTTPError as exc:
            return "", f"{self.site_name} request failed: {exc}"

        if resp.status_code == 403:
            return "", (
                f"{self.site_name} returned 403 — the site is blocking automated "
                "requests. Do not retry or disguise the request. Fall back to linking."
            )
        if resp.status_code != 200:
            return "", f"{self.site_name} returned HTTP {resp.status_code}."

        text = _html_to_text(resp.text)
        try:
            cache_file.write_text(json.dumps({"url": url, "text": text}))
        except OSError:
            pass
        return text, None

    # -- question-time tool ---------------------------------------------------

    def tool_spec(self) -> dict[str, Any]:
        browse = ", ".join((self._opt("browse_urls", {}) or {}).keys()) or "cards"
        if self.mode == "fetch":
            desc = self.cfg.description or f"Look up and read a page on {self.site_name}."
        else:
            desc = (
                (self.cfg.description or f"Build links to {self.site_name}.")
                + f" This returns URLs only — {self.site_name} is not fetched. "
                "Use it to give the user a clickable source for a card or event."
            )
        return {
            "name": self.tool_name,
            "description": desc,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Card, category or event name to link to.",
                    },
                    "card_id": {
                        "type": "string",
                        "description": (
                            "Only pass this if you have the site's own 7-digit card id "
                            "(e.g. '1029341'). Do NOT pass the 'ID' number shown on the "
                            "wiki — that is a different, shorter numbering scheme and "
                            "produces a dead link. When unsure, use `query` instead and "
                            "let the site's search resolve the name."
                        ),
                    },
                    "browse": {
                        "type": "string",
                        "description": f"Link a top-level section instead. One of: {browse}.",
                    },
                },
                "required": [],
            },
        }

    def run_tool(self, tool_input: dict[str, Any]) -> ToolResult:
        query = (tool_input.get("query") or "").strip()
        card_id = (tool_input.get("card_id") or "").strip()
        browse = (tool_input.get("browse") or "").strip()

        if browse:
            url = self.browse_url(browse)
            if not url:
                return ToolResult(f"{self.site_name} has no section named {browse!r}.")
        elif card_id:
            if not card_id.isdigit():
                return ToolResult("card_id must be numeric.", is_error=True)
            url = self.card_url(card_id)
        elif query:
            url = self.search_url(query)
        else:
            return ToolResult("Give a query, a card_id or a browse section.", is_error=True)

        label = query or browse or f"card {card_id}"
        citation = f"[{self.site_name}: {label}]({url})"

        if self.mode != "fetch":
            return ToolResult(
                f"{self.site_name} link for {label}: {url}\n"
                f"(Reference mode — the page was not fetched, so treat the link as a "
                f"pointer for the user, not as a fact you have read. Get the actual "
                f"mechanics from the wiki tool.)",
                citations=[citation],
            )

        text, error = self.fetch_page(url)
        if error:
            return ToolResult(
                f"{error}\nStill give the user the link: {url}", citations=[citation]
            )
        if not text.strip():
            return ToolResult(
                f"{self.site_name} returned a page with no readable text — it is likely a "
                f"JavaScript app that renders client-side. Link instead: {url}",
                citations=[citation],
            )
        max_chars = int(self._fetch_opts.get("max_chars", 6000))
        return ToolResult(
            f"--- {self.site_name}: {label}\n{text[:max_chars]}\nsource: {url}",
            citations=[citation],
        )

    # -- diagnostics ----------------------------------------------------------

    def health(self) -> str:
        if self.mode != "fetch":
            return f"reference mode (links only) -> {self.base_url}"
        probe = self.base_url + "/"
        allowed = self.robots_allows(probe)
        if not allowed:
            return f"fetch mode BUT robots.txt disallows {self.user_agent!r} — will refuse"
        try:
            r = httpx.get(
                probe, timeout=15.0, headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
            return f"fetch mode, robots ok, HTTP {r.status_code}"
        except httpx.HTTPError as exc:
            return f"fetch mode, robots ok, but unreachable ({exc})"


def _html_to_text(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = _ANY_TAG.sub(" ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return _WS.sub(" ", text).strip()
