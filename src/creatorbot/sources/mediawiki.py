"""MediaWiki lookups — the structured facts layer.

Used against the Dragon Ball Z Dokkan Battle Fandom wiki, whose api.php is
open, documented and CC BY-SA licensed. Queried live at question time so the
answer reflects the current article rather than a stale snapshot.

Nothing is written to the corpus by default; this is a read-through source.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx

from .base import Source, ToolResult

log = logging.getLogger(__name__)

_DROP_ELEMENTS = re.compile(r"<(script|style|sup)[^>]*>.*?</\1>", re.S | re.I)
_CELL_END = re.compile(r"</(td|th)>", re.I)
_BLOCK_END = re.compile(r"</(tr|p|div|h[1-6]|li|ul|ol)>", re.I)
_ANY_TAG = re.compile(r"<[^>]+>")

# Fandom appends an auto-generated Q&A block that is padding, not card data.
_CUTOFF = re.compile(r"\bQuick Answers\b", re.I)


class MediaWikiSource(Source):
    ingests = False

    @property
    def name(self) -> str:
        return "wiki"

    @property
    def api_url(self) -> str:
        url = self.cfg.get("api_url")
        if not url:
            raise ValueError("mediawiki source needs an `api_url`")
        return url

    @property
    def site_name(self) -> str:
        return self.cfg.get("site_name", "wiki")

    @property
    def max_chars(self) -> int:
        return int(self.cfg.get("max_chars_per_page", 6000))

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=25.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    f"creatorbot/0.1 (+https://github.com/; persona={self.persona.id}) "
                    "python-httpx"
                ),
                "Accept": "application/json",
            },
        )

    # -- API calls ------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        with self._client() as client:
            r = client.get(self.api_url, params=params)
            r.raise_for_status()
            data = r.json()
        return (data.get("query", {}) or {}).get("search", []) or []

    def page_text(self, title: str) -> str:
        """Fetch a page as rendered HTML and flatten it to readable text.

        `action=parse` rather than raw wikitext, because Dokkan card pages are
        almost entirely template calls — the leader skill, super attack, passive
        and link skills all live *inside* `{{...}}` and vanish if you strip
        templates. Letting the wiki render them first is the only way to get the
        actual numbers. (Fandom does not enable TextExtracts, so `prop=extracts`
        is not an option here.)
        """
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
        }
        with self._client() as client:
            r = client.get(self.api_url, params=params)
            r.raise_for_status()
            data = r.json()

        if "error" in data:
            log.debug("parse error for %r: %s", title, data["error"].get("code"))
            return ""
        rendered = (data.get("parse", {}) or {}).get("text", "")
        if not rendered:
            return ""
        return self._flatten(rendered)[: self.max_chars]

    @staticmethod
    def _flatten(rendered_html: str) -> str:
        text = _DROP_ELEMENTS.sub(" ", rendered_html)
        # Card data is laid out in tables; keep the cell boundaries legible so
        # "ATK +150%" doesn't run into the next stat.
        text = _CELL_END.sub(" | ", text)
        text = _BLOCK_END.sub("\n", text)
        text = _ANY_TAG.sub(" ", text)
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"(?:\s*\|\s*){2,}", " | ", text)   # collapse empty cells
        text = re.sub(r"\n\s*\n+", "\n", text).strip()

        cut = _CUTOFF.search(text)
        if cut:
            text = text[: cut.start()].strip()
        return text

    def page_url(self, title: str) -> str:
        base = self.api_url.rsplit("/api.php", 1)[0]
        return f"{base}/wiki/{quote(title.replace(' ', '_'))}"

    # -- question-time tool ---------------------------------------------------

    def tool_spec(self) -> dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": self.cfg.description
            or f"Search and read articles on the {self.site_name}.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search terms, or an exact article title. Card articles are "
                            "titled with the card's flavour name plus the character, e.g. "
                            "'Fused Fury Super Saiyan Gogeta'."
                        ),
                    },
                    "read_top_result": {
                        "type": "boolean",
                        "description": (
                            "Default true: also fetch the full text of the best match. "
                            "Set false to just list candidate titles."
                        ),
                    },
                },
                "required": ["query"],
            },
        }

    def run_tool(self, tool_input: dict[str, Any]) -> ToolResult:
        query = (tool_input.get("query") or "").strip()
        if not query:
            return ToolResult("No query supplied.", is_error=True)
        read_top = tool_input.get("read_top_result", True)

        try:
            results = self.search(query, limit=5)
        except httpx.HTTPError as exc:
            return ToolResult(f"{self.site_name} is unreachable right now ({exc}).", is_error=True)

        if not results:
            return ToolResult(
                f"No {self.site_name} article matches {query!r}. Try the character's name "
                "on its own, or use web search for very recent releases."
            )

        lines = [f"Candidate articles on {self.site_name}:"]
        citations = []
        for r in results:
            lines.append(f"  - {r['title']}")
            citations.append(f"[{r['title']}]({self.page_url(r['title'])})")

        if read_top:
            top = results[0]["title"]
            try:
                body = self.page_text(top)
            except httpx.HTTPError as exc:
                body = ""
                log.warning("wiki page fetch failed: %s", exc)
            if body:
                lines.append(
                    f"\n--- Full article: {top}\n{body}\n"
                    f"source: {self.page_url(top)} ({self.cfg.get('license', 'see wiki')})"
                )
            else:
                lines.append(f"\n(Could not read the body of '{top}'.)")

        return ToolResult("\n".join(lines), citations=citations[:3])

    def health(self) -> str:
        try:
            with self._client() as client:
                r = client.get(
                    self.api_url,
                    params={"action": "query", "meta": "siteinfo", "format": "json"},
                )
            return "reachable" if r.status_code == 200 else f"HTTP {r.status_code}"
        except httpx.HTTPError as exc:
            return f"unreachable ({exc})"
