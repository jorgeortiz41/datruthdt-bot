"""The answering engine: a Claude tool-use loop over the persona's sources.

Client-side tools (transcript search, wiki, card-site links) are executed here.
Web search runs server-side as an Anthropic-hosted tool in the same request, so
one loop covers both.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .config import PersonaConfig
from .embeddings import get_embedder
from .persona import build_system_prompt
from .sources import build_sources, web_search_config
from .store import CorpusStore

log = logging.getLogger(__name__)

# Server-tool variant with dynamic filtering (Opus 4.6+ / Sonnet 4.6+).
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class Engine:
    """Owns the Anthropic client, the corpus and the source tools."""

    def __init__(self, persona: PersonaConfig, *, client: Any | None = None):
        self.persona = persona
        self.store = CorpusStore(persona.db_path)
        self.client = client or anthropic.AsyncAnthropic()

        self.embedder = get_embedder()
        self.sources = build_sources(persona, self.store)
        # Sources that search the corpus need the embedder for dense retrieval.
        for src in self.sources:
            src._embedder = self.embedder  # noqa: SLF001 — deliberate injection

        self._tools_by_name = {}
        self.tools: list[dict[str, Any]] = []
        for src in self.sources:
            spec = src.tool_spec()
            if spec:
                self.tools.append(spec)
                self._tools_by_name[spec["name"]] = src

        ws = web_search_config(persona)
        if ws:
            tool: dict[str, Any] = {
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
                "max_uses": int(ws.get("max_uses", 4)),
            }
            if ws.get("allowed_domains"):
                tool["allowed_domains"] = list(ws.get("allowed_domains"))
            self.tools.append(tool)

        self.system_prompt = build_system_prompt(
            persona, [t.get("name", "") for t in self.tools]
        )

    def close(self) -> None:
        self.store.close()

    # -- tool execution -------------------------------------------------------

    async def _run_tool(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool, list[str]]:
        source = self._tools_by_name.get(name)
        if source is None:
            return f"Unknown tool {name!r}.", True, []
        try:
            # Sources are synchronous (httpx/sqlite); keep the event loop free.
            result = await asyncio.to_thread(source.run_tool, tool_input)
        except Exception as exc:
            log.exception("tool %s failed", name)
            return f"Tool {name} failed: {exc}", True, []
        return result.text, result.is_error, result.citations

    # -- main entry point -----------------------------------------------------

    async def answer(
        self, question: str, history: list[dict[str, Any]] | None = None
    ) -> Answer:
        """Answer one question, running the tool loop to completion."""
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": question})

        citations: list[str] = []
        tools_used: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}

        # Static prefix -> cacheable. Everything volatile lives in `messages`.
        system = [
            {
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

        for iteration in range(self.persona.max_tool_iterations):
            try:
                response = await self.client.messages.create(
                    model=self.persona.model,
                    max_tokens=self.persona.max_tokens,
                    system=system,
                    messages=messages,
                    tools=self.tools,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.persona.effort},
                )
            except anthropic.BadRequestError as exc:
                log.error("bad request: %s", exc)
                return Answer(
                    text=f"Something's wrong with how I asked the model: {exc.message}",
                    stop_reason="error",
                )
            except anthropic.RateLimitError:
                return Answer(
                    text="I'm getting rate limited right now — give me a minute and ask again.",
                    stop_reason="rate_limit",
                )
            except anthropic.APIStatusError as exc:
                log.error("api error %s: %s", exc.status_code, exc)
                return Answer(
                    text="The API is having a moment. Try again in a bit.",
                    stop_reason="error",
                )
            except anthropic.APIConnectionError:
                return Answer(
                    text="I can't reach the API right now — check the network.",
                    stop_reason="error",
                )

            for k in usage:
                usage[k] += getattr(response.usage, k, 0) or 0

            if response.stop_reason == "refusal":
                detail = getattr(response, "stop_details", None)
                log.warning("refusal: %s", getattr(detail, "category", None))
                return Answer(
                    text="I'm not going to answer that one. Ask me about Dokkan.",
                    stop_reason="refusal",
                )

            # Always echo the full content back — it carries thinking blocks and
            # server-tool results the API needs on the next turn.
            messages.append({"role": "assistant", "content": response.content})

            # Server-side tool paused the turn; resume with no new user input.
            if response.stop_reason == "pause_turn":
                continue

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                citations.extend(_server_citations(response.content))
                if response.stop_reason == "max_tokens" and text:
                    text += " …(cut off)"
                return Answer(
                    text=text or "…I got nothing. Ask me again?",
                    citations=_dedupe(citations),
                    tools_used=tools_used,
                    stop_reason=response.stop_reason,
                    usage=usage,
                )

            # Execute every requested client tool concurrently.
            calls = [b for b in response.content if b.type == "tool_use"]
            citations.extend(_server_citations(response.content))
            results = await asyncio.gather(
                *(self._run_tool(c.name, dict(c.input or {})) for c in calls)
            )

            blocks = []
            for call, (text, is_error, cites) in zip(calls, results):
                tools_used.append(call.name)
                citations.extend(cites)
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": text or "(no results)",
                        "is_error": is_error,
                    }
                )
            # All tool_results must go back in ONE user message.
            messages.append({"role": "user", "content": blocks})

        log.warning("hit max_tool_iterations (%d)", self.persona.max_tool_iterations)
        return Answer(
            text="I went round in circles looking that up. Try asking it more narrowly?",
            citations=_dedupe(citations),
            tools_used=tools_used,
            stop_reason="max_iterations",
            usage=usage,
        )


def _server_citations(content: list[Any]) -> list[str]:
    """Pull source links out of server-side web_search result blocks."""
    out = []
    for block in content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        results = getattr(block, "content", None)
        if not isinstance(results, list):  # an error object, not a result list
            continue
        for r in results:
            url = getattr(r, "url", None)
            title = getattr(r, "title", None) or url
            if url:
                out.append(f"[{title}]({url})")
    return out


def _dedupe(items: list[str], limit: int = 5) -> list[str]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:limit]
