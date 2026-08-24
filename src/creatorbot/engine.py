"""The answering engine: a Grok tool-use loop over the persona's sources.

Uses xAI's Responses API (OpenAI-compatible, /v1/responses) with native
web_search + custom function tools. xAI's /v1/chat/completions no longer
supports a server-side web_search tool — Live Search on that endpoint is
deprecated in favor of the Responses API's Agent Tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .config import PersonaConfig
from .embeddings import get_embedder
from .persona import build_system_prompt
from .sources import build_sources, web_search_config
from .store import CorpusStore

log = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class Engine:
    """Owns the Grok client, the corpus and the source tools."""

    def __init__(self, persona: PersonaConfig, *, client: Any | None = None):
        self.persona = persona
        self.store = CorpusStore(persona.db_path)

        api_key = os.getenv("XAI_API_KEY")
        if not api_key and client is None:
            raise RuntimeError("XAI_API_KEY is required")

        self.client = client or AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        self.embedder = get_embedder()
        self.sources = build_sources(persona, self.store)
        for src in self.sources:
            src._embedder = self.embedder

        # Build Responses-API-style tools list. Function tools are flat here
        # (name/description/parameters at the top level), unlike the nested
        # {"type": "function", "function": {...}} shape chat.completions uses.
        self.tools: list[dict[str, Any]] = []
        self._tools_by_name: dict[str, Any] = {}

        for src in self.sources:
            spec = src.tool_spec()
            if not spec:
                continue
            tool = {
                "type": "function",
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            }
            self.tools.append(tool)
            self._tools_by_name[spec["name"]] = src

        # Native xAI web search (server-side Agent Tool).
        ws = web_search_config(persona)
        if ws:
            web_tool: dict[str, Any] = {"type": "web_search"}
            if ws.get("allowed_domains"):
                web_tool["filters"] = {
                    "allowed_domains": list(ws.get("allowed_domains"))[:5]
                }
            self.tools.append(web_tool)

        self.system_prompt = build_system_prompt(
            persona, [t.get("name") or t.get("type") for t in self.tools]
        )

    def close(self) -> None:
        self.store.close()

    async def _run_tool(
        self, name: str, tool_input: dict[str, Any]
    ) -> tuple[str, bool, list[str]]:
        source = self._tools_by_name.get(name)
        if source is None:
            return f"Unknown tool {name!r}.", True, []
        try:
            result = await asyncio.to_thread(source.run_tool, tool_input)
        except Exception as exc:
            log.exception("tool %s failed", name)
            return f"Tool {name} failed: {exc}", True, []
        return result.text, result.is_error, result.citations

    async def answer(
        self, question: str, history: list[dict[str, Any]] | None = None
    ) -> Answer:
        """Answer one question, running the tool loop to completion."""
        input_items: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if history:
            input_items.extend(history)
        input_items.append({"role": "user", "content": question})

        citations: list[str] = []
        tools_used: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        for iteration in range(self.persona.max_tool_iterations):
            try:
                response = await self.client.responses.create(
                    model=self.persona.model,
                    input=input_items,
                    tools=self.tools if self.tools else None,
                    tool_choice="auto",
                    max_output_tokens=self.persona.max_tokens,
                    temperature=0.7,
                )
            except Exception as exc:
                log.error("Grok API error: %s", exc)
                return Answer(
                    text="The API is having a moment. Try again in a bit.",
                    stop_reason="error",
                )

            output = response.output or []
            citations.extend(_server_citations(output))

            if response.usage:
                usage["input_tokens"] += (
                    getattr(response.usage, "input_tokens", 0) or 0
                )
                usage["output_tokens"] += (
                    getattr(response.usage, "output_tokens", 0) or 0
                )

            calls = [item for item in output if getattr(item, "type", None) == "function_call"]

            # No function calls → final answer.
            if not calls:
                text = clean_reply(getattr(response, "output_text", "") or "")
                return Answer(
                    text=text or "…I got nothing. Ask me again?",
                    citations=_dedupe(citations),
                    tools_used=tools_used,
                    stop_reason=getattr(response, "status", None),
                    usage=usage,
                )

            # Echo each function call back, then execute it and append its result.
            for call in calls:
                call_id = call.call_id
                name = call.name
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": call.arguments,
                    }
                )
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                tools_used.append(name)
                text, _is_error, cites = await self._run_tool(name, args)
                citations.extend(cites)

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": text or "(no results)",
                    }
                )

        log.warning("hit max_tool_iterations (%d)", self.persona.max_tool_iterations)
        return Answer(
            text="I went round in circles looking that up. Try asking it more narrowly?",
            citations=_dedupe(citations),
            tools_used=tools_used,
            stop_reason="max_iterations",
            usage=usage,
        )


def _server_citations(output: list[Any]) -> list[str]:
    """Pull source links out of web_search's url_citation annotations."""
    out = []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) != "output_text":
                continue
            for ann in getattr(block, "annotations", None) or []:
                if getattr(ann, "type", None) != "url_citation":
                    continue
                url = getattr(ann, "url", None)
                title = getattr(ann, "title", None) or url
                if url:
                    out.append(f"[{title}]({url})")
    return out


#: Matches the `?t=`/`&t=` seek parameter on a citation deep link.
_SEEK_PARAM = re.compile(r"[?&]t=\d+")
#: Pulls the URL out of a `[label](url)` markdown citation.
_CITE_URL = re.compile(r"\]\(([^)]+)\)")


def clean_reply(text: str) -> str:
    """Strip markup that leaks into Discord."""
    text = re.sub(r"</?cite\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?(antml|thinking|search_quality[^>]*)\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _citation_key(citation: str) -> str:
    """Identity of the *source* a citation points at, ignoring position in it.

    Several passages from one video are separate hits with different `?t=`
    timestamps, so deduping on the formatted string keeps all of them and the
    footer shows the same video two or three times. Key on the URL with the
    seek parameter stripped instead, so one source appears once.
    """
    match = _CITE_URL.search(citation)
    if not match:
        return citation
    return _SEEK_PARAM.sub("", match.group(1)).rstrip("?&")


def _dedupe(items: list[str], limit: int = 1) -> list[str]:
    """First (highest-ranked) citation per distinct source."""
    seen, out = set(), []
    for item in items:
        key = _citation_key(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:limit]
