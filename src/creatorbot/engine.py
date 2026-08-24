"""The answering engine: a Grok tool-use loop over the persona's sources.

Uses the xAI API (OpenAI-compatible) with native web_search + custom function tools.
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

        # Build OpenAI-style tools list
        self.tools: list[dict[str, Any]] = []
        self._tools_by_name: dict[str, Any] = {}

        for src in self.sources:
            spec = src.tool_spec()
            if not spec:
                continue
            # Convert to OpenAI function format
            tool = {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get(
                        "input_schema", {"type": "object", "properties": {}}
                    ),
                },
            }
            self.tools.append(tool)
            self._tools_by_name[spec["name"]] = src

        # Native xAI web search (server-side)
        ws = web_search_config(persona)
        if ws:
            web_tool: dict[str, Any] = {"type": "web_search"}
            # xAI supports allowed_domains on the tool
            if ws.get("allowed_domains"):
                web_tool["allowed_domains"] = list(ws["allowed_domains"])[:5]
            self.tools.append(web_tool)

        self.system_prompt = build_system_prompt(
            persona,
            [t.get("function", {}).get("name") or t.get("type") for t in self.tools],
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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        citations: list[str] = []
        tools_used: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        for iteration in range(self.persona.max_tool_iterations):
            try:
                response = await self.client.chat.completions.create(
                    model=self.persona.model,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                    tool_choice="auto",
                    max_tokens=self.persona.max_tokens,
                    temperature=0.7,
                )
            except Exception as exc:
                log.error("Grok API error: %s", exc)
                return Answer(
                    text="The API is having a moment. Try again in a bit.",
                    stop_reason="error",
                )

            choice = response.choices[0]
            message = choice.message

            # Track usage
            if response.usage:
                usage["input_tokens"] += (
                    getattr(response.usage, "prompt_tokens", 0) or 0
                )
                usage["output_tokens"] += (
                    getattr(response.usage, "completion_tokens", 0) or 0
                )

            # Append assistant message
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            # No tool calls → final answer
            if not message.tool_calls:
                text = clean_reply(message.content or "")
                return Answer(
                    text=text or "…I got nothing. Ask me again?",
                    citations=_dedupe(citations),
                    tools_used=tools_used,
                    stop_reason=choice.finish_reason,
                    usage=usage,
                )

            # Execute tool calls
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                tools_used.append(name)
                text, _is_error, cites = await self._run_tool(name, args)
                citations.extend(cites)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": text or "(no results)",
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


def _dedupe(items: list[str], limit: int = 8) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:limit]
