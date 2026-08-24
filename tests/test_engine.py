"""Engine tool-loop tests against a fake xAI Responses API client.

These verify the loop mechanics that are easy to get subtly wrong:
function_call/function_call_output round-tripping and the iteration cap.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from creatorbot.config import load_persona
from creatorbot.engine import Engine


def function_call(name, call_id="tc_1", args=None):
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(args or {}),
    )


def message(text, annotations=None):
    block = SimpleNamespace(type="output_text", text=text, annotations=annotations or [])
    return SimpleNamespace(type="message", role="assistant", content=[block])


def response(output_items, output_text, status):
    return SimpleNamespace(
        output=output_items,
        output_text=output_text,
        status=status,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class FakeResponses:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def create(self, **kwargs):
        # The engine mutates its `input` list in place across iterations, so
        # snapshot it — otherwise every recorded call aliases the final state.
        recorded = dict(kwargs)
        recorded["input"] = [dict(m) for m in kwargs.get("input", [])]
        self.calls.append(recorded)
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.responses = FakeResponses(script)


@pytest.fixture
def persona():
    return load_persona("datruthdt")


def build(persona, script):
    return Engine(persona, client=FakeClient(script))


def test_plain_answer_no_tools(persona):
    eng = build(
        persona,
        [response([message("Bro that unit is busted.")], "Bro that unit is busted.", "completed")],
    )
    try:
        ans = asyncio.run(eng.answer("is he good?"))
        assert ans.text == "Bro that unit is busted."
        assert ans.stop_reason == "completed"
        assert ans.usage["input_tokens"] == 10
    finally:
        eng.close()


def test_tool_call_then_answer(persona):
    script = [
        response(
            [function_call("search_videos", args={"query": "Gogeta"})], "", "completed"
        ),
        response([message("Here's the link.")], "Here's the link.", "completed"),
    ]
    eng = build(persona, script)
    try:
        ans = asyncio.run(eng.answer("link me gogeta"))
        assert ans.text == "Here's the link."
        assert "search_videos" in ans.tools_used

        # Second request must carry: system, user, function_call, function_call_output.
        second = eng.client.responses.calls[1]["input"]
        assert second[2]["type"] == "function_call"
        assert second[2]["name"] == "search_videos"
        assert second[3]["type"] == "function_call_output"
        assert second[3]["call_id"] == "tc_1"
    finally:
        eng.close()


def test_parallel_tool_calls_each_get_a_result(persona):
    script = [
        response(
            [
                function_call("search_videos", call_id="a", args={"query": "x"}),
                function_call("search_videos", call_id="b", args={"query": "y"}),
            ],
            "",
            "completed",
        ),
        response([message("done")], "done", "completed"),
    ]
    eng = build(persona, script)
    try:
        asyncio.run(eng.answer("two things"))
        msgs = eng.client.responses.calls[1]["input"]
        calls = [m for m in msgs if m.get("type") == "function_call"]
        outputs = [m for m in msgs if m.get("type") == "function_call_output"]
        assert len(calls) == 2
        assert {m["call_id"] for m in outputs} == {"a", "b"}
    finally:
        eng.close()


def test_unknown_tool_returns_error_result_not_crash(persona):
    script = [
        response([function_call("nonexistent_tool", args={})], "", "completed"),
        response([message("recovered")], "recovered", "completed"),
    ]
    eng = build(persona, script)
    try:
        ans = asyncio.run(eng.answer("q"))
        assert ans.text == "recovered"
        msgs = eng.client.responses.calls[1]["input"]
        tool_output = next(m for m in msgs if m.get("type") == "function_call_output")
        assert "Unknown tool" in tool_output["output"]
    finally:
        eng.close()


def test_iteration_cap_terminates(persona):
    # Always asks for a tool — must stop, not spin.
    script = [
        response(
            [function_call("search_videos", call_id=f"t{i}", args={"query": "x"})],
            "",
            "completed",
        )
        for i in range(20)
    ]
    eng = build(persona, script)
    try:
        ans = asyncio.run(eng.answer("loop forever"))
        assert ans.stop_reason == "max_iterations"
        assert len(eng.client.responses.calls) == persona.max_tool_iterations
    finally:
        eng.close()


def test_system_prompt_is_stable_across_calls(persona):
    script = [
        response([message("a")], "a", "completed"),
        response([message("b")], "b", "completed"),
    ]
    eng = build(persona, script)
    try:
        asyncio.run(eng.answer("one"))
        asyncio.run(eng.answer("two"))
        c0, c1 = eng.client.responses.calls
        assert c0["input"][0]["role"] == "system"
        assert c0["input"][0]["content"] == eng.system_prompt
        # Byte-identical prefix across calls.
        assert c0["input"][0]["content"] == c1["input"][0]["content"]
    finally:
        eng.close()


def test_request_shape_matches_model_requirements(persona):
    eng = build(persona, [response([message("x")], "x", "completed")])
    try:
        asyncio.run(eng.answer("q"))
        call = eng.client.responses.calls[0]
        assert call["model"] == "grok-4.6"
        assert call["tool_choice"] == "auto"
        assert call["max_output_tokens"] == persona.max_tokens
        names = [t.get("name") or t.get("type") for t in call["tools"]]
        assert "search_wiki" in names and "web_search" in names
    finally:
        eng.close()


def test_url_citation_annotations_become_citations(persona):
    ann = SimpleNamespace(type="url_citation", url="https://example.com/x", title="X")
    eng = build(
        persona,
        [response([message("answer", annotations=[ann])], "answer", "completed")],
    )
    try:
        ans = asyncio.run(eng.answer("q"))
        assert ans.citations == ["[X](https://example.com/x)"]
    finally:
        eng.close()
