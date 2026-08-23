"""Engine tool-loop tests against a fake Anthropic client.

These verify the loop mechanics that are easy to get subtly wrong:
tool_result batching, pause_turn resumption, full-content echo, refusal
handling and the iteration cap.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from creatorbot.config import load_persona
from creatorbot.engine import Engine


def block(**kw):
    return SimpleNamespace(**kw)


def text_block(t):
    return block(type="text", text=t)


def tool_use(name, tid="tu_1", inp=None):
    return block(type="tool_use", name=name, id=tid, input=inp or {})


def response(content, stop_reason):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0
        ),
    )


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def create(self, **kwargs):
        # The engine mutates its `messages` list in place across iterations, so
        # snapshot it — otherwise every recorded call aliases the final state.
        recorded = dict(kwargs)
        recorded["messages"] = [dict(m) for m in kwargs.get("messages", [])]
        self.calls.append(recorded)
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


@pytest.fixture
def persona():
    return load_persona("datruthdt")


def build(persona, script):
    return Engine(persona, client=FakeClient(script))


def test_plain_answer_no_tools(persona):
    eng = build(persona, [response([text_block("Bro that unit is busted.")], "end_turn")])
    try:
        ans = asyncio.run(eng.answer("is he good?"))
        assert ans.text == "Bro that unit is busted."
        assert ans.stop_reason == "end_turn"
        assert ans.usage["input_tokens"] == 10
    finally:
        eng.close()


def test_tool_call_then_answer(persona):
    script = [
        response([tool_use("search_videos", inp={"query": "Gogeta"})], "tool_use"),
        response([text_block("Here's the link.")], "end_turn"),
    ]
    eng = build(persona, script)
    try:
        ans = asyncio.run(eng.answer("link me gogeta"))
        assert ans.text == "Here's the link."
        assert "search_videos" in ans.tools_used

        # Second request must carry: user, assistant(content), user(tool_result)
        second = eng.client.messages.calls[1]["messages"]
        assert second[1]["role"] == "assistant"
        assert second[2]["role"] == "user"
        results = second[2]["content"]
        assert len(results) == 1
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tu_1"
    finally:
        eng.close()


def test_parallel_tool_results_go_in_one_message(persona):
    script = [
        response(
            [
                tool_use("search_videos", tid="a", inp={"query": "x"}),
                tool_use("search_videos", tid="b", inp={"query": "y"}),
            ],
            "tool_use",
        ),
        response([text_block("done")], "end_turn"),
    ]
    eng = build(persona, script)
    try:
        asyncio.run(eng.answer("two things"))
        results = eng.client.messages.calls[1]["messages"][2]["content"]
        # Splitting these across messages degrades future parallel tool use.
        assert len(results) == 2
        assert {r["tool_use_id"] for r in results} == {"a", "b"}
    finally:
        eng.close()


def test_pause_turn_resumes_without_new_user_message(persona):
    script = [
        response([text_block("searching...")], "pause_turn"),
        response([text_block("found it")], "end_turn"),
    ]
    eng = build(persona, script)
    try:
        ans = asyncio.run(eng.answer("what's new"))
        assert ans.text == "found it"
        msgs = eng.client.messages.calls[1]["messages"]
        # user, assistant — no synthetic user turn injected.
        assert [m["role"] for m in msgs] == ["user", "assistant"]
    finally:
        eng.close()


def test_unknown_tool_returns_error_result_not_crash(persona):
    script = [
        response([tool_use("nonexistent_tool", inp={})], "tool_use"),
        response([text_block("recovered")], "end_turn"),
    ]
    eng = build(persona, script)
    try:
        ans = asyncio.run(eng.answer("q"))
        assert ans.text == "recovered"
        result = eng.client.messages.calls[1]["messages"][2]["content"][0]
        assert result["is_error"] is True
    finally:
        eng.close()


def test_refusal_is_handled(persona):
    eng = build(persona, [response([], "refusal")])
    try:
        ans = asyncio.run(eng.answer("something disallowed"))
        assert ans.stop_reason == "refusal"
        assert ans.text
    finally:
        eng.close()


def test_iteration_cap_terminates(persona):
    # Always asks for a tool — must stop, not spin.
    script = [
        response([tool_use("search_videos", tid=f"t{i}", inp={"query": "x"})], "tool_use")
        for i in range(20)
    ]
    eng = build(persona, script)
    try:
        ans = asyncio.run(eng.answer("loop forever"))
        assert ans.stop_reason == "max_iterations"
        assert len(eng.client.messages.calls) == persona.max_tool_iterations
    finally:
        eng.close()


def test_system_prompt_is_cached_and_stable(persona):
    script = [response([text_block("a")], "end_turn"), response([text_block("b")], "end_turn")]
    eng = build(persona, script)
    try:
        asyncio.run(eng.answer("one"))
        asyncio.run(eng.answer("two"))
        c0, c1 = eng.client.messages.calls
        assert c0["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        # Byte-identical prefix across calls, or the cache never hits.
        assert c0["system"][0]["text"] == c1["system"][0]["text"]
    finally:
        eng.close()


def test_request_shape_matches_model_requirements(persona):
    eng = build(persona, [response([text_block("x")], "end_turn")])
    try:
        asyncio.run(eng.answer("q"))
        call = eng.client.messages.calls[0]
        assert call["model"] == "claude-opus-5"
        assert call["thinking"] == {"type": "adaptive"}
        assert call["output_config"]["effort"] == persona.effort
        # budget_tokens is rejected with a 400 on Opus 5.
        assert "budget_tokens" not in call["thinking"]
        names = [t.get("name") for t in call["tools"]]
        assert "search_wiki" in names and "web_search" in names
    finally:
        eng.close()
