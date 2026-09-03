"""Execution-path tests for a turn: parallel tool dispatch, mid-turn steering,
cancellation, and stream aggregation.

The wire-translation half of this (session items -> chat messages) lives in
test_integration.py. This file covers what happens *while* a turn runs, where
the model has already answered with tool calls and the orchestrator has to
dispatch them, fold in anything that arrived meanwhile, and survive being
interrupted at either of two distinct moments.
"""

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from session_manager.event import Event
from session_manager.orchestrator import SessionOrchestrator, _media_user_message
from session_manager.session import SessionManager


# ── Stream mocks ─────────────────────────────────────────


def _tc_delta(index, id, name, arguments):
    tc = MagicMock()
    tc.index = index
    tc.id = id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _chunk(*, content=None, reasoning=None, tool_calls=None, finish=None):
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _usage_chunk():
    chunk = MagicMock()
    chunk.choices = []
    u = MagicMock()
    u.prompt_tokens, u.completion_tokens, u.total_tokens = 10, 5, 15
    u.prompt_tokens_details = MagicMock(cached_tokens=0)
    u.completion_tokens_details = MagicMock(reasoning_tokens=0)
    chunk.usage = u
    return chunk


def _stream(*chunks):
    def _make():
        async def _gen():
            for c in chunks:
                yield c
        return _gen()
    return _make


def _stream_then(action, *chunks):
    """Run `action` after the last chunk is consumed but before the stream
    ends — the only window where cancellation lands on a *complete* response
    that hasn't dispatched its tools yet."""
    def _make():
        async def _gen():
            for c in chunks:
                yield c
            action()
        return _gen()
    return _make


def _stream_after_first(action, *chunks):
    """Run `action` once one chunk has arrived, so the cancel is observed
    mid-stream with partial output already collected."""
    def _make():
        async def _gen():
            first, *rest = chunks
            yield first
            res = action()
            if asyncio.iscoroutine(res):
                await res
            for c in rest:
                yield c
        return _gen()
    return _make


def _tool_turn(*calls, finish="tool_calls"):
    """One assistant turn that answers with N tool calls."""
    return _stream(
        _chunk(tool_calls=[_tc_delta(i, c["id"], c["name"], c.get("args", "{}"))
                           for i, c in enumerate(calls)], finish=finish),
        _usage_chunk())


def _mock_llm_turns(orch, *makers, captured=None):
    """Each model round trip in the tool loop consumes the next stream."""
    queue = list(makers)
    async def fake_create(**kwargs):
        if captured is not None:
            captured.append(kwargs)
        return queue.pop(0)()
    orch.llm = MagicMock()
    orch.llm.chat = MagicMock()
    orch.llm.chat.completions = MagicMock()
    orch.llm.chat.completions.create = fake_create


def _orch(session_dir, debounce: float = 0):
    return SessionOrchestrator(
        config={"model": {"name": "test-model"},
                "session": {"debounce_seconds": debounce},
                "binaries": {"store": str(session_dir / "b"), "retention_days": 30}},
        session_dir=session_dir,
    )


@pytest.fixture
def sessions(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _tool_msgs(messages):
    return [m for m in messages if m.get("role") == "tool"]


def _dev_events(history):
    return [json.loads(m["content"]) for m in history if m.get("role") == "developer"]


# ── Parallel tool dispatch ───────────────────────────────
#
# A turn calling read_receipt + typing_indicator + send_message used to pay
# three sequential round trips to three different MCP servers.


@pytest.mark.asyncio
async def test_tool_calls_in_one_batch_run_concurrently(sessions):
    orch = _orch(sessions)
    order = []

    async def slow_tool(call_id, name, arguments):
        order.append(("start", name))
        await asyncio.sleep(0.1)
        order.append(("end", name))
        return {"type": "function_call_output", "call_id": call_id, "output": name}, []

    orch._execute_tool_call = slow_tool
    _mock_llm_turns(
        orch,
        _tool_turn({"id": "a", "name": "one"}, {"id": "b", "name": "two"},
                   {"id": "c", "name": "three"}),
        _stream(_chunk(content="done", finish="stop"), _usage_chunk()))

    started = time.monotonic()
    assert await orch.handle_event(
        Event(source="matrix", text="go", metadata={})) == "done"
    elapsed = time.monotonic() - started

    # Every call started before any finished — the ordering signal, which
    # holds regardless of how loaded the machine is.
    assert [phase for phase, _ in order[:3]] == ["start", "start", "start"]
    assert elapsed < 0.25, f"sequential dispatch would take ~0.3s, took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_results_stay_aligned_with_their_calls(sessions):
    """gather preserves positional order, so a fast tool finishing first must
    not shift its result onto another call's id."""
    orch = _orch(sessions)
    delays = {"slow": 0.15, "medium": 0.05, "fast": 0.0}

    async def tool(call_id, name, arguments):
        await asyncio.sleep(delays[name])
        return {"type": "function_call_output", "call_id": call_id,
                "output": f"{name}-out"}, []

    orch._execute_tool_call = tool
    captured: list = []
    _mock_llm_turns(
        orch,
        _tool_turn({"id": "a", "name": "slow"}, {"id": "b", "name": "medium"},
                   {"id": "c", "name": "fast"}),
        _stream(_chunk(content="ok", finish="stop"), _usage_chunk()),
        captured=captured)

    await orch.handle_event(Event(source="matrix", text="go", metadata={}))

    results = [(m["tool_call_id"], m["content"]) for m in _tool_msgs(captured[1]["messages"])]
    assert results == [("a", "slow-out"), ("b", "medium-out"), ("c", "fast-out")]


# ── Steering into a running turn ─────────────────────────


@pytest.mark.asyncio
async def test_midturn_event_rides_out_on_the_last_tool_result(sessions):
    """The event reaches the model within the turn it arrived in, without a
    message being inserted — role alternation and tool_call pairing are
    untouched, so delivery can't malform the next request."""
    orch = _orch(sessions)

    async def tool(call_id, name, arguments):
        # Arrives while the lock is held, so handle_event routes it to steer.
        await orch.handle_event(
            Event(source="matrix", text="you there?", metadata={"room_id": "!r"}))
        return {"type": "function_call_output", "call_id": call_id, "output": "out"}, []

    orch._execute_tool_call = tool
    captured: list = []
    _mock_llm_turns(
        orch,
        _tool_turn({"id": "a", "name": "check"}),
        _stream(_chunk(content="still here", finish="stop"), _usage_chunk()),
        captured=captured)

    assert await orch.handle_event(
        Event(source="matrix", text="hi", metadata={})) == "still here"

    second = captured[1]["messages"]
    # No message was inserted to carry it: the event is delivered purely by
    # modifying content that was already there.
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    parts = _tool_msgs(second)[-1]["content"]
    assert parts[0] == {"type": "text", "text": "out"}      # tool output stays first
    payloads = [json.loads(p["text"]) for p in parts[1:]]
    assert json.loads(payloads[0]["content"])["source"] == "matrix"
    assert payloads[1] == {"role": "user", "content": "you there?"}


@pytest.mark.asyncio
async def test_steered_event_is_persisted_exactly_once(sessions):
    """It rides out inside a tool result *and* is written as ordinary event
    items — the transcript must not end up with it twice."""
    orch = _orch(sessions)

    async def tool(call_id, name, arguments):
        await orch.handle_event(Event(source="matrix", text="ping", metadata={}))
        return {"type": "function_call_output", "call_id": call_id, "output": "out"}, []

    orch._execute_tool_call = tool
    _mock_llm_turns(
        orch,
        _tool_turn({"id": "a", "name": "check"}),
        _stream(_chunk(content="ok", finish="stop"), _usage_chunk()))

    await orch.handle_event(Event(source="matrix", text="hi", metadata={}))

    history = SessionManager(store_dir=sessions).load("main")
    assert [m.get("content") for m in history].count("ping") == 1


@pytest.mark.asyncio
async def test_event_with_no_tool_result_to_carry_it_becomes_its_own_turn(sessions):
    """A turn that answers with plain text has nothing to append to, so the
    event must fall back to a normal turn rather than being dropped."""
    orch = _orch(sessions)
    captured: list = []

    def inject():
        return orch.handle_event(Event(source="matrix", text="you there?", metadata={}))

    _mock_llm_turns(
        orch,
        _stream_after_first(inject,
                            _chunk(content="thinking"),
                            _chunk(content=" out loud", finish="stop"),
                            _usage_chunk()),
        _stream(_chunk(content="yes", finish="stop"), _usage_chunk()),
        captured=captured)

    await orch.handle_event(Event(source="matrix", text="hi", metadata={}))

    assert len(captured) == 2, "deferred event should have run a second turn"
    texts = [p["text"] for m in captured[1]["messages"] if m["role"] == "user"
             for p in (m["content"] if isinstance(m["content"], list) else [])]
    assert "you there?" in texts


# ── Interruption ─────────────────────────────────────────
#
# Two distinct moments: mid-stream (partial output collected) and after a
# complete response but before its tools are dispatched.


@pytest.mark.asyncio
async def test_interrupt_midstream_records_what_arrived(sessions):
    orch = _orch(sessions)

    def cancel():
        orch._get_session("main").cancel.set()

    _mock_llm_turns(orch, _stream_after_first(
        cancel,
        _chunk(content="I was saying"),
        _chunk(content=" more", finish="stop"),
        _usage_chunk()))

    assert await orch.handle_event(
        Event(source="matrix", text="hi", metadata={})) is None

    history = SessionManager(store_dir=sessions).load("main")
    ev = next(e for e in _dev_events(history) if e["event"] == "interrupted")
    assert ev["partial"] == [{"text": "I was saying"}]


@pytest.mark.asyncio
async def test_cancel_before_dispatch_runs_no_tools_and_records_them(sessions):
    """A batch that has started runs to completion, so the check sits before
    dispatch: every tool_call needs a matching result or the next request is
    malformed."""
    orch = _orch(sessions)
    ran = []

    async def tool(call_id, name, arguments):
        ran.append(name)
        return {"type": "function_call_output", "call_id": call_id, "output": "x"}, []

    orch._execute_tool_call = tool

    def cancel():
        orch._get_session("main").cancel.set()

    _mock_llm_turns(orch, _stream_then(
        cancel,
        _chunk(tool_calls=[_tc_delta(0, "a", "send_message", '{"text": "hi"}')],
               finish="tool_calls"),
        _usage_chunk()))

    assert await orch.handle_event(
        Event(source="matrix", text="hi", metadata={})) is None
    assert ran == [], "tools must not be dispatched once cancelled"

    history = SessionManager(store_dir=sessions).load("main")
    ev = next(e for e in _dev_events(history) if e["event"] == "interrupted")
    assert ev["pending"] == [{"tool": "send_message", "arguments": {"text": "hi"}}]


# ── Reasoning across tool iterations ─────────────────────


@pytest.mark.asyncio
async def test_reasoning_rides_back_on_the_in_turn_assistant_message(sessions):
    """Models trained with preserved thinking history need their own prior
    reasoning replayed to keep thinking across tool iterations."""
    orch = _orch(sessions)

    async def tool(call_id, name, arguments):
        return {"type": "function_call_output", "call_id": call_id, "output": "13:00"}, []

    orch._execute_tool_call = tool
    captured: list = []
    _mock_llm_turns(
        orch,
        _stream(_chunk(reasoning="Need the clock."),
                _chunk(tool_calls=[_tc_delta(0, "a", "get_time", "{}")],
                       finish="tool_calls"),
                _usage_chunk()),
        _stream(_chunk(content="13:00", finish="stop"), _usage_chunk()),
        captured=captured)

    await orch.handle_event(Event(source="matrix", text="time?", metadata={}))

    call_msg = next(m for m in captured[1]["messages"] if m.get("tool_calls"))
    assert call_msg["reasoning_content"] == "Need the clock."


# ── Tool-result media ────────────────────────────────────


def test_media_becomes_one_user_message_of_image_parts():
    """Binaries can't ride inside a tool message (string content only), so
    they follow as a user message."""
    items = [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,AAA"}]}]
    assert _media_user_message(items) == {
        "role": "user",
        "content": [{"type": "image_url",
                     "image_url": {"url": "data:image/png;base64,AAA"}}]}


def test_audio_is_deliberately_absent_from_media_messages():
    """The models we target reject audio_url outright, so mcp.py hands back a
    pointer instead and nothing should synthesize a message for it."""
    items = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]
    assert _media_user_message(items) is None
    assert _media_user_message([]) is None


@pytest.mark.asyncio
async def test_tool_media_follows_the_tool_results(sessions):
    orch = _orch(sessions)

    async def tool(call_id, name, arguments):
        return ({"type": "function_call_output", "call_id": call_id, "output": "a photo"},
                [{"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,AAA"}]}])

    orch._execute_tool_call = tool
    captured: list = []
    _mock_llm_turns(
        orch,
        _tool_turn({"id": "a", "name": "read_image"}),
        _stream(_chunk(content="nice", finish="stop"), _usage_chunk()),
        captured=captured)

    await orch.handle_event(Event(source="matrix", text="look", metadata={}))

    last = captured[1]["messages"][-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "image_url"


# ── Stream aggregation ───────────────────────────────────


@pytest.mark.asyncio
async def test_tool_call_arguments_accumulate_across_deltas(sessions):
    """Arguments arrive split across chunks; the id and name arrive once."""
    orch = _orch(sessions)
    _mock_llm_turns(orch, _stream(
        _chunk(tool_calls=[_tc_delta(0, "a", "send_message", '{"te')]),
        _chunk(tool_calls=[_tc_delta(0, None, None, 'xt": "hi"}')]),
        _chunk(finish="tool_calls"),
        _usage_chunk()))

    response, partial = await orch._stream_response({"model": "m"}, asyncio.Event())
    assert partial is None
    assert response["tool_calls"] == [
        {"id": "a", "type": "function",
         "function": {"name": "send_message", "arguments": '{"text": "hi"}'}}]
    assert response["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_parallel_tool_calls_are_ordered_by_delta_index(sessions):
    orch = _orch(sessions)
    _mock_llm_turns(orch, _stream(
        _chunk(tool_calls=[_tc_delta(1, "b", "second", "{}")]),
        _chunk(tool_calls=[_tc_delta(0, "a", "first", "{}")], finish="tool_calls"),
        _usage_chunk()))

    response, _ = await orch._stream_response({"model": "m"}, asyncio.Event())
    assert [tc["function"]["name"] for tc in response["tool_calls"]] == ["first", "second"]


@pytest.mark.asyncio
async def test_empty_reasoning_delta_is_counted_but_adds_no_text(sessions):
    """Distinguishes "field absent" from "field arrived empty" — the usage
    counters can't tell those apart and neither can a length check."""
    orch = _orch(sessions)
    _mock_llm_turns(orch, _stream(
        _chunk(reasoning=""),
        _chunk(content="hi", finish="stop"),
        _usage_chunk()))

    response, _ = await orch._stream_response({"model": "m"}, asyncio.Event())
    assert response["reasoning_deltas"] == 1
    assert response["reasoning"] == ""


# ── Debounce edge ────────────────────────────────────────


@pytest.mark.asyncio
async def test_debounce_with_nothing_collected_runs_no_turn(sessions):
    """The timer can fire on an already-drained batch; that must not start a
    turn with no events in it."""
    orch = _orch(sessions, debounce=0.01)
    orch.llm = MagicMock()  # any model call would raise
    state = orch._get_session("main")
    state.debounce_batch = []
    assert await orch._run_after_debounce("main") is None
