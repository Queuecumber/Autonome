"""Smoke test for the orchestrator's event-handling pipeline.

Mocks the LLM (chat completions streaming) and verifies that an inbound
event flows through to a final response and gets persisted to the session.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from session_manager.event import Event
from session_manager.orchestrator import SessionOrchestrator, _to_chat_messages
from session_manager.session import SessionManager

# One turn as persisted: an event pair, reasoning, a tool call, its result,
# then the reply.
_TURN = [
    {"role": "developer", "content": '{"event": "message"}'},
    {"role": "user", "content": "what time is it?"},
    {"type": "comment", "kind": "usage", "input_tokens": 10},
    {"type": "reasoning", "content": "I should check the clock."},
    {"type": "function_call", "call_id": "c1", "name": "get_time", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "c1", "output": "13:00"},
    {"role": "assistant", "content": "It's 13:00."},
]


# ── Wire translation ─────────────────────────────────────


def test_event_and_text_become_one_multipart_user_message():
    """Context and text are separate items in the session file but one
    message on the wire, so the payload each text belongs to is explicit
    rather than positional."""
    msgs = _to_chat_messages(_TURN)
    assert msgs[0] == {"role": "user", "content": [
        {"type": "text", "text": '{"event": "message"}'},
        {"type": "text", "text": "what time is it?"},
    ]}


def test_batched_events_extend_the_same_message():
    """Several queued events draining together interleave into one message
    rather than emitting a pair each."""
    batched = [
        {"role": "developer", "content": '{"event": "message"}'},
        {"role": "user", "content": "Hello"},
        {"role": "developer", "content": '{"event": "continuity"}'},
        {"role": "user", "content": "✨"},
    ]
    msgs = _to_chat_messages(batched)
    assert len(msgs) == 1
    assert [p["text"] for p in msgs[0]["content"]] == [
        '{"event": "message"}', "Hello", '{"event": "continuity"}', "✨"]


def test_nothing_rides_as_developer_on_the_wire():
    """Chat completions coerces `developer` to `system`, which means a
    system message mid-conversation — rejected by strict templates and
    worthless on permissive ones. The role stays a session-file marker."""
    assert all(m["role"] != "developer" for m in _to_chat_messages(_TURN))


def test_comments_are_never_sent():
    """Comments are telemetry — usage counts and boundaries."""
    assert "usage" not in json.dumps(_to_chat_messages(_TURN))


def test_reasoning_rides_back_on_the_assistant_message():
    """Models trained with preserved thinking history read a transcript
    where no prior turn reasoned as a cue to stop reasoning themselves,
    which then persists nothing and makes the next turn equally bare."""
    call = next(m for m in _to_chat_messages(_TURN) if m.get("tool_calls"))
    assert call["reasoning_content"] == "I should check the clock."
    # never as a standalone item — it belongs to the assistant turn
    assert all(m["role"] != "reasoning" for m in _to_chat_messages(_TURN))


def test_reasoning_replay_can_be_disabled():
    msgs = _to_chat_messages(_TURN, replay_reasoning=False)
    assert "I should check the clock" not in json.dumps(msgs)


def test_replay_reasoning_defaults_on(tmp_path):
    assert _orch(tmp_path).replay_reasoning is True
    assert _orch(tmp_path, replay_reasoning=False).replay_reasoning is False


def test_tool_calls_and_results_round_trip():
    msgs = _to_chat_messages(_TURN)
    call = next(m for m in msgs if m.get("tool_calls"))
    assert call["tool_calls"][0]["function"]["name"] == "get_time"
    result = next(m for m in msgs if m["role"] == "tool")
    assert result["tool_call_id"] == "c1" and result["content"] == "13:00"
    assert msgs[-1] == {"role": "assistant", "content": "It's 13:00."}


def test_parallel_tool_calls_merge_into_one_assistant_message():
    """A turn that made three calls persists three items; on the wire they
    belong to one assistant message so each result can be matched to it."""
    items = [
        {"type": "function_call", "call_id": "a", "name": "one", "arguments": "{}"},
        {"type": "function_call", "call_id": "b", "name": "two", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "a", "output": "1"},
        {"type": "function_call_output", "call_id": "b", "output": "2"},
    ]
    msgs = _to_chat_messages(items)
    assert len(msgs[0]["tool_calls"]) == 2
    assert [m["role"] for m in msgs] == ["assistant", "tool", "tool"]


# ── Chat completions stream mocks ────────────────────────


def _chunk(*, content=None, tool_call=None, finish=None):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = None
    delta.tool_calls = None
    if tool_call is not None:
        tc = MagicMock()
        tc.index = 0
        tc.id = tool_call["id"]
        tc.function = MagicMock()
        tc.function.name = tool_call["name"]
        tc.function.arguments = tool_call["arguments"]
        delta.tool_calls = [tc]
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _usage_chunk():
    chunk = MagicMock()
    chunk.choices = []
    u = MagicMock()
    u.prompt_tokens, u.completion_tokens, u.total_tokens = 100, 50, 150
    u.prompt_tokens_details = MagicMock(cached_tokens=0)
    u.completion_tokens_details = MagicMock(reasoning_tokens=0)
    chunk.usage = u
    return chunk


def _stream(text: str):
    async def _gen():
        yield _chunk(content=text, finish="stop")
        yield _usage_chunk()
    return _gen()


def _orch(tmp_path, model: str = "test-model", **model_cfg):
    return SessionOrchestrator(
        config={"model": {"name": model, **model_cfg},
                "binaries": {"store": str(tmp_path / "b"), "retention_days": 30}},
        session_dir=tmp_path,
    )


def _mock_llm(orch, text: str, captured: dict | None = None):
    async def fake_create(**kwargs):
        if captured is not None:
            captured.clear()
            captured.update(kwargs)
        return _stream(text)
    orch.llm = MagicMock()
    orch.llm.chat = MagicMock()
    orch.llm.chat.completions = MagicMock()
    orch.llm.chat.completions.create = fake_create


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ── End to end ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_flows_to_response_and_persists(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    orch = _orch(sessions_dir)
    _mock_llm(orch, "hello back")

    result = await orch.handle_event(
        Event(source="matrix", text="hi", metadata={"room_id": "!r"}))
    assert result == "hello back"

    history = SessionManager(store_dir=sessions_dir).load("main")
    assert "hello back" in [m.get("content") for m in history if m.get("role") == "assistant"]


@pytest.mark.asyncio
async def test_turn_is_sent_as_chat_messages_with_no_developer_role(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    orch = _orch(sessions_dir)
    captured: dict = {}
    _mock_llm(orch, "ok", captured)

    await orch.handle_event(Event(source="matrix", text="first", metadata={}))
    msgs = captured["messages"]
    assert "input" not in captured and "instructions" not in captured
    assert msgs[0]["role"] == "system"
    assert all(m["role"] != "developer" for m in msgs)
    # The synthetic boot event and the real event batch into ONE message:
    # two payload/text pairs, four parts, no second user message.
    users = [m for m in msgs if m["role"] == "user"]
    assert len(users) == 1
    parts = users[0]["content"]
    assert [p["type"] for p in parts] == ["text"] * 4
    assert json.loads(parts[0]["text"])["event"] == "boot"
    assert json.loads(parts[2]["text"])["event"] == "message"
    assert parts[3]["text"] == "first"


@pytest.mark.asyncio
async def test_usage_comment_uses_chat_token_fields(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    orch = _orch(sessions_dir)
    _mock_llm(orch, "hi")

    await orch.handle_event(Event(source="matrix", text="hi", metadata={}))
    history = SessionManager(store_dir=sessions_dir).load("main")
    usage = [m for m in history if m.get("kind") == "usage"]
    assert usage and usage[0]["input_tokens"] == 100 and usage[0]["output_tokens"] == 50


@pytest.mark.asyncio
async def test_explicit_session_id_routes(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    orch = _orch(sessions_dir)
    _mock_llm(orch, "ack")

    await orch.handle_event(Event(session_id="cron-target", source="time", text="tick"))
    mgr = SessionManager(store_dir=sessions_dir)
    assert mgr.load("cron-target") != []
    assert mgr.load("main") == []


@pytest.mark.asyncio
async def test_tool_failure_becomes_output_not_a_dead_turn(tmp_path):
    """handle_event runs in a bare create_task, so an exception escaping a
    tool call would kill the turn and discard everything collected for it."""
    orch = _orch(tmp_path)
    conn = MagicMock()
    conn.call_tool = AsyncMock(
        side_effect=ValueError("Cannot inline resource of type 'video/mp4'"))
    conn.binary_params = {}
    orch._tool_to_mcp = {"read_thing": conn}

    result, media = await orch._execute_tool_call("c1", "read_thing", "{}")
    assert result["type"] == "function_call_output"
    assert "video/mp4" in result["output"]
    assert media == []


# ── Steering ─────────────────────────────────────────────
#
# An event arriving mid-turn is appended to the next tool result instead of
# cancelling the turn: nothing in flight is lost and no partial state has to
# be reconstructed. Role alternation is untouched — only existing content is
# modified — so tool_call pairing can't be broken by delivery.

from session_manager.orchestrator import _apply_steer_to_tool_results  # noqa: E402

_STEER = [
    {"role": "developer", "content": '{"event": "message", "source": "matrix"}'},
    {"role": "user", "content": "are you still there?"},
]


def _tool_parts(msg):
    return [json.loads(p["text"]) for p in msg["content"][1:]]


def test_steer_turns_tool_content_into_parts_without_restructuring():
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "first"},
        {"role": "tool", "tool_call_id": "b", "content": "second"},
    ]
    before = [m["role"] for m in msgs]
    assert _apply_steer_to_tool_results(msgs, 2, _STEER) is True
    assert [m["role"] for m in msgs] == before            # no message inserted
    assert msgs[1]["content"] == "first"                  # earlier result untouched
    parts = msgs[-1]["content"]
    assert parts[0] == {"type": "text", "text": "second"}  # tool output stays first
    # the events keep the exact shape they'd have had on their own
    assert _tool_parts(msgs[-1]) == _STEER


def test_steer_batches_multiple_events_in_order():
    msgs = [{"role": "tool", "tool_call_id": "a", "content": "out"}]
    two = _STEER + [{"role": "developer", "content": '{"event": "reaction"}'},
                    {"role": "user", "content": "👍"}]
    assert _apply_steer_to_tool_results(msgs, 1, two) is True
    assert _tool_parts(msgs[0]) == two


def test_steer_reports_failure_when_no_tool_result_to_carry_it():
    """Caller must fall back to delivering it as a normal next-turn event."""
    msgs = [{"role": "assistant", "content": "done"}]
    assert _apply_steer_to_tool_results(msgs, 1, _STEER) is False
    assert msgs == [{"role": "assistant", "content": "done"}]


def test_steer_extends_already_multipart_tool_content():
    msgs = [{"role": "tool", "tool_call_id": "a",
             "content": [{"type": "text", "text": "body"}]}]
    assert _apply_steer_to_tool_results(msgs, 1, _STEER) is True
    assert msgs[0]["content"][0] == {"type": "text", "text": "body"}
    assert _tool_parts(msgs[0]) == _STEER


def test_steer_is_a_noop_with_no_batch_or_no_events():
    msgs = [{"role": "tool", "tool_call_id": "a", "content": "x"}]
    assert _apply_steer_to_tool_results(msgs, 0, _STEER) is False
    assert _apply_steer_to_tool_results(msgs, 1, []) is False


@pytest.mark.asyncio
async def test_midturn_event_steers_instead_of_cancelling(tmp_path):
    """A busy session parks the event for delivery rather than setting cancel."""
    orch = _orch(tmp_path)
    state = orch._get_session("main")
    await state.lock.acquire()
    try:
        cancel = asyncio.Event()
        state.cancel = cancel
        assert await orch.handle_event(
            Event(source="matrix", text="you there?", metadata={})) is None
        assert not cancel.is_set()                      # turn was not torn down
        assert [e.text for e in state.pending_steer] == ["you there?"]
        assert state.passive_queue == []
    finally:
        state.lock.release()


@pytest.mark.asyncio
async def test_passive_event_still_waits_for_the_turn_to_end(tmp_path):
    """Low-value events shouldn't add noise to a turn already in progress."""
    orch = _orch(tmp_path)
    state = orch._get_session("main")
    await state.lock.acquire()
    try:
        await orch.handle_event(
            Event(source="time", text="✨", energy="passive", metadata={}))
        assert state.pending_steer == []
        assert [e.text for e in state.passive_queue] == ["✨"]
    finally:
        state.lock.release()
