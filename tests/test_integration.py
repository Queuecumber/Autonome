"""Smoke test for the orchestrator's event-handling pipeline.

Mocks the LLM (Responses streaming) and verifies that an inbound event
flows through to a final response and gets persisted to the session.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from session_manager.event import Event
from session_manager.orchestrator import SessionOrchestrator, _to_input_items
from session_manager.session import SessionManager

# One turn as persisted: an event pair, reasoning before a tool call, the
# tool result, then reasoning before the final text.
_TURN = [
    {"role": "developer", "content": '{"event": "message"}'},
    {"role": "user", "content": "what time is it?"},
    {"type": "comment", "kind": "usage", "input_tokens": 10},
    {"type": "reasoning", "content": "I should check the clock."},
    {"type": "function_call", "call_id": "c1", "name": "get_time", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "c1", "output": "13:00"},
    {"type": "reasoning", "content": "Now answer."},
    {"role": "assistant", "content": "It's 13:00."},
]


# ── Input construction ───────────────────────────────────


def test_to_input_items_passes_session_items_through():
    """The session format is the Responses format: function_call and
    function_call_output go back on the wire exactly as stored."""
    items = _to_input_items(_TURN)
    assert items == [
        {"role": "developer", "content": '{"event": "message"}'},
        {"role": "user", "content": "what time is it?"},
        {"type": "function_call", "call_id": "c1", "name": "get_time", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "13:00"},
        {"role": "assistant", "content": "It's 13:00."},
    ]


def test_to_input_items_drops_comments_and_reasoning():
    """Comments are telemetry. Reasoning is persisted as flat text, not in
    the item shape the API returns, so it can't be replayed verbatim."""
    kept = _to_input_items(_TURN)
    assert not any(i.get("type") in ("comment", "reasoning") for i in kept)


def test_to_input_items_can_remap_the_developer_role():
    """Escape hatch for a backend that rejects developer items."""
    assert _to_input_items(_TURN, developer_role="user")[0] == {
        "role": "user", "content": '{"event": "message"}'}


def _orch(tmp_path, model: str = "test-model", **model_cfg):
    return SessionOrchestrator(
        config={"model": {"name": model, **model_cfg},
                "session": {"max_history_tokens": 100000},
                "binaries": {"store": str(tmp_path / "b"), "retention_days": 30}},
        session_dir=tmp_path,
    )


def test_developer_role_defaults_to_developer(tmp_path):
    """Responses keeps developer distinct and in position, so it's the
    default here — unlike chat completions, which folds it into system."""
    assert _orch(tmp_path).developer_role == "developer"
    assert _orch(tmp_path, developer_role="user").developer_role == "user"


def test_developer_role_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="developer_role"):
        _orch(tmp_path, developer_role="system")


# ── Responses stream mocks ───────────────────────────────


def _message_item(text: str):
    item = MagicMock()
    item.type = "message"
    content = MagicMock()
    content.text = text
    item.content = [content]
    return item


def _usage():
    u = MagicMock()
    u.input_tokens, u.output_tokens, u.total_tokens = 100, 50, 150
    u.input_tokens_details = MagicMock(cached_tokens=0)
    u.output_tokens_details = MagicMock(reasoning_tokens=0)
    return u


def _stream(text: str):
    """Async iterator of Responses events for a plain text answer."""
    item = _message_item(text)
    response = MagicMock()
    response.output = [item]
    response.usage = _usage()

    async def _gen():
        done = MagicMock()
        done.type = "response.output_item.done"
        done.item = item
        yield done
        completed = MagicMock()
        completed.type = "response.completed"
        completed.response = response
        yield completed

    return _gen()


def _mock_llm(orch, text: str, captured: dict | None = None):
    async def fake_create(**kwargs):
        if captured is not None:
            captured.clear()
            captured.update(kwargs)
        return _stream(text)
    orch.llm = MagicMock()
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = fake_create


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ── End to end ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_flows_to_response_and_persists(tmp_path):
    """End-to-end: build an event, hand it to the orchestrator, get a
    response back, confirm it lands in the session file."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    orch = _orch(sessions_dir)
    _mock_llm(orch, "hello back")

    result = await orch.handle_event(
        Event(source="matrix", text="hi", metadata={"room_id": "!r"}))
    assert result == "hello back"

    history = SessionManager(store_dir=sessions_dir, max_history_tokens=100000).load("main")
    assert "hello back" in [m.get("content") for m in history if m.get("role") == "assistant"]


@pytest.mark.asyncio
async def test_turn_is_sent_as_responses_input_not_messages(tmp_path):
    """The call carries `input` + `instructions`, never chat `messages`, and
    developer events keep their own role."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    orch = _orch(sessions_dir)
    captured: dict = {}
    _mock_llm(orch, "ok", captured)

    await orch.handle_event(Event(source="matrix", text="first", metadata={}))
    assert "messages" not in captured
    assert isinstance(captured["instructions"], str)
    assert any(i.get("role") == "developer" for i in captured["input"])


@pytest.mark.asyncio
async def test_usage_comment_uses_responses_token_fields(tmp_path):
    """Responses reports input_tokens/output_tokens, not prompt/completion."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    orch = _orch(sessions_dir)
    _mock_llm(orch, "hi")

    await orch.handle_event(Event(source="matrix", text="hi", metadata={}))
    history = SessionManager(store_dir=sessions_dir, max_history_tokens=100000).load("main")
    usage = [m for m in history if m.get("kind") == "usage"]
    assert usage and usage[0]["input_tokens"] == 100 and usage[0]["output_tokens"] == 50


@pytest.mark.asyncio
async def test_explicit_session_id_routes(tmp_path):
    """An event with an explicit session_id lands there, not in main."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    orch = _orch(sessions_dir)
    _mock_llm(orch, "ack")

    await orch.handle_event(Event(session_id="cron-target", source="time", text="tick"))

    mgr = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    assert mgr.load("cron-target") != []
    assert mgr.load("main") == []


@pytest.mark.asyncio
async def test_tool_failure_becomes_output_not_a_dead_turn(tmp_path):
    """handle_event runs in a bare create_task, so an exception escaping a
    tool call would kill the turn and discard everything collected for it.
    Failures have to come back as tool output instead."""
    orch = _orch(tmp_path)
    conn = MagicMock()
    conn.call_tool = AsyncMock(
        side_effect=ValueError("Cannot inline resource of type 'video/mp4'"))
    conn.binary_params = {}
    orch._tool_to_mcp = {"read_thing": conn}

    result, media = await orch._execute_tool_call("c1", "read_thing", "{}")
    assert result["type"] == "function_call_output"
    assert result["call_id"] == "c1"
    assert "video/mp4" in result["output"]
    assert media == []
