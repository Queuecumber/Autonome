"""Smoke test for the orchestrator's event-handling pipeline.

Mocks the LLM (chat completions streaming) and verifies that an inbound
event flows through to a final response and gets persisted to the
session.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from session_manager.event import Event
from session_manager.orchestrator import (
    SessionOrchestrator,
    _media_user_message,
    _is_anthropic_model,
    _to_chat_messages,
)
from session_manager.session import SessionManager

# One turn: an event, a reasoning block before a tool call, the tool result,
# then a second reasoning block before the final text.
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


def test_to_chat_messages_drops_reasoning_by_default():
    """Anthropic can't replay unsigned reasoning, so it stays dropped."""
    msgs = _to_chat_messages(_TURN)
    assert not any("reasoning_content" in m for m in msgs)
    assert [m["role"] for m in msgs] == ["user", "user", "assistant", "tool", "assistant"]


def test_to_chat_messages_preserves_reasoning():
    """Each reasoning item rides back on the assistant message it preceded —
    the tool-call one and the final-text one alike."""
    msgs = _to_chat_messages(_TURN, reasoning_replay="native")
    tool_call_msg = next(m for m in msgs if m.get("tool_calls"))
    assert tool_call_msg["reasoning_content"] == "I should check the clock."
    assert msgs[-1] == {
        "role": "assistant",
        "reasoning_content": "Now answer.",
        "content": "It's 13:00.",
    }


def test_to_chat_messages_replays_reasoning_on_developer_channel():
    """Gateways that strip `reasoning_content` still get the thinking: it's
    duplicated onto a developer message placed immediately before the
    assistant message it belongs to."""
    msgs = _to_chat_messages(_TURN, developer_role="developer",
                             reasoning_replay="developer")
    tool_idx = next(i for i, m in enumerate(msgs) if m.get("tool_calls"))
    carrier = msgs[tool_idx - 1]
    assert carrier["role"] == "developer"
    assert json.loads(carrier["content"]) == {
        "event": "reasoning", "content": "I should check the clock."}
    # and the native field is still set, for backends that don't strip it
    assert msgs[tool_idx]["reasoning_content"] == "I should check the clock."


def test_native_replay_leaves_the_developer_channel_alone():
    """"native" is the default for non-Anthropic: the field is set, but no
    extra developer messages are spent on it."""
    msgs = _to_chat_messages(_TURN, developer_role="developer",
                             reasoning_replay="native")
    assert all("reasoning" not in (m.get("content") or "")
               for m in msgs if m["role"] == "developer")
    assert any("reasoning_content" in m for m in msgs)


def _orch(tmp_path, model: str, **model_cfg):
    return SessionOrchestrator(
        config={"model": {"name": model, **model_cfg},
                "session": {"max_history_tokens": 100},
                "binaries": {"store": str(tmp_path / "b"), "retention_days": 30}},
        session_dir=tmp_path,
    )


def test_reasoning_replay_defaults_by_model_family(tmp_path):
    """Anthropic drops it (no signature to replay); everyone else gets the
    native field. The developer-channel workaround is never automatic."""
    assert _orch(tmp_path, "aws/anthropic/bedrock-claude-opus-4-6").reasoning_replay == "none"
    assert _orch(tmp_path, "moonshotai/kimi-k3").reasoning_replay == "native"


def test_reasoning_replay_note_ships_only_when_opted_in(tmp_path):
    """The system-prompt section explaining the channel ships only where the
    channel is actually used."""
    assert "Your Prior Reasoning" not in _orch(
        tmp_path, "moonshotai/kimi-k3")._build_instructions()
    assert "Your Prior Reasoning" in _orch(
        tmp_path, "moonshotai/kimi-k3", reasoning_replay="developer")._build_instructions()


def test_reasoning_replay_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="reasoning_replay"):
        _orch(tmp_path, "moonshotai/kimi-k3", reasoning_replay="yes-please")


def test_to_chat_messages_developer_role_is_configurable():
    """Default folds developer into user for the Bedrock translation;
    backends that take `developer` natively get it verbatim."""
    assert _to_chat_messages(_TURN)[0]["role"] == "user"
    assert _to_chat_messages(_TURN, developer_role="developer")[0] == {
        "role": "developer", "content": '{"event": "message"}'}


def test_to_chat_messages_reasoning_never_orphans_a_message():
    """Reasoning with no assistant message behind it is discarded rather
    than emitted as a content-less assistant turn."""
    items = [{"type": "reasoning", "content": "thinking..."},
             {"role": "user", "content": "hi"}]
    assert _to_chat_messages(items, reasoning_replay="native") == [
        {"role": "user", "content": "hi"}]


def test_is_anthropic_model():
    assert _is_anthropic_model("aws/anthropic/bedrock-claude-opus-4-6")
    assert _is_anthropic_model("azure/anthropic/claude-opus-5")
    assert not _is_anthropic_model("moonshotai/kimi-k3")


def test_media_user_message_translates_input_image():
    """Responses-shape input_image parts become chat-completions image_url
    parts in a single follow-up user message."""
    image_items = [
        {"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAA"}]},
        {"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/jpeg;base64,BBB"}]},
    ]
    msg = _media_user_message(image_items)
    assert msg == {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
    ]}


def test_media_user_message_none_when_no_media():
    assert _media_user_message([]) is None
    assert _media_user_message([{"role": "user", "content": [
        {"type": "input_text", "text": "not an image"}]}]) is None


def test_media_user_message_translates_input_audio():
    """Audio rides as a data URI, not a format enum — the enum only accepts
    wav/mp3, while the data URI takes ogg/opus voice notes untranscoded."""
    msg = _media_user_message([
        {"role": "user", "content": [
            {"type": "input_audio", "audio_url": "data:audio/ogg;base64,AAA"}]},
    ])
    assert msg == {"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "data:audio/ogg;base64,AAA"}},
    ]}


def test_media_user_message_mixes_image_and_audio_in_order():
    msg = _media_user_message([
        {"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,IMG"}]},
        {"role": "user", "content": [
            {"type": "input_audio", "audio_url": "data:audio/ogg;base64,AUD"}]},
    ])
    assert [p["type"] for p in msg["content"]] == ["image_url", "audio_url"]


def _text_chunk(text: str, finish: str | None = None):
    """One chat-completions streaming chunk carrying text content."""
    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = None
    choice.delta = delta
    choice.finish_reason = finish
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _final_chunk():
    """Final chunk that carries usage info and no content."""
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = 100
    chunk.usage.completion_tokens = 50
    chunk.usage.total_tokens = 150
    chunk.usage.prompt_tokens_details = MagicMock()
    chunk.usage.prompt_tokens_details.cached_tokens = 0
    chunk.usage.cache_read_input_tokens = None
    chunk.usage.cache_creation_input_tokens = None
    return chunk


def _stream(text: str):
    """Async iterator simulating a non-tool chat-completions stream."""

    async def _gen():
        yield _text_chunk(text, finish="stop")
        yield _final_chunk()

    return _gen()


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.mark.asyncio
async def test_event_flows_to_response_and_persists(tmp_path):
    """End-to-end: build an event, hand it to the orchestrator, get a
    response back, confirm it lands in the session file."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    binary_dir = tmp_path / "binaries"

    config = {
        "model": {"name": "test-model"},
        "session": {"max_history_tokens": 100000},
        "binaries": {"store": str(binary_dir), "retention_days": 30},
    }

    orch = SessionOrchestrator(config=config, session_dir=sessions_dir)

    async def fake_create(**kwargs):
        return _stream("hello back")
    orch.llm = MagicMock()
    orch.llm.chat = MagicMock()
    orch.llm.chat.completions = MagicMock()
    orch.llm.chat.completions.create = fake_create

    event = Event(source="matrix", text="hi", metadata={"room_id": "!r"})
    result = await orch.handle_event(event)
    assert result == "hello back"

    # Persisted to "main" by default.
    mgr = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    history = mgr.load("main")
    contents = [m.get("content") for m in history if m.get("role") == "assistant"]
    assert "hello back" in contents


@pytest.mark.asyncio
async def test_rolling_cache_markers(tmp_path):
    """Each turn persists a turn_start comment, and the next turn places
    cache_control markers at the previous turn's boundary and the new tail
    (plus the pinned instructions marker)."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    config = {
        "model": {"name": "test-model"},
        "session": {"max_history_tokens": 100000},
        "binaries": {"store": str(tmp_path / "binaries"), "retention_days": 30},
    }
    orch = SessionOrchestrator(config=config, session_dir=sessions_dir)

    captured: dict = {}

    async def fake_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _stream("reply")
    orch.llm = MagicMock()
    orch.llm.chat = MagicMock()
    orch.llm.chat.completions = MagicMock()
    orch.llm.chat.completions.create = fake_create

    # Turn 1 — no boundary yet: markers on instructions + history tail only.
    await orch.handle_event(Event(source="matrix", text="first", metadata={}))
    history = SessionManager(store_dir=sessions_dir, max_history_tokens=100000).load("main")
    assert history[0] == {"type": "comment", "kind": "turn_start"}

    # Turn 2 — history now holds turn 1's block behind a turn_start comment.
    await orch.handle_event(Event(source="matrix", text="second", metadata={}))
    messages = captured["messages"]

    def has_marker(msg):
        content = msg.get("content")
        return isinstance(content, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in content
        )

    marked = [i for i, m in enumerate(messages) if has_marker(m)]
    # Instructions is message 0. The boot-event turn is turn 1's block, so
    # there is exactly one boundary: prev segment is empty, and the recent
    # segment (turn 1) gets a tail marker.
    assert 0 in marked
    # Exactly one history marker on turn 1's last message (the assistant
    # reply), placed before this turn's new event messages.
    history_marks = [i for i in marked if i > 0]
    assert len(history_marks) == 1
    assert messages[history_marks[0]].get("role") == "assistant"

    # Turn 3 — now a real prev segment exists; expect two history markers.
    await orch.handle_event(Event(source="matrix", text="third", metadata={}))
    messages = captured["messages"]
    history_marks = [i for i, m in enumerate(messages) if has_marker(m) and i > 0]
    assert len(history_marks) == 2


@pytest.mark.asyncio
async def test_explicit_session_id_routes(tmp_path):
    """An event with an explicit session_id lands there, not in main."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    binary_dir = tmp_path / "binaries"

    config = {
        "model": {"name": "test-model"},
        "session": {"max_history_tokens": 100000},
        "binaries": {"store": str(binary_dir), "retention_days": 30},
    }
    orch = SessionOrchestrator(config=config, session_dir=sessions_dir)

    async def fake_create(**kwargs):
        return _stream("ack")
    orch.llm = MagicMock()
    orch.llm.chat = MagicMock()
    orch.llm.chat.completions = MagicMock()
    orch.llm.chat.completions.create = fake_create

    event = Event(session_id="cron-target", source="time", text="tick")
    await orch.handle_event(event)

    mgr = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    assert mgr.load("cron-target") != []
    assert mgr.load("main") == []


@pytest.mark.asyncio
async def test_tool_failure_becomes_output_not_a_dead_turn(tmp_path):
    """handle_event runs in a bare create_task, so an exception escaping a
    tool call would kill the turn and discard everything collected for it.
    Failures have to come back as tool output instead."""
    orch = _orch(tmp_path, "test-model")

    conn = MagicMock()
    conn.call_tool = AsyncMock(side_effect=ValueError("Cannot inline resource of type 'video/mp4'"))
    conn.binary_params = {}
    orch._tool_to_mcp = {"read_thing": conn}

    result, media = await orch._execute_tool_call("c1", "read_thing", "{}")
    assert result["type"] == "function_call_output"
    assert result["call_id"] == "c1"
    assert "video/mp4" in result["output"]
    assert media == []
