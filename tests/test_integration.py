"""Smoke test for the orchestrator's event-handling pipeline.

Mocks the LLM (Responses API streaming) and verifies that an inbound
event flows through to a final response and gets persisted to the
session.
"""

from unittest.mock import MagicMock

import pytest

from session_manager.event import Event
from session_manager.orchestrator import SessionOrchestrator
from session_manager.session import SessionManager


def _stream_event(event_type: str, **fields) -> MagicMock:
    """One event of the OpenAI Responses streaming protocol."""
    e = MagicMock()
    e.type = event_type
    for k, v in fields.items():
        setattr(e, k, v)
    return e


def _mock_message_item(text: str) -> MagicMock:
    item = MagicMock()
    item.type = "message"
    content = MagicMock()
    content.text = text
    item.content = [content]
    return item


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.status = "completed"
    resp.output = [_mock_message_item(text)]
    resp.usage = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    resp.usage.total_tokens = 150
    resp.usage.output_tokens_details = MagicMock()
    resp.usage.output_tokens_details.reasoning_tokens = 0
    return resp


def _stream(text: str):
    """Async iterator simulating a non-tool response stream."""
    response = _mock_response(text)

    async def _gen():
        yield _stream_event("response.created")
        yield _stream_event("response.output_item.done", item=_mock_message_item(text))
        yield _stream_event("response.completed", response=response)

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
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = fake_create

    event = Event(source="matrix", text="hi", metadata={"room_id": "!r"})
    result = await orch.handle_event(event)
    assert result == "hello back"

    # Persisted to "main" by default.
    mgr = SessionManager(store_dir=sessions_dir)
    history = mgr.load("main")
    contents = [m.get("content") for m in history if m.get("role") == "assistant"]
    assert "hello back" in contents


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
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = fake_create

    event = Event(session_id="cron-target", source="time", text="tick")
    await orch.handle_event(event)

    mgr = SessionManager(store_dir=sessions_dir)
    assert mgr.load("cron-target") != []
    assert mgr.load("main") == []
