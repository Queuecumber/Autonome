"""Smoke test for the orchestrator's event-handling pipeline.

Mocks the LLM (chat completions streaming) and verifies that an inbound
event flows through to a final response and gets persisted to the
session.
"""

from unittest.mock import MagicMock

import pytest

from session_manager.event import Event
from session_manager.orchestrator import SessionOrchestrator
from session_manager.session import SessionManager


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
