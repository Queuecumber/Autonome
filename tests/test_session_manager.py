"""Tests for session manager orchestrator service."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from session_manager.orchestrator import SessionOrchestrator


@pytest.fixture
def orchestrator_config(tmp_workspace, tmp_path):
    return {
        "model": {
            "model": "claude-opus-4-6",
        },
        "workspace": str(tmp_workspace),
        "session": {
            "store": "/tmp/test-sessions",
            "max_history_tokens": 100000,
        },
        "binaries": {
            "store": str(tmp_path / "binaries"),
        },
    }


def _mock_llm_response(content="I'll handle it!", tool_calls=None):
    """Create a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    message.model_dump = MagicMock(return_value={
        "role": "assistant",
        "content": content,
        "tool_calls": [tc.model_dump() for tc in (tool_calls or [])] if tool_calls else None,
    })

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls" if tool_calls else "stop"

    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def orchestrator(orchestrator_config, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    orch = SessionOrchestrator(
        config=orchestrator_config,
        session_dir=tmp_path / "sessions",
    )
    # Mock the OpenAI client
    orch.llm = MagicMock()
    orch.llm.chat = MagicMock()
    orch.llm.chat.completions = MagicMock()
    orch.llm.chat.completions.create = AsyncMock(return_value=_mock_llm_response())
    return orch


def test_orchestrator_init(orchestrator):
    assert orchestrator.model == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_handle_event(orchestrator):
    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hello!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'll handle it!"

    history = orchestrator.session.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hello!"
    assert history[1]["content"] == "I'll handle it!"


@pytest.mark.asyncio
async def test_handle_event_sends_enriched_to_llm(orchestrator):
    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hey",
        "metadata": {"message_id": "ts_456"},
    }

    await orchestrator.handle_event(event)

    call_kwargs = orchestrator.llm.chat.completions.create.call_args.kwargs
    last_msg = call_kwargs["messages"][-1]
    assert "signal" in last_msg["content"]
    assert "ts_456" in last_msg["content"]
    assert "Hey" in last_msg["content"]


@pytest.mark.asyncio
async def test_handle_event_returns_none_on_failure(orchestrator):
    orchestrator.llm.chat.completions.create = AsyncMock(side_effect=Exception("connection refused"))

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hello",
        "metadata": {},
    }

    result = await orchestrator.handle_event(event)
    assert result is None


@pytest.mark.asyncio
async def test_system_prompt_includes_base_instructions(orchestrator):
    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hi",
        "metadata": {},
    }

    await orchestrator.handle_event(event)

    call_kwargs = orchestrator.llm.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]
    assert system_msg["role"] == "system"
    assert "MCP tools" in system_msg["content"]
    assert "send_message" in system_msg["content"]




