"""Tests for session manager orchestrator service."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from session_manager.server import create_app, SessionOrchestrator


@pytest.fixture
def orchestrator_config(tmp_workspace):
    return {
        "model": {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "api_key": "test-key",
        },
        "workspace": str(tmp_workspace),
        "mcp_servers": {},
        "channels": {
            "signal": {
                "account": "+10000000000",
                "signal_cli": "http://localhost:8080",
                "allow_from": ["+11111111111"],
            }
        },
        "session": {
            "store": "/tmp/test-sessions",
            "max_history_tokens": 100000,
        },
        "heartbeat": {
            "interval": "20m",
            "prompt": "Check HEARTBEAT.md",
        },
    }


def _mock_llm_response(content="I'll handle it!", tool_calls=None):
    """Create a mock litellm response."""
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
def orchestrator(orchestrator_config, tmp_path):
    return SessionOrchestrator(
        config=orchestrator_config,
        session_dir=tmp_path / "sessions",
    )


def test_orchestrator_init(orchestrator):
    assert orchestrator.model == "anthropic/claude-opus-4-6"
    assert orchestrator.api_key == "test-key"


def test_heartbeat_routes_to_primary_contact(orchestrator):
    assert orchestrator.heartbeat_source == "signal"
    assert orchestrator.heartbeat_session_id == "+11111111111"


@pytest.mark.asyncio
@patch("session_manager.server.litellm")
async def test_handle_event(mock_litellm, orchestrator):
    mock_litellm.acompletion = AsyncMock(return_value=_mock_llm_response("I'll handle it!"))

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hello!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'll handle it!"

    # Verify session was saved with clean text
    history = orchestrator.session.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hello!"
    assert history[1]["content"] == "I'll handle it!"


@pytest.mark.asyncio
@patch("session_manager.server.litellm")
async def test_handle_event_sends_enriched_to_llm(mock_litellm, orchestrator):
    """The LLM request includes metadata context."""
    mock_litellm.acompletion = AsyncMock(return_value=_mock_llm_response())

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hey",
        "metadata": {"message_id": "ts_456"},
    }

    await orchestrator.handle_event(event)

    # Check what was sent to litellm
    call_kwargs = mock_litellm.acompletion.call_args.kwargs
    last_msg = call_kwargs["messages"][-1]
    assert "signal" in last_msg["content"]
    assert "ts_456" in last_msg["content"]
    assert "Hey" in last_msg["content"]


@pytest.mark.asyncio
@patch("session_manager.server.litellm")
async def test_handle_heartbeat(mock_litellm, orchestrator):
    mock_litellm.acompletion = AsyncMock(return_value=_mock_llm_response())

    await orchestrator.handle_heartbeat()

    call_kwargs = mock_litellm.acompletion.call_args.kwargs
    last_msg = call_kwargs["messages"][-1]
    assert "HEARTBEAT" in last_msg["content"]

    history = orchestrator.session.load("signal", "+11111111111")
    assert len(history) == 2


@pytest.mark.asyncio
@patch("session_manager.server.litellm")
async def test_handle_event_returns_none_on_failure(mock_litellm, orchestrator):
    mock_litellm.acompletion = AsyncMock(side_effect=Exception("connection refused"))

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hello",
        "metadata": {},
    }

    result = await orchestrator.handle_event(event)
    assert result is None


@pytest.mark.asyncio
@patch("session_manager.server.litellm")
async def test_system_prompt_is_agents_md_only(mock_litellm, orchestrator, tmp_workspace):
    """Only AGENTS.md is injected as system prompt, not all workspace *.md files."""
    mock_litellm.acompletion = AsyncMock(return_value=_mock_llm_response())

    # Create AGENTS.md in workspace
    (tmp_workspace / "AGENTS.md").write_text("# You are an agent\nUse tools to respond.")

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hi",
        "metadata": {},
    }

    await orchestrator.handle_event(event)

    call_kwargs = mock_litellm.acompletion.call_args.kwargs
    system_msg = call_kwargs["messages"][0]
    assert system_msg["role"] == "system"
    assert "You are an agent" in system_msg["content"]
    # Should NOT include SOUL.md or USER.md content (agent reads those itself)
    assert "I am a test agent" not in system_msg["content"]


@patch("session_manager.server.litellm")
def test_http_event_endpoint(mock_litellm, orchestrator):
    mock_litellm.acompletion = AsyncMock(return_value=_mock_llm_response())

    app = create_app(orchestrator)
    client = TestClient(app)

    resp = client.post("/event", json={
        "source": "signal",
        "session_id": "+11111111111",
        "text": "test via HTTP",
        "metadata": {},
    })
    assert resp.status_code == 200
    assert "response" in resp.json()


@patch("session_manager.server.litellm")
def test_http_event_endpoint_returns_502_on_failure(mock_litellm, orchestrator):
    mock_litellm.acompletion = AsyncMock(side_effect=Exception("fail"))

    app = create_app(orchestrator)
    client = TestClient(app)

    resp = client.post("/event", json={
        "source": "signal",
        "session_id": "+11111111111",
        "text": "test",
        "metadata": {},
    })
    assert resp.status_code == 502


@patch("session_manager.server.litellm")
def test_http_heartbeat_endpoint(mock_litellm, orchestrator):
    mock_litellm.acompletion = AsyncMock(return_value=_mock_llm_response())

    app = create_app(orchestrator)
    client = TestClient(app)

    resp = client.post("/heartbeat")
    assert resp.status_code == 200
