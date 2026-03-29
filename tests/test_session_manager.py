"""Tests for session manager orchestrator service."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def _mock_proxy_response(content="I'll handle it!"):
    """Create a mock httpx response mimicking LiteLLM proxy."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={
        "choices": [{"message": {"role": "assistant", "content": content}}],
    })
    return resp


@pytest.fixture
def orchestrator(orchestrator_config, tmp_path):
    return SessionOrchestrator(
        config=orchestrator_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_path / "sessions",
    )


def test_orchestrator_init(orchestrator):
    assert orchestrator.litellm_url == "http://localhost:4000"
    assert len(orchestrator.mcp_tools) >= 1


def test_heartbeat_routes_to_primary_contact(orchestrator):
    assert orchestrator.heartbeat_source == "signal"
    assert orchestrator.heartbeat_session_id == "+11111111111"


@pytest.mark.asyncio
async def test_handle_event(orchestrator):
    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(return_value=_mock_proxy_response())

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
async def test_handle_event_sends_enriched_to_proxy(orchestrator):
    """The proxy request includes metadata context."""
    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(return_value=_mock_proxy_response())

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hey",
        "metadata": {"message_id": "ts_456"},
    }

    await orchestrator.handle_event(event)

    call_kwargs = orchestrator._http.post.call_args.kwargs
    payload = call_kwargs["json"]
    last_msg = payload["messages"][-1]
    assert "signal" in last_msg["content"]
    assert "ts_456" in last_msg["content"]
    assert "Hey" in last_msg["content"]
    # Verify MCP tools are included
    assert "tools" in payload
    assert len(payload["tools"]) >= 1


@pytest.mark.asyncio
async def test_handle_heartbeat(orchestrator):
    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(return_value=_mock_proxy_response())

    await orchestrator.handle_heartbeat()

    call_kwargs = orchestrator._http.post.call_args.kwargs
    payload = call_kwargs["json"]
    last_msg = payload["messages"][-1]
    assert "HEARTBEAT" in last_msg["content"]

    history = orchestrator.session.load("signal", "+11111111111")
    assert len(history) == 2


@pytest.mark.asyncio
async def test_handle_event_returns_none_on_failure(orchestrator):
    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=Exception("connection refused"))

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hello",
        "metadata": {},
    }

    result = await orchestrator.handle_event(event)
    assert result is None


@pytest.mark.asyncio
async def test_system_prompt_is_agents_md_only(orchestrator, tmp_workspace):
    """Only AGENTS.md is injected as system prompt."""
    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(return_value=_mock_proxy_response())

    (tmp_workspace / "AGENTS.md").write_text("# You are an agent\nUse tools to respond.")

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hi",
        "metadata": {},
    }

    await orchestrator.handle_event(event)

    call_kwargs = orchestrator._http.post.call_args.kwargs
    payload = call_kwargs["json"]
    system_msg = payload["messages"][0]
    assert system_msg["role"] == "system"
    assert "You are an agent" in system_msg["content"]
    assert "I am a test agent" not in system_msg["content"]


def test_http_event_endpoint(orchestrator):
    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(return_value=_mock_proxy_response())

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


def test_http_event_endpoint_returns_502_on_failure(orchestrator):
    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=Exception("fail"))

    app = create_app(orchestrator)
    client = TestClient(app)

    resp = client.post("/event", json={
        "source": "signal",
        "session_id": "+11111111111",
        "text": "test",
        "metadata": {},
    })
    assert resp.status_code == 502


def test_http_heartbeat_endpoint(orchestrator):
    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(return_value=_mock_proxy_response())

    app = create_app(orchestrator)
    client = TestClient(app)

    resp = client.post("/heartbeat")
    assert resp.status_code == 200
