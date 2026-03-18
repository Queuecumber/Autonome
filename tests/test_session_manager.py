"""Tests for session manager orchestrator service."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
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


@pytest.fixture
def mock_litellm_response():
    return {
        "choices": [{"message": {"role": "assistant", "content": "I'll handle it!"}}]
    }


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
    """Heartbeat routes to the first allowed Signal contact, not an orphan session."""
    assert orchestrator.heartbeat_source == "signal"
    assert orchestrator.heartbeat_session_id == "+11111111111"


@pytest.mark.asyncio
async def test_handle_event(orchestrator, mock_litellm_response):
    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hello!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'll handle it!"

    # Verify session was saved with clean text (no metadata)
    history = orchestrator.session.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hello!"  # Clean, no [signal | {...}] prefix
    assert history[1]["content"] == "I'll handle it!"


@pytest.mark.asyncio
async def test_handle_event_sends_enriched_to_llm(orchestrator, mock_litellm_response):
    """The LLM request includes metadata context, but history stores clean text."""
    captured_payload = {}

    async def mock_post(url, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hey",
        "metadata": {"message_id": "ts_456"},
    }

    await orchestrator.handle_event(event)

    # LLM sees enriched message
    last_msg = captured_payload["messages"][-1]
    assert "signal" in last_msg["content"]
    assert "ts_456" in last_msg["content"]
    assert "Hey" in last_msg["content"]


@pytest.mark.asyncio
async def test_handle_heartbeat(orchestrator, mock_litellm_response):
    captured_payload = {}

    async def mock_post(url, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    await orchestrator.handle_heartbeat()

    last_msg = captured_payload["messages"][-1]
    assert "HEARTBEAT" in last_msg["content"]
    assert "Check HEARTBEAT.md" in last_msg["content"]

    # Heartbeat saved in the primary contact's Signal session
    history = orchestrator.session.load("signal", "+11111111111")
    assert len(history) == 2


@pytest.mark.asyncio
async def test_handle_event_returns_none_on_failure(orchestrator):
    """When LiteLLM fails, handle_event returns None."""
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


def test_http_event_endpoint(orchestrator, mock_litellm_response):
    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

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


def test_http_heartbeat_endpoint(orchestrator, mock_litellm_response):
    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    app = create_app(orchestrator)
    client = TestClient(app)

    resp = client.post("/heartbeat")
    assert resp.status_code == 200
