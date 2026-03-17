import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from adapters.signal.adapter import SignalAdapter


@pytest.fixture
def adapter_config(sample_config, tmp_sessions):
    """Create adapter with mocked dependencies."""
    from agent_platform.config import load_config
    config = load_config(sample_config)
    return config


@pytest.fixture
def mock_litellm_response():
    """A mock LiteLLM chat completion response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm your agent.",
                }
            }
        ]
    }


def test_adapter_init(adapter_config, tmp_sessions):
    """Adapter initializes with config and creates session manager."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )
    assert adapter.litellm_url == "http://localhost:4000"
    assert adapter.allow_from == ["+11111111111"]


@pytest.mark.asyncio
async def test_handle_message_happy_path(adapter_config, tmp_sessions, mock_litellm_response):
    """handle_message sends to LiteLLM and saves session."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )

    async def mock_post(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        if "chat/completions" in url:
            mock_resp.json = MagicMock(return_value=mock_litellm_response)
        else:
            mock_resp.json = MagicMock(return_value={})
        return mock_resp

    adapter._http = AsyncMock()
    adapter._http.post = AsyncMock(side_effect=mock_post)

    result = await adapter.handle_message("+11111111111", "Hello!")
    assert result == "Hello! I'm your agent."

    # Verify session was saved
    history = adapter.session.load("+11111111111")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_handle_message_unauthorized(adapter_config, tmp_sessions):
    """Messages from unauthorized senders are ignored."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )
    result = await adapter.handle_message("+19999999999", "Hello!")
    assert result is None


@pytest.mark.asyncio
async def test_handle_heartbeat(adapter_config, tmp_sessions, mock_litellm_response):
    """Heartbeat sends configured prompt to first allowed contact."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )

    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "chat/completions" in url:
            resp.json = MagicMock(return_value=mock_litellm_response)
            payload = kwargs.get("json", {})
            last_msg = payload["messages"][-1]["content"]
            assert "[HEARTBEAT]" in last_msg
            assert "Check HEARTBEAT.md" in last_msg
        else:
            resp.json = MagicMock(return_value={})
        return resp

    adapter._http = AsyncMock()
    adapter._http.post = AsyncMock(side_effect=mock_post)

    result = await adapter.handle_heartbeat()
    assert result == "Hello! I'm your agent."

    history = adapter.session.load("+11111111111")
    assert len(history) == 2
    assert "[HEARTBEAT]" in history[0]["content"]


@pytest.mark.asyncio
async def test_poll_messages_processes_envelope(adapter_config, tmp_sessions, mock_litellm_response):
    """poll_messages correctly parses signal-cli envelope format."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )

    signal_response = [
        {
            "envelope": {
                "source": "+11111111111",
                "dataMessage": {
                    "message": "Test from Signal",
                    "timestamp": 1234567890,
                },
            }
        }
    ]

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=signal_response)
        return resp

    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "chat/completions" in url:
            resp.json = MagicMock(return_value=mock_litellm_response)
        else:
            resp.json = MagicMock(return_value={})
        return resp

    adapter._http = AsyncMock()
    adapter._http.get = AsyncMock(side_effect=mock_get)
    adapter._http.post = AsyncMock(side_effect=mock_post)

    await adapter.poll_messages()

    history = adapter.session.load("+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Test from Signal"
