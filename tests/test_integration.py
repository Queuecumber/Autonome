"""Integration smoke test: verify the full event flow with mocked externals."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from agent_platform.config import load_config
from agent_platform.session import SessionManager
from session_manager.server import SessionOrchestrator
from adapters.signal.model import SignalClient
from adapters.signal.mcp_server import SignalInterface


def _mock_llm_response(content="I'm here to help!"):
    message = MagicMock()
    message.content = content
    message.tool_calls = []
    message.model_dump = MagicMock(return_value={"role": "assistant", "content": content})

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
@patch("session_manager.server.litellm")
async def test_full_event_flow(mock_litellm, tmp_path):
    """End-to-end: config → orchestrator receives event → LLM called → session saved."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("# Agent\nUse tools to respond.\n")
    (workspace / "SOUL.md").write_text("# Soul\nYou are a helpful agent.\n")

    config_data = {
        "model": {"provider": "anthropic", "model": "claude-opus-4-6", "api_key": "test-key"},
        "workspace": str(workspace),
        "mcp_servers": {},
        "channels": {
            "signal": {
                "account": "+10000000000",
                "signal_cli": "http://localhost:8080",
                "allow_from": ["+11111111111"],
            },
        },
        "session": {"store": str(tmp_path / "sessions"), "max_history_tokens": 100000},
        "heartbeat": {"interval": "20m", "prompt": "Check HEARTBEAT.md"},
    }
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.dump(config_data))
    config = load_config(config_path)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    orchestrator = SessionOrchestrator(config=config, session_dir=sessions_dir)

    # Mock litellm
    mock_litellm.acompletion = AsyncMock(return_value=_mock_llm_response())

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hi there!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'm here to help!"

    # Verify system prompt is AGENTS.md only (not SOUL.md)
    call_kwargs = mock_litellm.acompletion.call_args.kwargs
    system_msg = call_kwargs["messages"][0]
    assert "Use tools to respond" in system_msg["content"]
    assert "helpful agent" not in system_msg["content"]  # SOUL.md not injected

    # Verify session persistence
    session = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    history = session.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hi there!"
    assert history[1]["content"] == "I'm here to help!"

    # Verify outbound MCP tools can be created
    signal_client = SignalClient(signal_cli_url="http://localhost:8080", account="+10000000000")
    interface = SignalInterface(client=signal_client, session_manager_url="http://localhost:5000")
    mcp = interface.mcp
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "send_message" in tool_names

    # Verify interface handles both directions
    assert interface.client.account == "+10000000000"
