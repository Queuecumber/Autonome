"""Integration smoke test: verify the full event flow with mocked externals."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from agent_platform.config import load_config, generate_litellm_config, build_mcp_tool_declarations
from agent_platform.session import SessionManager
from agent_platform.callbacks.system_prompt import build_system_message
from session_manager.server import SessionOrchestrator, create_app
from adapters.signal.inbound import SignalInbound
from adapters.signal.outbound_mcp import SignalSender, create_signal_mcp


@pytest.mark.asyncio
async def test_full_event_flow(tmp_path):
    """End-to-end: config -> session manager receives event -> LLM called -> session saved."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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

    # Verify system prompt
    system_msg = build_system_message(workspace)
    assert "helpful agent" in system_msg

    # Verify LiteLLM config generation
    litellm_config = generate_litellm_config(config)
    assert "workspace_fs" in litellm_config["mcp_servers"]
    assert "signal" in litellm_config["mcp_servers"]

    # Create orchestrator and handle an event
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    orchestrator = SessionOrchestrator(
        config=config,
        litellm_url="http://localhost:4000",
        session_dir=sessions_dir,
    )

    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "I'm here to help!"}}]
    }

    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hi there!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'm here to help!"

    # Verify session persistence with channel key
    session = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    history = session.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hi there!"  # Clean text, no metadata
    assert history[1]["content"] == "I'm here to help!"

    # Verify outbound MCP tools exist
    sender = SignalSender(signal_cli_url="http://localhost:8080", account="+10000000000")
    mcp = create_signal_mcp(sender)
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "send_message" in tool_names
    assert "send_attachment" in tool_names

    # Verify inbound listener can be created
    inbound = SignalInbound(
        signal_cli_url="http://localhost:8080",
        session_manager_url="http://localhost:5000",
        account="+10000000000",
        allow_from=["+11111111111"],
    )
    assert inbound.account == "+10000000000"
