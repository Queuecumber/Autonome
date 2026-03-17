"""Integration smoke test: verify the full message flow with mocked externals."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from agent_platform.config import load_config, generate_litellm_config, build_mcp_tool_declarations
from agent_platform.session import SessionManager
from agent_platform.callbacks.system_prompt import build_system_message
from adapters.signal.adapter import SignalAdapter
from adapters.signal.channel_mcp import create_channel_mcp, staged_attachments


@pytest.mark.asyncio
async def test_full_message_flow(tmp_path):
    """End-to-end: config -> system prompt -> session -> adapter -> response."""
    # 1. Set up workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "SOUL.md").write_text("# Soul\nYou are a helpful agent.\n")

    # 2. Create config
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

    # 3. Verify system prompt injection
    system_msg = build_system_message(workspace)
    assert "helpful agent" in system_msg

    # 4. Verify LiteLLM config generation
    litellm_config = generate_litellm_config(config)
    assert "workspace_fs" in litellm_config["mcp_servers"]
    assert "signal_channel" in litellm_config["mcp_servers"]

    # 5. Verify MCP tool declarations
    tools = build_mcp_tool_declarations(config)
    assert len(tools) == 2  # workspace_fs + signal_channel

    # 6. Create adapter and handle a message
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    adapter = SignalAdapter(
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
        if "chat/completions" in url:
            resp.json = MagicMock(return_value=mock_response)
            payload = kwargs.get("json", {})
            assert "tools" in payload
            assert len(payload["tools"]) == 2
        else:
            resp.json = MagicMock(return_value={})
        return resp

    adapter._http = AsyncMock()
    adapter._http.post = AsyncMock(side_effect=mock_post)

    result = await adapter.handle_message("+11111111111", "Hi there!")
    assert result == "I'm here to help!"

    # 7. Verify session persistence
    session = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    history = session.load("+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hi there!"
    assert history[1]["content"] == "I'm here to help!"

    # 8. Verify channel MCP tools exist
    mcp = create_channel_mcp()
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "stage_attachment" in tool_names
    assert "send_attachment" in tool_names
