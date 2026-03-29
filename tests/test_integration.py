"""Integration smoke test: verify the full event flow with mocked externals."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from agent_platform.config import load_config
from agent_platform.session import SessionManager
from session_manager.server import SessionOrchestrator
from adapters.signal.inbound import SignalInbound
from adapters.signal.outbound_mcp import SignalSender, create_signal_mcp


def _mock_proxy_response(content="I'm here to help!"):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={
        "choices": [{"message": {"role": "assistant", "content": content}}],
    })
    return resp


@pytest.mark.asyncio
async def test_full_event_flow(tmp_path):
    """End-to-end: config → orchestrator receives event → proxy called → session saved."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("# Agent\nUse tools to respond.\n")

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

    orchestrator = SessionOrchestrator(
        config=config,
        litellm_url="http://localhost:4000",
        session_dir=sessions_dir,
    )

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(return_value=_mock_proxy_response())

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hi there!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'm here to help!"

    # Verify MCP tool declarations were sent to proxy
    call_kwargs = orchestrator._http.post.call_args.kwargs
    payload = call_kwargs["json"]
    assert "tools" in payload
    tool_labels = [t["server_label"] for t in payload["tools"]]
    assert "workspace_fs" in tool_labels
    assert "memory" in tool_labels
    assert "signal" in tool_labels

    # Verify session persistence
    session = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    history = session.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hi there!"
    assert history[1]["content"] == "I'm here to help!"

    # Verify outbound MCP tools can be created
    sender = SignalSender(signal_cli_url="http://localhost:8080", account="+10000000000")
    mcp = create_signal_mcp(sender)
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "send_message" in tool_names

    # Verify inbound listener can be created
    inbound = SignalInbound(
        signal_cli_url="http://localhost:8080",
        session_manager_url="http://localhost:5000",
        account="+10000000000",
        allow_from=["+11111111111"],
    )
    assert inbound.account == "+10000000000"
