"""Integration smoke test: verify the full event flow with mocked externals."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from session_manager.session import SessionManager
from session_manager.server import SessionOrchestrator
from signal_adapter.model import SignalClient
from signal_adapter import mcp_server as signal_mcp


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
async def test_full_event_flow(tmp_path):
    """End-to-end: config → orchestrator receives event → LLM called → session saved."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("# Agent\nUse tools to respond.\n")

    config_data = {
        "model": {"model": "claude-opus-4-6", "api_key": "test-key", "api_base": "http://localhost:4000/v1"},
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

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    orchestrator = SessionOrchestrator(config=config_data, session_dir=sessions_dir)
    orchestrator.llm = MagicMock()
    orchestrator.llm.chat = MagicMock()
    orchestrator.llm.chat.completions = MagicMock()
    orchestrator.llm.chat.completions.create = AsyncMock(return_value=_mock_llm_response())

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hi there!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'm here to help!"

    # Verify system prompt is AGENTS.md only
    call_kwargs = orchestrator.llm.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]
    assert "Use tools to respond" in system_msg["content"]

    # Verify session persistence
    session = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    history = session.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hi there!"
    assert history[1]["content"] == "I'm here to help!"

    # Verify signal MCP tools can be created
    signal_client = SignalClient(signal_cli_url="http://localhost:8080", account="+10000000000")
    signal_mcp.client = signal_client
    signal_mcp.session_manager_url = "http://localhost:5000"
    tools = await signal_mcp.mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "send_message" in tool_names

    assert signal_mcp.client.account == "+10000000000"
