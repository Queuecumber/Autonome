"""Tests for Signal outbound MCP server."""

import pytest
from unittest.mock import AsyncMock

from adapters.signal.outbound_mcp import create_signal_mcp, SignalSender


@pytest.fixture
def mock_sender():
    sender = SignalSender(
        signal_cli_url="http://localhost:8080",
        account="+10000000000",
    )
    sender._http = AsyncMock()
    return sender


def test_signal_sender_init(mock_sender):
    assert mock_sender.account == "+10000000000"


@pytest.mark.asyncio
async def test_mcp_has_expected_tools():
    sender = SignalSender(
        signal_cli_url="http://localhost:8080",
        account="+10000000000",
    )
    mcp = create_signal_mcp(sender)
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "send_message" in tool_names
    assert "send_attachment" in tool_names
    assert "react" in tool_names
    assert "read_receipt" in tool_names
    assert "typing_indicator" in tool_names
