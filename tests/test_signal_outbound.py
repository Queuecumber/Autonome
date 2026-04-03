"""Tests for Signal MCP server and client model."""

import pytest
from unittest.mock import AsyncMock

from signal_adapter.model import SignalClient, Message, Attachment, Reaction
from signal_adapter import mcp_server as signal_mcp


@pytest.fixture
def mock_client():
    client = SignalClient(
        signal_cli_url="http://localhost:8080",
        account="+10000000000",
    )
    client._http = AsyncMock()
    return client


def test_client_init(mock_client):
    assert mock_client.account == "+10000000000"
    assert mock_client.ws_url == "ws://localhost:8080/v1/receive/+10000000000"


def test_message_dataclass():
    msg = Message(sender="+1234", timestamp=123, text="hello")
    assert msg.sender == "+1234"
    assert msg.text == "hello"
    assert msg.attachments == []


def test_parse_text_message(mock_client):
    envelope = {
        "source": "+11111111111",
        "dataMessage": {"message": "Hello", "timestamp": 123},
    }
    msg = mock_client._parse_message(envelope)
    assert msg is not None
    assert msg.text == "Hello"
    assert msg.sender == "+11111111111"


def test_parse_reaction(mock_client):
    envelope = {
        "source": "+11111111111",
        "dataMessage": {
            "timestamp": 123,
            "reaction": {
                "emoji": "👍",
                "targetSentTimestamp": 456,
                "targetAuthor": "+10000000000",
                "isRemove": False,
            },
        },
    }
    result = mock_client._parse_message(envelope)
    assert isinstance(result, Reaction)
    assert result.emoji == "👍"
    assert result.target_timestamp == 456
    assert result.sender == "+11111111111"


def test_parse_attachment(mock_client):
    envelope = {
        "source": "+11111111111",
        "dataMessage": {
            "message": "look",
            "timestamp": 123,
            "attachments": [
                {"id": "abc", "contentType": "image/png", "fileName": "photo.png"},
            ],
        },
    }
    msg = mock_client._parse_message(envelope)
    assert msg is not None
    assert len(msg.attachments) == 1
    assert msg.attachments[0].content_type == "image/png"


def test_parse_filters_unauthorized(mock_client):
    mock_client.allow_from = ["+11111111111"]
    envelope = {
        "source": "+19999999999",
        "dataMessage": {"message": "nope", "timestamp": 123},
    }
    assert mock_client._parse_message(envelope) is None


@pytest.mark.asyncio
async def test_mcp_has_expected_tools():
    client = SignalClient(signal_cli_url="http://localhost:8080", account="+10000000000")
    signal_mcp.client = client
    signal_mcp.session_manager_url = "http://localhost:5000"
    tools = await signal_mcp.mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "send_message" in tool_names
    assert "send_attachment" in tool_names
    assert "react" in tool_names
    assert "read_receipt" in tool_names
    assert "typing_indicator" in tool_names
