"""Tests for Signal interface — inbound event forwarding."""

import pytest
from unittest.mock import AsyncMock

from adapters.signal.model import SignalClient, Message, Reaction, Attachment
from adapters.signal import mcp_server


@pytest.fixture
def client():
    c = SignalClient(
        signal_cli_url="http://localhost:8080",
        account="+10000000000",
        allow_from=["+11111111111"],
    )
    c._http = AsyncMock()
    return c


@pytest.fixture
def interface(client):
    mcp_server.init(client, "http://localhost:5000")
    mcp_server._http = AsyncMock()
    mcp_server._http.post = AsyncMock()
    return mcp_server


def test_interface_init(interface):
    assert interface._client.account == "+10000000000"


@pytest.mark.asyncio
async def test_on_message_pushes_event(interface):
    msg = Message(sender="+11111111111", timestamp=1234567890, text="Hello from Signal")

    await interface.on_message(msg)

    interface._http.post.assert_called_once()
    event = interface._http.post.call_args.kwargs["json"]
    assert event["source"] == "signal"
    assert event["session_id"] == "+11111111111"
    assert event["text"] == "Hello from Signal"


@pytest.mark.asyncio
async def test_on_message_reaction(interface):
    msg = Message(
        sender="+11111111111",
        timestamp=123,
        reaction=Reaction(
            emoji="👍",
            sender="+11111111111",
            target_author="+10000000000",
            target_timestamp=456,
        ),
    )

    await interface.on_message(msg)

    event = interface._http.post.call_args.kwargs["json"]
    assert event["metadata"]["type"] == "reaction"
    assert event["metadata"]["emoji"] == "👍"


@pytest.mark.asyncio
async def test_on_message_attachment(interface):
    msg = Message(
        sender="+11111111111",
        timestamp=123,
        text="Check this",
        attachments=[Attachment(id="abc", content_type="application/pdf", filename="doc.pdf")],
    )

    await interface.on_message(msg)

    event = interface._http.post.call_args.kwargs["json"]
    assert "doc.pdf" in event["text"]
