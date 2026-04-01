"""Tests for Signal inbound listener."""

import pytest
from unittest.mock import AsyncMock

from adapters.signal.model import SignalClient, Message, Reaction, Attachment
from adapters.signal.inbound import SignalInbound


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
def inbound(client):
    ib = SignalInbound(client=client, session_manager_url="http://localhost:5000")
    ib._http = AsyncMock()
    ib._http.post = AsyncMock()
    return ib


def test_inbound_init(inbound):
    assert inbound.client.account == "+10000000000"


@pytest.mark.asyncio
async def test_on_message_pushes_event(inbound):
    msg = Message(sender="+11111111111", timestamp=1234567890, text="Hello from Signal")

    await inbound._on_message(msg)

    inbound._http.post.assert_called_once()
    event = inbound._http.post.call_args.kwargs["json"]
    assert event["source"] == "signal"
    assert event["session_id"] == "+11111111111"
    assert event["text"] == "Hello from Signal"


@pytest.mark.asyncio
async def test_on_message_reaction(inbound):
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

    await inbound._on_message(msg)

    event = inbound._http.post.call_args.kwargs["json"]
    assert event["metadata"]["type"] == "reaction"
    assert event["metadata"]["emoji"] == "👍"


@pytest.mark.asyncio
async def test_on_message_attachment(inbound):
    msg = Message(
        sender="+11111111111",
        timestamp=123,
        text="Check this",
        attachments=[Attachment(id="abc", content_type="application/pdf", filename="doc.pdf")],
    )

    await inbound._on_message(msg)

    event = inbound._http.post.call_args.kwargs["json"]
    assert "doc.pdf" in event["text"]
