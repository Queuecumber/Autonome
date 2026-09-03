"""Tests for Signal interface — inbound event forwarding."""

import pytest
from unittest.mock import AsyncMock

from signal_adapter.model import SignalClient, Message, Reaction, Attachment
from signal_adapter import server as signal_mcp


def _pushed(mod):
    """The event payload from the last notification the adapter pushed."""
    return mod._session.send_log_message.call_args.kwargs["data"]


@pytest.fixture
def setup_signal():
    """Initialize the signal mcp_server module with a mock client."""
    c = SignalClient(
        signal_cli_url="http://localhost:8080",
        account="+10000000000",
        allow_from=["+11111111111"],
    )
    c._http = AsyncMock()
    signal_mcp.client = c
    # Events ride out as MCP notifications on the connected client session.
    signal_mcp._session = AsyncMock()
    signal_mcp._session.send_log_message = AsyncMock()
    return signal_mcp


def test_interface_init(setup_signal):
    assert setup_signal.client.account == "+10000000000"


@pytest.mark.asyncio
async def test_on_message_pushes_event(setup_signal):
    msg = Message(sender="+11111111111", timestamp=1234567890, text="Hello from Signal")

    await setup_signal.on_message(msg)

    setup_signal._session.send_log_message.assert_called_once()
    assert (setup_signal._session.send_log_message.call_args.kwargs["logger"]
            == signal_mcp.EVENT_LOGGER)
    event = _pushed(setup_signal)
    assert event["source"] == "signal"
    # Adapters no longer dictate session_id — events route to the default.
    assert "session_id" not in event
    assert event["text"] == "Hello from Signal"


@pytest.mark.asyncio
async def test_on_message_reaction(setup_signal):
    import json
    reaction = Reaction(
        sender="+11111111111",
        emoji="👍",
        target_author="+10000000000",
        target_timestamp=456,
    )

    await setup_signal.on_message(reaction)

    event = _pushed(setup_signal)
    # Reactions are tagged event_type="reaction"; the structured payload
    # (type/emoji/target) is JSON-encoded into text.
    assert event["event_type"] == "reaction"
    payload = json.loads(event["text"])
    assert payload["type"] == "reaction"
    assert payload["emoji"] == "👍"
    assert payload["target"] == "456"


@pytest.mark.asyncio
async def test_on_message_attachment(setup_signal):
    msg = Message(
        sender="+11111111111",
        timestamp=123,
        text="Check this",
        attachments=[Attachment(id="abc", content_type="application/pdf", filename="doc.pdf")],
    )

    await setup_signal.on_message(msg)

    event = _pushed(setup_signal)
    assert event["text"] == "Check this"
    assert len(event["metadata"]["attachments"]) == 1
    assert event["metadata"]["attachments"][0]["filename"] == "doc.pdf"
    assert event["metadata"]["attachments"][0]["url"] == "signal:///attachment/abc"
