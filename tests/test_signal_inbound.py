"""Tests for Signal inbound listener."""

import pytest
from unittest.mock import AsyncMock

from adapters.signal.inbound import SignalInbound


@pytest.fixture
def inbound():
    return SignalInbound(
        signal_cli_url="http://localhost:8080",
        session_manager_url="http://localhost:5000",
        account="+10000000000",
        allow_from=["+11111111111"],
    )


def test_inbound_init(inbound):
    assert inbound.account == "+10000000000"
    assert inbound.allow_from == ["+11111111111"]


def test_ws_url_construction(inbound):
    assert inbound.ws_url == "ws://localhost:8080/v1/receive/+10000000000"


@pytest.mark.asyncio
async def test_handle_envelope_pushes_event(inbound):
    inbound._http = AsyncMock()
    inbound._http.post = AsyncMock()

    envelope = {
        "envelope": {
            "source": "+11111111111",
            "dataMessage": {
                "message": "Hello from Signal",
                "timestamp": 1234567890,
            },
        }
    }

    await inbound._handle_envelope(envelope)

    inbound._http.post.assert_called_once()
    call_args = inbound._http.post.call_args
    assert "/event" in call_args.args[0]
    event = call_args.kwargs["json"]
    assert event["source"] == "signal"
    assert event["session_id"] == "+11111111111"
    assert event["text"] == "Hello from Signal"


@pytest.mark.asyncio
async def test_handle_envelope_filters_unauthorized(inbound):
    inbound._http = AsyncMock()
    inbound._http.post = AsyncMock()

    envelope = {
        "envelope": {
            "source": "+19999999999",
            "dataMessage": {
                "message": "I'm not allowed",
                "timestamp": 1234567890,
            },
        }
    }

    await inbound._handle_envelope(envelope)
    inbound._http.post.assert_not_called()


@pytest.mark.asyncio
async def test_handle_envelope_skips_empty(inbound):
    inbound._http = AsyncMock()
    inbound._http.post = AsyncMock()

    await inbound._handle_envelope({"envelope": {"source": "+11111111111"}})
    await inbound._handle_envelope({"envelope": {"source": "+11111111111", "dataMessage": {}}})
    inbound._http.post.assert_not_called()


@pytest.mark.asyncio
async def test_handle_reaction(inbound):
    """Reactions are pushed as events with reaction metadata."""
    inbound._http = AsyncMock()
    inbound._http.post = AsyncMock()

    envelope = {
        "envelope": {
            "source": "+11111111111",
            "dataMessage": {
                "timestamp": 1234567890,
                "reaction": {
                    "emoji": "👍",
                    "targetSentTimestamp": 9876543210,
                    "targetAuthor": "+10000000000",
                    "isRemove": False,
                },
            },
        }
    }

    await inbound._handle_envelope(envelope)

    inbound._http.post.assert_called_once()
    event = inbound._http.post.call_args.kwargs["json"]
    assert event["metadata"]["type"] == "reaction"
    assert event["metadata"]["emoji"] == "👍"
    assert "👍" in event["text"]


@pytest.mark.asyncio
async def test_handle_attachment_metadata(inbound):
    """Attachments include metadata in the event."""
    inbound._http = AsyncMock()
    inbound._http.post = AsyncMock()

    envelope = {
        "envelope": {
            "source": "+11111111111",
            "dataMessage": {
                "message": "Check this out",
                "timestamp": 1234567890,
                "attachments": [
                    {
                        "contentType": "application/pdf",
                        "id": "abc123",
                        "fileName": "document.pdf",
                    }
                ],
            },
        }
    }

    await inbound._handle_envelope(envelope)

    inbound._http.post.assert_called_once()
    event = inbound._http.post.call_args.kwargs["json"]
    assert "document.pdf" in event["text"]
