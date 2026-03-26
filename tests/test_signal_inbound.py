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
    """WebSocket URL is derived from HTTP URL."""
    assert inbound.ws_url == "ws://localhost:8080/v1/receive/+10000000000"


@pytest.mark.asyncio
async def test_handle_envelope_pushes_event(inbound):
    """Valid envelopes are pushed to session manager."""
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

    # Should have: read receipt POST + event POST (and typing PUT calls)
    post_calls = inbound._http.post.call_args_list
    event_calls = [c for c in post_calls if "/event" in c.args[0]]
    receipt_calls = [c for c in post_calls if "/receipts" in c.args[0]]

    assert len(event_calls) == 1
    event = event_calls[0].kwargs["json"]
    assert event["source"] == "signal"
    assert event["session_id"] == "+11111111111"
    assert event["text"] == "Hello from Signal"

    assert len(receipt_calls) == 1

    # Typing indicator: start + stop
    put_calls = inbound._http.put.call_args_list
    assert len(put_calls) == 2  # start + stop


@pytest.mark.asyncio
async def test_handle_envelope_filters_unauthorized(inbound):
    """Messages from unauthorized senders are not pushed."""
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
    """Envelopes without dataMessage or text are skipped."""
    inbound._http = AsyncMock()
    inbound._http.post = AsyncMock()

    await inbound._handle_envelope({"envelope": {"source": "+11111111111"}})
    await inbound._handle_envelope({"envelope": {"source": "+11111111111", "dataMessage": {}}})
    inbound._http.post.assert_not_called()
