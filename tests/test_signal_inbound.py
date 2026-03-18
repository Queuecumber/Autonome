"""Tests for Signal inbound listener."""

import pytest
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_poll_pushes_event_to_session_manager(inbound):
    signal_response = [
        {
            "envelope": {
                "source": "+11111111111",
                "dataMessage": {
                    "message": "Hello from Signal",
                    "timestamp": 1234567890,
                },
            }
        }
    ]

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=signal_response)
        return resp

    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"status": "ok"})
        return resp

    inbound._http = AsyncMock()
    inbound._http.get = AsyncMock(side_effect=mock_get)
    inbound._http.post = AsyncMock(side_effect=mock_post)

    await inbound.poll_messages()

    inbound._http.post.assert_called_once()
    call_args = inbound._http.post.call_args
    assert "/event" in call_args.args[0]
    event = call_args.kwargs["json"]
    assert event["source"] == "signal"
    assert event["session_id"] == "+11111111111"
    assert event["text"] == "Hello from Signal"


@pytest.mark.asyncio
async def test_poll_filters_unauthorized(inbound):
    signal_response = [
        {
            "envelope": {
                "source": "+19999999999",
                "dataMessage": {
                    "message": "I'm not allowed",
                    "timestamp": 1234567890,
                },
            }
        }
    ]

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=signal_response)
        return resp

    inbound._http = AsyncMock()
    inbound._http.get = AsyncMock(side_effect=mock_get)
    inbound._http.post = AsyncMock()

    await inbound.poll_messages()
    inbound._http.post.assert_not_called()


@pytest.mark.asyncio
async def test_poll_skips_empty_messages(inbound):
    signal_response = [
        {"envelope": {"source": "+11111111111"}},
        {"envelope": {"source": "+11111111111", "dataMessage": {}}},
    ]

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=signal_response)
        return resp

    inbound._http = AsyncMock()
    inbound._http.get = AsyncMock(side_effect=mock_get)
    inbound._http.post = AsyncMock()

    await inbound.poll_messages()
    inbound._http.post.assert_not_called()
