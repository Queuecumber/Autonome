"""Events delivered over MCP instead of HTTP.

Adapters push inbound events as log notifications tagged `autonome/event`.
`notifications/message` is the one standard server->client channel that is
fire-and-forget and carries arbitrary structured data — the right shape for
something we never want a reply to, and inert to any client that doesn't
know us.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from session_manager.mcp import EVENT_LOGGER, MCPConnection


# ── Receiving (session-manager) ──────────────────────────


def _params(logger_name, data, level="info"):
    p = MagicMock()
    p.logger = logger_name
    p.data = data
    p.level = level
    return p


@pytest.mark.asyncio
async def test_event_notification_reaches_the_handler():
    got = []
    conn = MCPConnection("matrix", "http://x", on_event=got.append)
    await conn._on_log(_params(EVENT_LOGGER, {"source": "matrix", "text": "hi"}))
    assert got == [{"source": "matrix", "text": "hi"}]


@pytest.mark.asyncio
async def test_ordinary_server_logs_are_not_events(caplog):
    """A server's real log output shares this channel — it must be surfaced,
    not silently swallowed or mistaken for an event."""
    got = []
    conn = MCPConnection("matrix", "http://x", on_event=got.append)
    with caplog.at_level(logging.WARNING):
        await conn._on_log(_params("matrix.client", "sync failed", level="warning"))
    assert got == []
    assert "sync failed" in caplog.text


@pytest.mark.asyncio
async def test_non_object_event_payload_is_dropped(caplog):
    got = []
    conn = MCPConnection("matrix", "http://x", on_event=got.append)
    with caplog.at_level(logging.WARNING):
        await conn._on_log(_params(EVENT_LOGGER, "just a string"))
    assert got == []
    assert "not an object" in caplog.text


@pytest.mark.asyncio
async def test_handler_failure_does_not_escape_into_the_read_loop(caplog):
    """An exception here would propagate into the session's receive loop and
    tear down the connection — losing every later event, not just this one."""
    def boom(_):
        raise ValueError("bad event")
    conn = MCPConnection("matrix", "http://x", on_event=boom)
    with caplog.at_level(logging.ERROR):
        await conn._on_log(_params(EVENT_LOGGER, {"source": "matrix"}))
    assert "dispatch failed" in caplog.text


@pytest.mark.asyncio
async def test_no_handler_configured_is_harmless():
    conn = MCPConnection("matrix", "http://x")
    await conn._on_log(_params(EVENT_LOGGER, {"source": "matrix"}))


def _orch(tmp_path, monkeypatch):
    from session_manager.orchestrator import SessionOrchestrator
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    return SessionOrchestrator(
        config={"model": {"name": "m"},
                "binaries": {"store": str(tmp_path / "b"), "retention_days": 1}},
        session_dir=tmp_path)


@pytest.mark.asyncio
async def test_malformed_event_does_not_start_a_turn(tmp_path, monkeypatch, caplog):
    """Adapters can push anything, and a notification has no reply to carry
    an error back — so a bad payload is dropped rather than raised."""
    orch = _orch(tmp_path, monkeypatch)
    handle = AsyncMock()
    monkeypatch.setattr(orch, "handle_event", handle)
    with caplog.at_level(logging.WARNING):
        orch._dispatch_pushed_event({"source": "matrix", "energy": "very loud"})
    await asyncio.sleep(0)
    handle.assert_not_called()
    assert "malformed" in caplog.text


@pytest.mark.asyncio
async def test_valid_pushed_event_starts_a_turn(tmp_path, monkeypatch):
    orch = _orch(tmp_path, monkeypatch)
    handle = AsyncMock()
    monkeypatch.setattr(orch, "handle_event", handle)
    orch._dispatch_pushed_event({"source": "matrix", "text": "hi"})
    await asyncio.sleep(0)
    assert handle.call_args.args[0].text == "hi"


# ── Sending (adapters) ───────────────────────────────────


@pytest.mark.asyncio
async def test_channel_is_captured_at_initialize():
    """An inbound message has no request context of its own, and FastMCP
    exposes no session registry — initialize is the only hook that fires on
    every connection before any tool call."""
    from matrix_adapter import server as matrix_server

    matrix_server._session = None
    channel = matrix_server._EventChannel()
    session = MagicMock()
    ctx = MagicMock()
    ctx.fastmcp_context = MagicMock()
    ctx.fastmcp_context.session = session

    async def call_next(_):
        return "init-result"

    assert await channel.on_initialize(ctx, call_next) == "init-result"
    assert matrix_server._session is session


@pytest.mark.asyncio
async def test_reconnect_replaces_a_dead_session():
    from matrix_adapter import server as matrix_server

    first, second = MagicMock(), MagicMock()
    channel = matrix_server._EventChannel()

    async def call_next(_):
        return None

    for s in (first, second):
        ctx = MagicMock()
        ctx.fastmcp_context = MagicMock()
        ctx.fastmcp_context.session = s
        await channel.on_initialize(ctx, call_next)
    assert matrix_server._session is second


@pytest.mark.asyncio
async def test_event_is_tagged_so_it_is_distinguishable_from_a_log():
    from matrix_adapter import server as matrix_server

    session = MagicMock()
    session.send_log_message = AsyncMock()
    matrix_server._session = session
    matrix_server._event_queue = asyncio.Queue()
    await matrix_server._event_queue.put({"source": "matrix", "text": "hi"})

    task = asyncio.create_task(matrix_server._forward_events())
    await matrix_server._event_queue.join()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    kwargs = session.send_log_message.call_args.kwargs
    assert kwargs["logger"] == EVENT_LOGGER
    assert kwargs["data"] == {"source": "matrix", "text": "hi"}


@pytest.mark.asyncio
async def test_events_arriving_before_the_channel_is_up_are_held(monkeypatch):
    """The client's standing stream comes up a moment after initialize, and a
    push into that window is dropped silently rather than raising. The queue
    is what keeps a message that lands there from vanishing."""
    from matrix_adapter import server as matrix_server

    matrix_server._session = None
    matrix_server._event_queue = asyncio.Queue()
    # Collapse the backoff without removing the yield point — a no-op sleep
    # would let the retry loop starve everything else on the loop.
    real_sleep = asyncio.sleep
    async def quick(_):
        await real_sleep(0.001)
    monkeypatch.setattr(matrix_server.asyncio, "sleep", quick)
    await matrix_server._event_queue.put({"source": "matrix", "text": "early"})

    task = asyncio.create_task(matrix_server._forward_events())
    await real_sleep(0.05)          # spin against no channel for a while

    session = MagicMock()
    session.send_log_message = AsyncMock()
    matrix_server._session = session
    await asyncio.wait_for(matrix_server._event_queue.join(), timeout=5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # It arrives only because it was held: had the push been attempted and
    # dropped during the gap, attaching the channel would deliver nothing.
    assert session.send_log_message.call_args.kwargs["data"]["text"] == "early"
