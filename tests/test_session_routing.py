"""Tests for the session-routing contract.

Adapters don't dictate session_id — events without one route to the
default session ("main"). A schedule (or other source) can opt into a
specific session by setting session_id explicitly.
"""

import pytest

from session_manager.event import DEFAULT_SESSION_ID, Event


# ── Event.from_dict ──────────────────────────────────────


def test_event_default_session_id_when_omitted():
    e = Event.from_dict({"text": "hi"})
    assert e.session_id == DEFAULT_SESSION_ID


def test_event_default_session_id_when_empty():
    e = Event.from_dict({"session_id": "", "text": "hi"})
    assert e.session_id == DEFAULT_SESSION_ID


def test_event_default_session_id_when_null():
    e = Event.from_dict({"session_id": None, "text": "hi"})
    assert e.session_id == DEFAULT_SESSION_ID


def test_event_explicit_session_id_passes_through():
    e = Event.from_dict({"session_id": "custom-session", "text": "hi"})
    assert e.session_id == "custom-session"


def test_event_constructor_default_session_id():
    e = Event()
    assert e.session_id == DEFAULT_SESSION_ID


def test_default_session_id_is_main():
    """Sentinel: changing the default name is a behavior change.

    External integrations and persisted schedules may reference 'main';
    bump this constant deliberately and update accordingly.
    """
    assert DEFAULT_SESSION_ID == "main"


# ── Adapter to_event() shape ─────────────────────────────


def test_matrix_message_to_event_omits_session_id():
    from matrix_adapter.model import Message, Room, Sender

    msg = Message(
        sender=Sender(id="@alice:matrix.example", name="alice"),
        room=Room(id="!room:matrix.example", display_name="alice"),
        event_id="$evt",
        text="hello",
    )
    event = msg.to_event()
    assert "session_id" not in event


def test_matrix_reaction_to_event_omits_session_id():
    from matrix_adapter.model import Reaction, Room, Sender

    reaction = Reaction(
        sender=Sender(id="@alice:matrix.example", name="alice"),
        room=Room(id="!room:matrix.example", display_name="alice"),
        event_id="$evt",
        emoji="👍",
        target_event_id="$target",
    )
    event = reaction.to_event()
    assert "session_id" not in event


def test_signal_message_to_event_omits_session_id():
    from signal_adapter.model import Message

    msg = Message(sender="+15551234567", timestamp=123, text="hi")
    event = msg.to_event()
    assert "session_id" not in event


def test_signal_reaction_to_event_omits_session_id():
    from signal_adapter.model import Reaction

    reaction = Reaction(
        sender="+15551234567",
        emoji="👍",
        target_author="+10000000000",
        target_timestamp=456,
    )
    event = reaction.to_event()
    assert "session_id" not in event


# ── time-mcp _fire payload shape ─────────────────────────


@pytest.mark.asyncio
async def test_time_fire_omits_session_id_when_unset(monkeypatch):
    from time_mcp import server as time_server
    from time_mcp.server import Schedule

    posted: dict = {}

    class _Resp:
        def raise_for_status(self): pass

    class _Http:
        async def post(self, url, json):
            posted["url"] = url
            posted["json"] = json
            return _Resp()

    monkeypatch.setattr(time_server, "_http", _Http(), raising=False)
    monkeypatch.setattr(time_server, "session_manager_url", "http://test", raising=False)

    sched = Schedule(id="continuity", cron="*/20 * * * *", message="✨")
    await time_server._fire(sched)

    assert "session_id" not in posted["json"]
    assert posted["json"]["event_type"] == "continuity"


@pytest.mark.asyncio
async def test_time_fire_includes_session_id_when_set(monkeypatch):
    from time_mcp import server as time_server
    from time_mcp.server import Schedule

    posted: dict = {}

    class _Resp:
        def raise_for_status(self): pass

    class _Http:
        async def post(self, url, json):
            posted["json"] = json
            return _Resp()

    monkeypatch.setattr(time_server, "_http", _Http(), raising=False)
    monkeypatch.setattr(time_server, "session_manager_url", "http://test", raising=False)

    sched = Schedule(
        id="custom-cron", cron="0 9 * * *", message="hi",
        session_id="other-session",
    )
    await time_server._fire(sched)

    assert posted["json"]["session_id"] == "other-session"
    assert posted["json"]["event_type"] == "cron"
