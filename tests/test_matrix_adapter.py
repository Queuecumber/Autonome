"""Tests for the matrix-adapter capability + reliability surface.

Covers the pin/reaction/redaction/message build pipeline, outbound tools
(pin/unpin/edit/redact/send_message with relations, get_message, get_user_profile),
event-handler dispatch (sender filter, allowed_rooms, state-event bypass),
the event queue + retry on session-manager unreachable, and the small
pure helpers (_format_ts, _sender_dict, _room_dict, _parse_relation,
_compute_trust_status).

MatrixClient is constructed via __new__ everywhere so we don't actually
spin up nio's AsyncClient/store.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Pure helpers ─────────────────────────────────────────


def test_format_ts_none_returns_none():
    from matrix_adapter.model import _format_ts
    assert _format_ts(None) is None


def test_format_ts_renders_local_time():
    from matrix_adapter.model import _format_ts
    out = _format_ts(0)  # epoch
    assert out is not None
    assert "1969" in out or "1970" in out  # local-zone dependent


def test_sender_dict_shape():
    from matrix_adapter.model import Sender, _sender_dict
    s = Sender(id="@a:b", name="Alice", avatar_url="mxc://srv/x")
    assert _sender_dict(s) == {"id": "@a:b", "name": "Alice", "avatar": "mxc://srv/x"}


def test_room_dict_shape():
    from matrix_adapter.model import Room, _room_dict
    r = Room(id="!r:s", display_name="Chat", encrypted=True, member_count=3, pinned_event_ids=["$p"])
    out = _room_dict(r)
    assert out["id"] == "!r:s"
    assert out["name"] == "Chat"
    assert out["encrypted"] is True
    assert out["member_count"] == 3
    assert out["pinned_event_ids"] == ["$p"]


def test_parse_relation_thread():
    from matrix_adapter.model import _parse_relation
    rel = _parse_relation({"rel_type": "m.thread", "event_id": "$root"})
    assert rel is not None
    assert rel.relation_type == "m.thread"
    assert rel.related_event_id == "$root"


def test_parse_relation_replace():
    from matrix_adapter.model import _parse_relation
    rel = _parse_relation({"rel_type": "m.replace", "event_id": "$orig"})
    assert rel.relation_type == "m.replace"


def test_parse_relation_in_reply_to():
    from matrix_adapter.model import _parse_relation
    rel = _parse_relation({"m.in_reply_to": {"event_id": "$prev"}})
    assert rel is not None
    assert rel.relation_type == "m.in_reply_to"
    assert rel.related_event_id == "$prev"


def test_parse_relation_none():
    from matrix_adapter.model import _parse_relation
    assert _parse_relation({}) is None


def test_room_name_falls_back():
    from matrix_adapter.model import Room
    assert Room(id="!r:s").name == "!r:s"
    assert Room(id="!r:s", canonical_alias="#chat:s").name == "#chat:s"
    assert Room(id="!r:s", display_name="Chat", canonical_alias="#chat:s").name == "Chat"


def test_sender_name_defaults_to_id():
    from matrix_adapter.model import Sender
    s = Sender(id="@a:b")
    assert s.name == "@a:b"


# ── Dataclass to_event projections ───────────────────────


@pytest.fixture
def alice():
    from matrix_adapter.model import Sender
    return Sender(id="@alice:srv", name="Alice", avatar_url="mxc://srv/av")


@pytest.fixture
def room():
    from matrix_adapter.model import Room
    return Room(id="!r:s", display_name="Chat", encrypted=True, member_count=2)


def test_message_to_event_text(alice, room):
    from matrix_adapter.model import Message
    msg = Message(sender=alice, room=room, event_id="$e", text="hi",
                  sent_at="2026-05-15 12:00:00 EDT (Friday)", verified=True)
    out = msg.to_event()
    assert out["source"] == "matrix"
    assert out["text"] == "hi"
    md = out["metadata"]
    assert md["message_id"] == "$e"
    assert md["sender"]["avatar"] == "mxc://srv/av"
    assert md["room"]["encrypted"] is True
    assert md["sent_at"].startswith("2026-")
    assert md["verified"] is True


def test_message_to_event_with_attachment_and_relation(alice, room):
    from matrix_adapter.model import Attachment, Message, MessageRelation
    att = Attachment(url="mxc://srv/x", content_type="image/png", filename="a.png", caption="cap")
    rel = MessageRelation(related_event_id="$root", relation_type="m.thread")
    msg = Message(sender=alice, room=room, event_id="$e", text=None,
                  attachments=[att], relation=rel)
    md = msg.to_event()["metadata"]
    assert md["attachments"][0]["filename"] == "a.png"
    assert md["relation"] == {"related_event_id": "$root", "relation_type": "m.thread"}


def test_reaction_to_event(alice, room):
    from matrix_adapter.model import Reaction
    r = Reaction(sender=alice, room=room, event_id="$r", emoji="👍", target_event_id="$t")
    out = r.to_event()
    assert out["event_type"] == "reaction"
    assert out["energy"] == "passive"
    payload = json.loads(out["text"])
    assert payload == {"type": "reaction", "emoji": "👍", "target_event_id": "$t"}


def test_redaction_to_event(alice, room):
    from matrix_adapter.model import Redaction
    r = Redaction(sender=alice, room=room, target_event_id="$t", reason="spam")
    out = r.to_event()
    assert out["event_type"] == "redaction"
    assert out["energy"] == "passive"
    payload = json.loads(out["text"])
    assert payload["target_event_id"] == "$t"
    assert payload["reason"] == "spam"


def test_pin_to_event_pin_kind(alice, room):
    from matrix_adapter.model import Pin
    p = Pin(sender=alice, room=room, target_event_id="$t", pinned=True)
    out = p.to_event()
    assert out["event_type"] == "pin"
    assert out["energy"] == "passive"
    assert json.loads(out["text"])["type"] == "pin"


def test_pin_to_event_unpin_kind(alice, room):
    from matrix_adapter.model import Pin
    p = Pin(sender=alice, room=room, target_event_id="$t", pinned=False)
    assert p.to_event()["event_type"] == "unpin"


# ── Build helpers ────────────────────────────────────────


def _new_client():
    from matrix_adapter.model import MatrixClient
    c = MatrixClient.__new__(MatrixClient)
    c.user_id = "@me:srv"
    c.allowed_rooms = None
    c._known_pins = {}
    c._synced_rooms = set()
    c._on_message = None
    return c


def _mock_nio_room(room_id: str = "!r:s"):
    room = MagicMock()
    room.room_id = room_id
    room.display_name = "Chat"
    room.canonical_alias = None
    room.encrypted = True
    room.users = {"@alice:srv": MagicMock(), "@me:srv": MagicMock()}
    room.user_name = MagicMock(side_effect=lambda uid: {"@alice:srv": "Alice"}.get(uid, uid))
    room.avatar_url = MagicMock(side_effect=lambda uid: f"mxc://srv/{uid[1:5]}")
    return room


def test_build_text_populates_all_fields():
    client = _new_client()
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.event_id = "$e"
    event.body = "hi"
    event.source = {"content": {"m.relates_to": {"rel_type": "m.thread", "event_id": "$root"}}}
    event.server_timestamp = 1715000000000
    event.verified = True

    msg = client._build_text(room, event)
    assert msg.sender.id == "@alice:srv"
    assert msg.sender.name == "Alice"
    assert msg.text == "hi"
    assert msg.relation.relation_type == "m.thread"
    assert msg.verified is True
    assert msg.sent_at is not None


def test_build_reaction():
    client = _new_client()
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.event_id = "$r"
    event.source = {"content": {"m.relates_to": {"key": "👍", "event_id": "$t"}}}
    event.server_timestamp = 0
    event.verified = False

    r = client._build_reaction(room, event)
    assert r.emoji == "👍"
    assert r.target_event_id == "$t"


def test_build_redaction():
    client = _new_client()
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.redacts = "$t"
    event.reason = "oops"
    event.server_timestamp = 0
    event.verified = False

    r = client._build_redaction(room, event)
    assert r.target_event_id == "$t"
    assert r.reason == "oops"


def test_build_media_unencrypted_inlines_mime():
    client = _new_client()
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.event_id = "$e"
    event.url = "mxc://srv/abc"
    event.body = "img.png"
    event.source = {"content": {"info": {"mimetype": "image/png"}}}
    event.server_timestamp = 0
    event.verified = False

    msg = client._build_media(room, event)
    assert len(msg.attachments) == 1
    assert "mime=image%2Fpng" in msg.attachments[0].url


def test_extract_media_encrypted_inlines_keys():
    client = _new_client()
    event = MagicMock()
    event.url = "mxc://srv/enc"
    event.body = "f.bin"
    event.source = {"content": {
        "info": {"mimetype": "application/pdf"},
        "file": {
            "url": "mxc://srv/enc",
            "key": {"k": "KEY"},
            "iv": "IV",
            "hashes": {"sha256": "HASH"},
        },
    }}
    att = client._extract_media(event)
    assert "k=KEY" in att.url
    assert "iv=IV" in att.url
    assert "hash=HASH" in att.url
    assert "mime=application%2Fpdf" in att.url


# ── _handle_event dispatch ───────────────────────────────


@pytest.mark.asyncio
async def test_handle_event_drops_self_timeline():
    from matrix_adapter.model import RoomMessageText
    client = _new_client()
    room = _mock_nio_room()
    event = MagicMock(spec=RoomMessageText)
    event.sender = "@me:srv"
    # Register against the mock's actual type — dispatch uses type(event).
    client._handlers = {type(event): AsyncMock()}
    await client._handle_event(room, event)
    client._handlers[type(event)].assert_not_called()


@pytest.mark.asyncio
async def test_handle_event_allows_self_state_events():
    """UnknownEvents (state events) bypass the self-sender filter."""
    from matrix_adapter.model import UnknownEvent
    client = _new_client()
    room = _mock_nio_room()
    # A real UnknownEvent instance so isinstance(event, UnknownEvent) is True
    # and type(event) is UnknownEvent.
    event = UnknownEvent.__new__(UnknownEvent)
    event.sender = "@me:srv"
    event.type = "m.room.topic"
    handler = AsyncMock()
    client._handlers = {UnknownEvent: handler}
    await client._handle_event(room, event)
    handler.assert_awaited_once_with(room, event)


@pytest.mark.asyncio
async def test_handle_event_respects_allowed_rooms():
    from matrix_adapter.model import RoomMessageText
    client = _new_client()
    client.allowed_rooms = ["!keep:s"]
    room = _mock_nio_room("!other:s")
    event = MagicMock(spec=RoomMessageText)
    event.sender = "@alice:srv"
    client._handlers = {type(event): AsyncMock()}
    await client._handle_event(room, event)
    client._handlers[type(event)].assert_not_called()


# ── _on_pinned_events ────────────────────────────────────


@pytest.mark.asyncio
async def test_on_pinned_events_initial_seed_suppresses_emit():
    """First state event for a room during initial sync just seeds; no Pin emitted."""
    client = _new_client()
    client._on_message = AsyncMock()
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.source = {"content": {"pinned": ["$p1", "$p2"]}}

    await client._on_pinned_events(room, event)
    assert client._known_pins[room.room_id] == ["$p1", "$p2"]
    client._on_message.assert_not_called()


@pytest.mark.asyncio
async def test_on_pinned_events_diff_emits_after_sync():
    """Once the room is in _synced_rooms, delta emits Pin events."""
    client = _new_client()
    client._on_message = AsyncMock()
    client._synced_rooms = {"!r:s"}
    client._known_pins = {"!r:s": ["$old"]}
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.server_timestamp = 0
    event.verified = False
    event.source = {"content": {"pinned": ["$old", "$new"]}}

    await client._on_pinned_events(room, event)
    # One Pin emit for the added id.
    assert client._on_message.await_count == 1
    pin = client._on_message.await_args.args[0]
    assert pin.target_event_id == "$new"
    assert pin.pinned is True


@pytest.mark.asyncio
async def test_on_pinned_events_unpin_diff():
    client = _new_client()
    client._on_message = AsyncMock()
    client._synced_rooms = {"!r:s"}
    client._known_pins = {"!r:s": ["$a", "$b"]}
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.server_timestamp = 0
    event.verified = False
    event.source = {"content": {"pinned": ["$a"]}}

    await client._on_pinned_events(room, event)
    pin = client._on_message.await_args.args[0]
    assert pin.target_event_id == "$b"
    assert pin.pinned is False


@pytest.mark.asyncio
async def test_on_pinned_events_self_sender_tracks_no_emit():
    """Our own pin state changes update _known_pins but don't emit Pin events."""
    client = _new_client()
    client._on_message = AsyncMock()
    client._synced_rooms = {"!r:s"}
    client._known_pins = {"!r:s": []}
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@me:srv"
    event.source = {"content": {"pinned": ["$x"]}}

    await client._on_pinned_events(room, event)
    assert client._known_pins["!r:s"] == ["$x"]
    client._on_message.assert_not_called()


# ── Per-type handlers ────────────────────────────────────


@pytest.mark.asyncio
async def test_on_text_dispatches_to_on_message():
    client = _new_client()
    client._on_message = AsyncMock()
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.event_id = "$e"
    event.body = "hi"
    event.source = {"content": {}}
    event.server_timestamp = 0
    event.verified = False
    await client._on_text(room, event)
    client._on_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_reaction_dispatches():
    client = _new_client()
    client._on_message = AsyncMock()
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.event_id = "$r"
    event.source = {"content": {"m.relates_to": {"key": "👍", "event_id": "$t"}}}
    event.server_timestamp = 0
    event.verified = False
    await client._on_reaction(room, event)
    client._on_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_redaction_dispatches():
    client = _new_client()
    client._on_message = AsyncMock()
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.redacts = "$t"
    event.reason = None
    event.server_timestamp = 0
    event.verified = False
    await client._on_redaction(room, event)
    client._on_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_unknown_routes_pinned_events():
    client = _new_client()
    event = MagicMock()
    event.type = "m.room.pinned_events"
    event.sender = "@alice:srv"
    event.source = {"content": {"pinned": []}}
    room = _mock_nio_room()
    await client._on_unknown(room, event)
    # Seeded silently because room not in _synced_rooms.
    assert client._known_pins[room.room_id] == []


@pytest.mark.asyncio
async def test_on_unknown_ignores_other_types():
    client = _new_client()
    event = MagicMock()
    event.type = "m.room.topic"
    room = _mock_nio_room()
    # No exception, no state change.
    await client._on_unknown(room, event)
    assert room.room_id not in client._known_pins


# ── _compute_trust_status ────────────────────────────────


def _device(verified=False, blacklisted=False):
    d = MagicMock()
    d.verified = verified
    d.blacklisted = blacklisted
    return d


def test_trust_status_unknown_when_no_devices():
    client = _new_client()
    client._client = MagicMock()
    client._client.device_store.active_user_devices = MagicMock(return_value=iter([]))
    assert client._compute_trust_status("@a:b") == "unknown"


def test_trust_status_verified():
    client = _new_client()
    client._client = MagicMock()
    client._client.device_store.active_user_devices = MagicMock(
        return_value=iter([_device(verified=True), _device(verified=True)])
    )
    assert client._compute_trust_status("@a:b") == "verified"


def test_trust_status_partial():
    client = _new_client()
    client._client = MagicMock()
    client._client.device_store.active_user_devices = MagicMock(
        return_value=iter([_device(verified=True), _device(verified=False)])
    )
    assert client._compute_trust_status("@a:b") == "partial"


def test_trust_status_unverified():
    client = _new_client()
    client._client = MagicMock()
    client._client.device_store.active_user_devices = MagicMock(
        return_value=iter([_device(verified=False)])
    )
    assert client._compute_trust_status("@a:b") == "unverified"


def test_trust_status_blacklisted_wins():
    client = _new_client()
    client._client = MagicMock()
    client._client.device_store.active_user_devices = MagicMock(
        return_value=iter([_device(verified=True), _device(blacklisted=True)])
    )
    assert client._compute_trust_status("@a:b") == "blacklisted"


# ── get_user_profile ─────────────────────────────────────


@pytest.mark.asyncio
async def test_get_user_profile_basic():
    from matrix_adapter.model import ProfileGetResponse
    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock(spec=ProfileGetResponse)
    resp.displayname = "Alice"
    resp.avatar_url = "mxc://srv/av"
    client._client.profile_get = AsyncMock(return_value=resp)
    client._client.device_store.active_user_devices = MagicMock(return_value=iter([]))

    profile = await client.get_user_profile("@alice:srv")
    assert profile.display_name == "Alice"
    assert profile.avatar_url == "mxc://srv/av"
    assert profile.trust_status == "unknown"
    assert profile.power_level is None


@pytest.mark.asyncio
async def test_get_user_profile_includes_power_level():
    from matrix_adapter.model import ProfileGetResponse
    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock(spec=ProfileGetResponse)
    resp.displayname = "Alice"
    resp.avatar_url = None
    client._client.profile_get = AsyncMock(return_value=resp)
    client._client.device_store.active_user_devices = MagicMock(return_value=iter([]))

    room = _mock_nio_room()
    room.power_levels = MagicMock()
    room.power_levels.get_user_level = MagicMock(return_value=50)
    client._client.rooms = {"!r:s": room}

    profile = await client.get_user_profile("@alice:srv", room_id="!r:s")
    assert profile.power_level == 50


@pytest.mark.asyncio
async def test_get_user_profile_raises_on_error():
    client = _new_client()
    client._client = MagicMock()
    client._client.profile_get = AsyncMock(return_value=MagicMock())  # not a ProfileGetResponse
    with pytest.raises(RuntimeError, match="profile_get"):
        await client.get_user_profile("@alice:srv")


# ── get_message ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_message_text():
    from matrix_adapter.model import RoomMessageText
    client = _new_client()
    client._client = MagicMock()
    event = MagicMock(spec=RoomMessageText)
    event.sender = "@alice:srv"
    event.event_id = "$e"
    event.body = "hi"
    event.source = {"content": {}}
    event.server_timestamp = 0
    event.verified = True
    resp = MagicMock()
    resp.event = event
    client._client.room_get_event = AsyncMock(return_value=resp)
    client._client.rooms = {"!r:s": _mock_nio_room()}

    msg = await client.get_message("!r:s", "$e")
    assert msg.text == "hi"
    assert msg.verified is True


@pytest.mark.asyncio
async def test_get_message_megolm_raises():
    from matrix_adapter.model import MegolmEvent
    client = _new_client()
    client._client = MagicMock()
    event = MagicMock(spec=MegolmEvent)
    resp = MagicMock()
    resp.event = event
    client._client.room_get_event = AsyncMock(return_value=resp)
    with pytest.raises(RuntimeError, match="encrypted"):
        await client.get_message("!r:s", "$e")


@pytest.mark.asyncio
async def test_get_message_non_message_raises():
    from matrix_adapter.model import ReactionEvent
    client = _new_client()
    client._client = MagicMock()
    event = MagicMock(spec=ReactionEvent)
    resp = MagicMock()
    resp.event = event
    client._client.room_get_event = AsyncMock(return_value=resp)
    client._client.rooms = {"!r:s": _mock_nio_room()}
    with pytest.raises(RuntimeError, match="not a message"):
        await client.get_message("!r:s", "$e")


# ── pin_message / unpin_message ──────────────────────────


@pytest.mark.asyncio
async def test_pin_message_adds_to_state():
    from matrix_adapter.model import RoomGetStateEventResponse, RoomPutStateResponse
    client = _new_client()
    client._client = MagicMock()

    get_resp = MagicMock(spec=RoomGetStateEventResponse)
    get_resp.content = {"pinned": ["$existing"]}
    client._client.room_get_state_event = AsyncMock(return_value=get_resp)

    put_resp = MagicMock(spec=RoomPutStateResponse)
    client._client.room_put_state = AsyncMock(return_value=put_resp)

    await client.pin_message("!r:s", "$new")
    put_args = client._client.room_put_state.await_args
    assert put_args.args[2] == {"pinned": ["$existing", "$new"]}
    assert client._known_pins["!r:s"] == ["$existing", "$new"]


@pytest.mark.asyncio
async def test_unpin_message_removes_from_state():
    from matrix_adapter.model import RoomGetStateEventResponse, RoomPutStateResponse
    client = _new_client()
    client._client = MagicMock()

    get_resp = MagicMock(spec=RoomGetStateEventResponse)
    get_resp.content = {"pinned": ["$a", "$b"]}
    client._client.room_get_state_event = AsyncMock(return_value=get_resp)

    put_resp = MagicMock(spec=RoomPutStateResponse)
    client._client.room_put_state = AsyncMock(return_value=put_resp)

    await client.unpin_message("!r:s", "$a")
    put_args = client._client.room_put_state.await_args
    assert put_args.args[2] == {"pinned": ["$b"]}


@pytest.mark.asyncio
async def test_pin_message_noop_when_already_pinned():
    from matrix_adapter.model import RoomGetStateEventResponse
    client = _new_client()
    client._client = MagicMock()
    get_resp = MagicMock(spec=RoomGetStateEventResponse)
    get_resp.content = {"pinned": ["$x"]}
    client._client.room_get_state_event = AsyncMock(return_value=get_resp)
    client._client.room_put_state = AsyncMock()

    await client.pin_message("!r:s", "$x")
    client._client.room_put_state.assert_not_called()


@pytest.mark.asyncio
async def test_get_pinned_events_no_state_returns_empty():
    """When the room has no pinned_events state set yet (error response),
    treat as empty list."""
    client = _new_client()
    client._client = MagicMock()
    client._client.room_get_state_event = AsyncMock(return_value=MagicMock())  # not RoomGetStateEventResponse
    assert await client._get_pinned_events("!r:s") == []


# ── send_message relations ───────────────────────────────


@pytest.mark.asyncio
async def test_send_message_plain():
    from matrix_adapter.model import RoomSendResponse
    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock(spec=RoomSendResponse)
    resp.event_id = "$new"
    client._client.room_send = AsyncMock(return_value=resp)

    out = await client.send_message("!r:s", "hello")
    assert out == "$new"
    body = client._client.room_send.await_args.args[2]
    assert body["body"] == "hello"
    assert "m.relates_to" not in body


@pytest.mark.asyncio
async def test_send_message_thread_reply():
    from matrix_adapter.model import MessageRelation, RoomSendResponse
    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock(spec=RoomSendResponse)
    resp.event_id = "$new"
    client._client.room_send = AsyncMock(return_value=resp)

    rel = MessageRelation(related_event_id="$root", relation_type="m.thread")
    await client.send_message("!r:s", "reply", relation=rel)
    body = client._client.room_send.await_args.args[2]
    assert body["m.relates_to"] == {"rel_type": "m.thread", "event_id": "$root"}


@pytest.mark.asyncio
async def test_send_message_in_reply_to():
    from matrix_adapter.model import MessageRelation, RoomSendResponse
    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock(spec=RoomSendResponse)
    resp.event_id = "$new"
    client._client.room_send = AsyncMock(return_value=resp)

    rel = MessageRelation(related_event_id="$prev", relation_type="m.in_reply_to")
    await client.send_message("!r:s", "ack", relation=rel)
    body = client._client.room_send.await_args.args[2]
    assert body["m.relates_to"] == {"m.in_reply_to": {"event_id": "$prev"}}


@pytest.mark.asyncio
async def test_send_message_edit_replace():
    from matrix_adapter.model import MessageRelation, RoomSendResponse
    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock(spec=RoomSendResponse)
    resp.event_id = "$new"
    client._client.room_send = AsyncMock(return_value=resp)

    rel = MessageRelation(related_event_id="$orig", relation_type="m.replace")
    await client.send_message("!r:s", "fixed", relation=rel)
    body = client._client.room_send.await_args.args[2]
    assert body["m.relates_to"]["rel_type"] == "m.replace"
    assert body["m.new_content"]["body"] == "fixed"
    assert body["body"].startswith("* ")


@pytest.mark.asyncio
async def test_redact_message_calls_nio():
    from matrix_adapter.model import RoomRedactResponse
    client = _new_client()
    client._client = MagicMock()
    client._client.room_redact = AsyncMock(return_value=MagicMock(spec=RoomRedactResponse))
    await client.redact_message("!r:s", "$e", reason="bad")
    client._client.room_redact.assert_awaited_once_with("!r:s", "$e", reason="bad")


@pytest.mark.asyncio
async def test_send_reaction():
    from matrix_adapter.model import RoomSendResponse
    client = _new_client()
    client._client = MagicMock()
    client._client.room_send = AsyncMock(return_value=MagicMock(spec=RoomSendResponse))
    await client.send_reaction("!r:s", "$t", "👍")
    body = client._client.room_send.await_args.args[2]
    assert body["m.relates_to"]["key"] == "👍"


# ── server.py: event forwarding queue ────────────────────


@pytest.mark.asyncio
async def test_on_message_enqueues():
    from matrix_adapter import server as matrix_server

    matrix_server._event_queue = asyncio.Queue()
    msg = MagicMock()
    msg.to_event = MagicMock(return_value={"source": "matrix", "text": "hi"})
    await matrix_server.on_message(msg)
    assert matrix_server._event_queue.qsize() == 1
    assert matrix_server._event_queue.get_nowait() == {"source": "matrix", "text": "hi"}


# ── MatrixClient utility surface ─────────────────────────


@pytest.mark.asyncio
async def test_download_attachment_unencrypted():
    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock()
    resp.body = b"raw-bytes"
    client._client.download = AsyncMock(return_value=resp)

    out = await client.download_attachment("srv", "abc")
    assert out == b"raw-bytes"
    client._client.download.assert_awaited_once_with("mxc://srv/abc")


@pytest.mark.asyncio
async def test_download_attachment_encrypted_decrypts(monkeypatch):
    import matrix_adapter.model as model_mod
    monkeypatch.setattr(model_mod, "decrypt_attachment",
                        MagicMock(return_value=b"plaintext"))

    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock()
    resp.body = b"ciphertext"
    client._client.download = AsyncMock(return_value=resp)

    out = await client.download_attachment("srv", "abc", k="K", iv="I", hash="H")
    assert out == b"plaintext"
    model_mod.decrypt_attachment.assert_called_once_with(b"ciphertext", "K", "H", "I")


@pytest.mark.asyncio
async def test_send_typing_and_read_receipt():
    client = _new_client()
    client._client = MagicMock()
    client._client.room_typing = AsyncMock()
    client._client.room_read_markers = AsyncMock()
    await client.send_typing("!r:s", typing=True)
    client._client.room_typing.assert_awaited_once()
    await client.send_read_receipt("!r:s", "$e")
    client._client.room_read_markers.assert_awaited_once()


def test_get_room_members_returns_empty_for_unknown_room():
    client = _new_client()
    client._client = MagicMock()
    client._client.rooms = {}
    assert client.get_room_members("!nope:s") == []


def test_get_room_members_lists_users():
    client = _new_client()
    client._client = MagicMock()
    room = MagicMock()
    u1 = MagicMock()
    u1.display_name = "Alice"
    u2 = MagicMock()
    u2.display_name = "Bob"
    room.users = {"@alice:s": u1, "@bob:s": u2}
    client._client.rooms = {"!r:s": room}
    out = client.get_room_members("!r:s")
    names = {s.name for s in out}
    assert names == {"Alice", "Bob"}


@pytest.mark.asyncio
async def test_upload_and_send_attachment_unencrypted():
    from matrix_adapter.model import RoomSendResponse
    client = _new_client()
    client._client = MagicMock()
    room = MagicMock()
    room.encrypted = False
    client._client.rooms = {"!r:s": room}

    upload_resp = MagicMock()
    upload_resp.content_uri = "mxc://srv/new"
    client._client.upload = AsyncMock(return_value=(upload_resp, None))
    client._client.room_send = AsyncMock(return_value=MagicMock(spec=RoomSendResponse))

    await client.upload_and_send_attachment("!r:s", b"data", "image/png", "x.png")
    body = client._client.room_send.await_args.args[2]
    assert body["url"] == "mxc://srv/new"
    assert body["msgtype"] == "m.image"
    assert "file" not in body


@pytest.mark.asyncio
async def test_upload_and_send_attachment_encrypted_uses_file_field():
    from matrix_adapter.model import RoomSendResponse
    client = _new_client()
    client._client = MagicMock()
    room = MagicMock()
    room.encrypted = True
    client._client.rooms = {"!r:s": room}

    upload_resp = MagicMock()
    upload_resp.content_uri = "mxc://srv/new"
    client._client.upload = AsyncMock(return_value=(upload_resp, {"key": {"k": "K"}}))
    client._client.room_send = AsyncMock(return_value=MagicMock(spec=RoomSendResponse))

    await client.upload_and_send_attachment("!r:s", b"data", "image/png", "x.png", caption="hi")
    body = client._client.room_send.await_args.args[2]
    assert "url" not in body
    assert body["file"]["url"] == "mxc://srv/new"
    assert body["body"] == "hi"  # caption used when present


@pytest.mark.asyncio
async def test_set_display_name():
    client = _new_client()
    client._client = MagicMock()
    client._client.set_displayname = AsyncMock()
    await client.set_display_name("New Name")
    client._client.set_displayname.assert_awaited_once_with("New Name")


@pytest.mark.asyncio
async def test_upload_avatar(monkeypatch):
    client = _new_client()
    client._client = MagicMock()
    upload_resp = MagicMock()
    upload_resp.content_uri = "mxc://srv/avatar"
    client._client.upload = AsyncMock(return_value=(upload_resp, None))
    client._client.set_avatar = AsyncMock()
    await client.upload_avatar(b"jpgdata", "image/jpeg", "avatar.jpg")
    client._client.set_avatar.assert_awaited_once_with("mxc://srv/avatar")


@pytest.mark.asyncio
async def test_room_send_raises_on_error():
    client = _new_client()
    client._client = MagicMock()
    # Returns something that isn't RoomSendResponse → raise.
    client._client.room_send = AsyncMock(return_value=MagicMock())
    with pytest.raises(RuntimeError, match="room_send"):
        await client._room_send("!r:s", "m.room.message", {})


@pytest.mark.asyncio
async def test_pin_message_room_put_state_failure_raises():
    from matrix_adapter.model import RoomGetStateEventResponse
    client = _new_client()
    client._client = MagicMock()
    get_resp = MagicMock(spec=RoomGetStateEventResponse)
    get_resp.content = {"pinned": []}
    client._client.room_get_state_event = AsyncMock(return_value=get_resp)
    client._client.room_put_state = AsyncMock(return_value=MagicMock())  # not RoomPutStateResponse
    with pytest.raises(RuntimeError, match="room_put_state"):
        await client.pin_message("!r:s", "$t")


@pytest.mark.asyncio
async def test_redact_message_failure_raises():
    client = _new_client()
    client._client = MagicMock()
    client._client.room_redact = AsyncMock(return_value=MagicMock())  # not RoomRedactResponse
    with pytest.raises(RuntimeError, match="room_redact"):
        await client.redact_message("!r:s", "$e")


@pytest.mark.asyncio
async def test_get_message_room_get_event_failure_raises():
    client = _new_client()
    client._client = MagicMock()
    resp = MagicMock()
    resp.event = None  # no event attribute
    client._client.room_get_event = AsyncMock(return_value=resp)
    with pytest.raises(RuntimeError, match="room_get_event"):
        await client.get_message("!r:s", "$e")


@pytest.mark.asyncio
async def test_get_message_unknown_room_raises():
    from matrix_adapter.model import RoomMessageText
    client = _new_client()
    client._client = MagicMock()
    event = MagicMock(spec=RoomMessageText)
    resp = MagicMock()
    resp.event = event
    client._client.room_get_event = AsyncMock(return_value=resp)
    client._client.rooms = {}  # room not synced
    with pytest.raises(RuntimeError, match="not in client state"):
        await client.get_message("!r:s", "$e")


def test_extract_media_top_filename_differs_from_body():
    """When MSC2530 filename is set and differs from body, body is the caption."""
    client = _new_client()
    event = MagicMock()
    event.url = "mxc://srv/x"
    event.body = "look at this"
    event.source = {"content": {"info": {"mimetype": "image/png"}, "filename": "photo.png"}}
    att = client._extract_media(event)
    assert att.filename == "photo.png"
    assert att.caption == "look at this"


def test_extract_media_top_filename_same_as_body():
    """When top filename equals body, treat as legacy (no caption)."""
    client = _new_client()
    event = MagicMock()
    event.url = "mxc://srv/x"
    event.body = "photo.png"
    event.source = {"content": {"info": {"mimetype": "image/png", "filename": "photo.png"}}}
    att = client._extract_media(event)
    assert att.filename == "photo.png"
    assert att.caption is None


@pytest.mark.asyncio
async def test_on_pinned_events_noop_when_set_unchanged():
    client = _new_client()
    client._on_message = AsyncMock()
    client._synced_rooms = {"!r:s"}
    client._known_pins = {"!r:s": ["$a"]}
    room = _mock_nio_room()
    event = MagicMock()
    event.sender = "@alice:srv"
    event.source = {"content": {"pinned": ["$a"]}}  # same set
    await client._on_pinned_events(room, event)
    client._on_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_unsupported_relation_raises():
    from matrix_adapter.model import MessageRelation
    client = _new_client()
    # Build a relation with a bad type by going around the literal-typing check.
    rel = MessageRelation.__new__(MessageRelation)
    rel.related_event_id = "$r"
    rel.relation_type = "bogus"
    with pytest.raises(ValueError, match="Unsupported relation"):
        await client.send_message("!r:s", "hi", relation=rel)


# ── server.py tool wrappers ──────────────────────────────


@pytest.fixture
def matrix_server_client(monkeypatch):
    """Wire a MagicMock as the matrix server's client global. `client`
    is a module-level type annotation rather than an actual attribute,
    so use raising=False to suppress the AttributeError on the first set."""
    from matrix_adapter import server as matrix_server
    fake = MagicMock()
    monkeypatch.setattr(matrix_server, "client", fake, raising=False)
    return fake


@pytest.mark.asyncio
async def test_tool_send_message_plain(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.send_typing = AsyncMock()
    matrix_server_client.send_message = AsyncMock(return_value="$id")
    out = await matrix_server.send_message("!r:s", "hi")
    assert out == "$id"


@pytest.mark.asyncio
async def test_tool_send_message_with_action_prepends(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.send_typing = AsyncMock()
    matrix_server_client.send_message = AsyncMock(return_value="$id")
    await matrix_server.send_message("!r:s", "hi", action="leans back")
    sent_text = matrix_server_client.send_message.await_args.args[1]
    assert sent_text.startswith("> leans back")


@pytest.mark.asyncio
async def test_tool_send_message_validates_relation_pair(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.send_typing = AsyncMock()
    with pytest.raises(ValueError, match="must both"):
        await matrix_server.send_message("!r:s", "hi", related_event_id="$x")  # no type


@pytest.mark.asyncio
async def test_tool_react(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.send_reaction = AsyncMock()
    await matrix_server.react("!r:s", "$t", "👍")
    matrix_server_client.send_reaction.assert_awaited_once_with("!r:s", "$t", "👍")


@pytest.mark.asyncio
async def test_tool_read_receipt(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.send_read_receipt = AsyncMock()
    await matrix_server.read_receipt("!r:s", "$e")
    matrix_server_client.send_read_receipt.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_typing_indicator(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.send_typing = AsyncMock()
    await matrix_server.typing_indicator("!r:s")
    matrix_server_client.send_typing.assert_awaited_once_with("!r:s", typing=True)
    await matrix_server.typing_indicator("!r:s", stop=True)
    matrix_server_client.send_typing.assert_awaited_with("!r:s", typing=False)


@pytest.mark.asyncio
async def test_tool_edit_message(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.send_typing = AsyncMock()
    matrix_server_client.send_message = AsyncMock(return_value="$new")
    await matrix_server.edit_message("!r:s", "$orig", "fixed")
    rel = matrix_server_client.send_message.await_args.kwargs.get("relation") or matrix_server_client.send_message.await_args.args[2]
    assert rel.relation_type == "m.replace"
    assert rel.related_event_id == "$orig"


@pytest.mark.asyncio
async def test_tool_redact_message(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.redact_message = AsyncMock()
    await matrix_server.redact_message("!r:s", "$e", reason="spam")
    matrix_server_client.redact_message.assert_awaited_once_with("!r:s", "$e", "spam")


@pytest.mark.asyncio
async def test_tool_pin_message(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.pin_message = AsyncMock()
    await matrix_server.pin_message("!r:s", "$t")
    matrix_server_client.pin_message.assert_awaited_once_with("!r:s", "$t")


@pytest.mark.asyncio
async def test_tool_unpin_message(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.unpin_message = AsyncMock()
    await matrix_server.unpin_message("!r:s", "$t")
    matrix_server_client.unpin_message.assert_awaited_once_with("!r:s", "$t")


@pytest.mark.asyncio
async def test_tool_get_room_members(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.get_room_members = MagicMock(return_value=[])
    out = await matrix_server.get_room_members("!r:s")
    assert out == []


@pytest.mark.asyncio
async def test_tool_get_message(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.get_message = AsyncMock(return_value="msg")
    out = await matrix_server.get_message("!r:s", "$e")
    assert out == "msg"


@pytest.mark.asyncio
async def test_tool_get_user_profile(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.get_user_profile = AsyncMock(return_value="profile")
    out = await matrix_server.get_user_profile("@a:b", room_id="!r:s")
    assert out == "profile"


@pytest.mark.asyncio
async def test_tool_send_attachment(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.upload_and_send_attachment = AsyncMock()
    await matrix_server.send_attachment("!r:s", b"bytes", "x.png", content_type="image/png", caption="hi")
    matrix_server_client.upload_and_send_attachment.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_update_profile_name_only(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.set_display_name = AsyncMock()
    matrix_server_client.upload_avatar = AsyncMock()
    await matrix_server.update_profile(display_name="New")
    matrix_server_client.set_display_name.assert_awaited_once_with("New")
    matrix_server_client.upload_avatar.assert_not_called()


@pytest.mark.asyncio
async def test_tool_update_profile_avatar_detection(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.upload_avatar = AsyncMock()
    # Real PNG magic bytes so filetype.guess returns image/png.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    await matrix_server.update_profile(avatar=png)
    args = matrix_server_client.upload_avatar.await_args.args
    assert args[1] == "image/png"


@pytest.mark.asyncio
async def test_tool_update_profile_bad_avatar_raises(matrix_server_client):
    from matrix_adapter import server as matrix_server
    with pytest.raises(ValueError, match="avatar image type"):
        await matrix_server.update_profile(avatar=b"not-an-image")


@pytest.mark.asyncio
async def test_tool_mxc_resource_unencrypted(matrix_server_client):
    from matrix_adapter import server as matrix_server
    matrix_server_client.download_attachment = AsyncMock(return_value=b"data")
    out = await matrix_server.mxc_resource(server="srv", media_id="abc", mime="text/plain")
    content = out.contents[0]
    assert content.content == b"data"
    assert content.mime_type == "text/plain"


@pytest.mark.asyncio
async def test_forward_events_retries_on_failure(monkeypatch):
    """When session-manager rejects, the forwarder retries until it succeeds."""
    from matrix_adapter import server as matrix_server

    matrix_server._event_queue = asyncio.Queue()
    matrix_server.session_manager_url = "http://sm"

    calls = 0
    async def flaky_post(url, json):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("session-manager down")
        return MagicMock()

    matrix_server._http = MagicMock()
    matrix_server._http.post = AsyncMock(side_effect=flaky_post)

    # Don't actually sleep between retries.
    monkeypatch.setattr(matrix_server.asyncio, "sleep", AsyncMock())

    await matrix_server._event_queue.put({"source": "matrix"})
    task = asyncio.create_task(matrix_server._forward_events())
    # Wait until the queue drains.
    await matrix_server._event_queue.join()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert calls == 3
