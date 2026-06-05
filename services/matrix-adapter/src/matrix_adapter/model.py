"""Matrix data model — all interaction with the homeserver lives here."""

import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Literal, Self, Union
from urllib.parse import urlencode, urlsplit, urlunsplit

import mistune
from nio import (
    AsyncClient,
    AsyncClientConfig,
    InviteMemberEvent,
    KeyVerificationCancel,
    KeyVerificationKey,
    KeyVerificationMac,
    KeyVerificationStart,
    LocalProtocolError,
    LoginResponse,
    MatrixRoom,
    MegolmEvent,
    ProfileGetResponse,
    ReactionEvent,
    RoomEncryptedFile,
    RoomEncryptedImage,
    RoomGetStateEventResponse,
    RoomMessageFile,
    RoomMessageImage,
    RoomMessageText,
    RoomPutStateResponse,
    RoomRedactResponse,
    RoomSendResponse,
    SyncResponse,
    ToDeviceError,
    UnknownEvent,
)
from nio.crypto.attachments import decrypt_attachment
from nio.events.room_events import RedactionEvent

logger = logging.getLogger(__name__)

# Markdown → HTML for outbound messages. task_lists is deliberately omitted:
# Matrix's org.matrix.custom.html allowlist rejects <input>, so checklists
# render as empty boxes. Users can fall back to emoji.
_MARKDOWN = mistune.create_markdown(plugins=["strikethrough", "table", "url"])


def _format_ts(ts_ms: int | None) -> str | None:
    """Render a Matrix server timestamp (ms epoch) in the platform's
    default local-time format, matching what time-mcp and the orchestrator
    show the agent elsewhere."""
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


@dataclass
class Sender:
    id: str
    name: str | None = None
    avatar_url: str | None = None

    def __post_init__(self):
        if not self.name:
            self.name = self.id

    @classmethod
    def from_nio(cls, room: MatrixRoom, user_id: str) -> Self:
        return cls(
            id=user_id,
            name=room.user_name(user_id),
            avatar_url=room.avatar_url(user_id),
        )

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "avatar": self.avatar_url}


@dataclass
class UserProfile:
    user_id: str
    display_name: str | None = None
    avatar_url: str | None = None
    trust_status: Literal["verified", "partial", "unverified", "blacklisted", "unknown"] = "unknown"
    power_level: int | None = None


@dataclass
class Room:
    id: str
    display_name: str | None = None
    canonical_alias: str | None = None
    encrypted: bool = False
    member_count: int = 0

    @property
    def name(self) -> str:
        return self.display_name or self.canonical_alias or self.id

    @classmethod
    def from_nio(cls, room: MatrixRoom) -> Self:
        return cls(
            id=room.room_id,
            display_name=room.display_name,
            canonical_alias=room.canonical_alias,
            encrypted=room.encrypted,
            member_count=len(room.users or {}),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "encrypted": self.encrypted,
            "member_count": self.member_count,
        }


@dataclass
class Attachment:
    url: str
    content_type: str | None = None
    filename: str | None = None
    size: int | None = None
    caption: str | None = None

    @classmethod
    def from_nio_event(
        cls,
        event: RoomMessageImage | RoomMessageFile | RoomEncryptedImage | RoomEncryptedFile,
    ) -> Self:
        """Build from a nio media event. MSC2530: body is the caption,
        filename lives in a top-level `filename` field. Legacy: body is the
        filename, no explicit filename field."""
        content = getattr(event, "source", {}).get("content", {})
        info = content.get("info", {})

        params = {"mime": info.get("mimetype", "application/octet-stream")}
        file_info = content.get("file")
        if file_info:
            key = file_info.get("key") or {}
            hashes = file_info.get("hashes") or {}
            params["k"] = key.get("k", "")
            params["iv"] = file_info.get("iv", "")
            params["hash"] = hashes.get("sha256", "")
        parts = urlsplit(event.url)
        url = urlunsplit(parts._replace(query=urlencode({k: v for k, v in params.items() if v})))

        top_filename = content.get("filename") or info.get("filename")
        if top_filename and top_filename != event.body:
            filename = top_filename
            caption = event.body
        else:
            filename = top_filename or event.body
            caption = None

        return cls(
            url=url,
            content_type=info.get("mimetype"),
            filename=filename,
            size=info.get("size"),
            caption=caption,
        )


@dataclass
class MessageRelation:
    related_event_id: str
    relation_type: Literal["m.thread", "m.replace", "m.in_reply_to"]

    @classmethod
    def from_relates_to(cls, relates_to: dict) -> Self | None:
        """Parse a Matrix event's `m.relates_to` content.

        Threads and edits put `event_id` at the top alongside `rel_type`.
        Replies use the nested `m.in_reply_to.event_id` shape with no
        top-level `rel_type`.
        """
        rel_type = relates_to.get("rel_type")
        if rel_type in ("m.thread", "m.replace") and relates_to.get("event_id"):
            return cls(related_event_id=relates_to["event_id"], relation_type=rel_type)
        reply_id = relates_to.get("m.in_reply_to", {}).get("event_id")
        if reply_id:
            return cls(related_event_id=reply_id, relation_type="m.in_reply_to")
        return None

    def apply(self, text: str, html: str) -> tuple[str, str, dict]:
        """Return (body, formatted_body, extra_content) for an outbound
        send_message that participates in this relation. Edits prefix body
        with `* ` per the spec and carry an `m.new_content` block."""
        match self.relation_type:
            case "m.replace":
                return (
                    f"* {text}",
                    f"* {html}",
                    {
                        "m.new_content": {
                            "msgtype": "m.text",
                            "body": text,
                            "format": "org.matrix.custom.html",
                            "formatted_body": html,
                        },
                        "m.relates_to": {"rel_type": "m.replace", "event_id": self.related_event_id},
                    },
                )
            case "m.in_reply_to":
                return (text, html, {"m.relates_to": {"m.in_reply_to": {"event_id": self.related_event_id}}})
            case "m.thread":
                return (text, html, {"m.relates_to": {"rel_type": "m.thread", "event_id": self.related_event_id}})
            case _:
                raise ValueError(f"Unsupported relation_type: {self.relation_type!r}")


@dataclass
class Message:
    sender: Sender
    room: Room
    event_id: str
    text: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    relation: MessageRelation | None = None
    sent_at: str | None = None
    verified: bool = False

    @classmethod
    def from_nio(
        cls,
        room: MatrixRoom,
        event: RoomMessageText | RoomMessageImage | RoomMessageFile | RoomEncryptedImage | RoomEncryptedFile,
    ) -> Self:
        """Build from a nio text or media room-message event."""
        if isinstance(event, RoomMessageText):
            text: str | None = event.body
            attachments: list[Attachment] = []
        else:
            attachment = Attachment.from_nio_event(event)
            text = attachment.caption
            attachments = [attachment]
        return cls(
            sender=Sender.from_nio(room, event.sender),
            room=Room.from_nio(room),
            event_id=event.event_id,
            text=text,
            attachments=attachments,
            relation=MessageRelation.from_relates_to(event.source.get("content", {}).get("m.relates_to", {})),
            sent_at=_format_ts(event.server_timestamp),
            verified=event.verified,
        )

    def to_event(self, source: str = "matrix") -> dict:
        metadata: dict = {
            "message_id": self.event_id,
            "sender": self.sender.to_dict(),
            "room": self.room.to_dict(),
            "sent_at": self.sent_at,
            "verified": self.verified,
        }
        if self.attachments:
            metadata["attachments"] = [
                {
                    "url": att.url,
                    "content_type": att.content_type,
                    "filename": att.filename,
                    "caption": att.caption,
                }
                for att in self.attachments
            ]
        if self.relation:
            metadata["relation"] = {
                "related_event_id": self.relation.related_event_id,
                "relation_type": self.relation.relation_type,
            }
        return {
            "source": source,
            "text": self.text or "",
            "metadata": metadata,
        }


@dataclass
class Reaction:
    sender: Sender
    room: Room
    event_id: str
    emoji: str
    target_event_id: str
    sent_at: str | None = None
    verified: bool = False

    @classmethod
    def from_nio(cls, room: MatrixRoom, event: ReactionEvent) -> Self:
        relates_to = event.source.get("content", {}).get("m.relates_to", {})
        return cls(
            sender=Sender.from_nio(room, event.sender),
            room=Room.from_nio(room),
            event_id=event.event_id,
            emoji=relates_to.get("key", ""),
            target_event_id=relates_to.get("event_id", ""),
            sent_at=_format_ts(event.server_timestamp),
            verified=event.verified,
        )

    def to_event(self, source: str = "matrix") -> dict:
        return {
            "source": source,
            "event_type": "reaction",
            "energy": "passive",
            "text": json.dumps({"type": "reaction", "emoji": self.emoji, "target_event_id": self.target_event_id}),
            "metadata": {
                "message_id": self.event_id,
                "sender": self.sender.to_dict(),
                "room": self.room.to_dict(),
                "sent_at": self.sent_at,
                "verified": self.verified,
            },
        }

@dataclass
class Redaction:
  sender: Sender
  room: Room
  target_event_id: str
  reason: str | None
  sent_at: str | None = None
  verified: bool = False

  @classmethod
  def from_nio(cls, room: MatrixRoom, event: RedactionEvent) -> Self:
      return cls(
          sender=Sender.from_nio(room, event.sender),
          room=Room.from_nio(room),
          target_event_id=event.redacts,
          reason=event.reason,
          sent_at=_format_ts(event.server_timestamp),
          verified=event.verified,
      )

  def to_event(self, source: str = "matrix") -> dict:
    return {
        "source": source,
        "event_type": "redaction",
        "energy": "passive",
        "text": json.dumps({"type": "redaction", "target_event_id": self.target_event_id, "reason": self.reason}),
        "metadata": {
            "sender": self.sender.to_dict(),
            "room": self.room.to_dict(),
            "sent_at": self.sent_at,
            "verified": self.verified,
        },
    }


@dataclass
class RoomPins:
    """A room's `m.room.pinned_events` state changed. Carries the full
    current list of pinned event ids — clients diff against their own
    view if they want add/remove granularity."""
    sender: Sender
    room: Room
    pinned_event_ids: list[str]
    sent_at: str | None = None
    verified: bool = False

    def to_event(self, source: str = "matrix") -> dict:
        return {
            "source": source,
            "event_type": "pinned_events",
            "energy": "passive",
            "text": json.dumps({
                "type": "pinned_events",
                "pinned_event_ids": list(self.pinned_event_ids),
            }),
            "metadata": {
                "sender": self.sender.to_dict(),
                "room": self.room.to_dict(),
                "sent_at": self.sent_at,
                "verified": self.verified,
            },
        }

class MatrixClient:
    """Unified client for Matrix — reading and writing."""

    def __init__(
        self,
        homeserver: str,
        user_id: str,
        device_id: str = "AUTONOME",
        password: str | None = None,
        access_token: str | None = None,
        allowed_rooms: list[str] | None = None,
        store_path: str = "/data/crypto",
    ):
        self.homeserver = homeserver
        self.user_id = user_id
        self.device_id = device_id
        self.password = password
        self.access_token = access_token
        self.allowed_rooms = allowed_rooms
        config = AsyncClientConfig(store_sync_tokens=True, encryption_enabled=True)
        self._client = AsyncClient(
            homeserver, user_id, device_id=device_id,
            store_path=store_path, config=config,
        )
        self._on_message: Callable[[Message | Reaction | Redaction | RoomPins], Awaitable[None]] | None = None
        self._synced_rooms: set[str] = set()

        # Event type → handler dispatch table
        self._handlers: dict[type, Callable] = {
            RoomMessageText: self._on_text,
            RoomMessageImage: self._on_media,
            RoomEncryptedImage: self._on_media,
            RoomMessageFile: self._on_media,
            RoomEncryptedFile: self._on_media,
            ReactionEvent: self._on_reaction,
            RedactionEvent: self._on_redaction,
            UnknownEvent: self._on_unknown,
        }

    # ── Auth ─────────────────────────────────────────────────

    async def login(self) -> None:
        creds_path = Path(self._client.store_path) / "credentials.json" if self._client.store_path else None
        if creds_path and creds_path.exists():
            creds = json.loads(creds_path.read_text())
            self._client.access_token = creds["access_token"]
            self._client.user_id = creds["user_id"]
            self._client.device_id = creds["device_id"]
            self._client.load_store()
            logger.info("Restored session for %s device %s", creds["user_id"], creds["device_id"])
        elif self.access_token:
            self._client.access_token = self.access_token
            self._client.user_id = self.user_id
            self._client.device_id = self.device_id
            self._client.load_store()
            logger.info("Using provided access token for %s device %s", self.user_id, self.device_id)
        else:
            resp = await self._client.login(self.password, device_name="Autonome")
            if not isinstance(resp, LoginResponse):
                raise RuntimeError(f"Matrix login failed: {resp}")
            logger.info("Logged in as %s device %s", self.user_id, resp.device_id)
            if creds_path:
                creds_path.parent.mkdir(parents=True, exist_ok=True)
                creds_path.write_text(json.dumps({
                    "user_id": resp.user_id,
                    "device_id": resp.device_id,
                    "access_token": resp.access_token,
                }))

        # Defer the first sync until after listen() registers event callbacks
        # — otherwise nio advances the sync token over the backlog without
        # firing handlers.

    # ── Listening ────────────────────────────────────────────

    async def listen(self, on_message: Callable[[Message | Reaction | Redaction | RoomPins], Awaitable[None]]) -> None:
        self._on_message = on_message

        self._client.add_response_callback(self._handle_sync, SyncResponse)
        self._client.add_event_callback(self._handle_invite, InviteMemberEvent)
        self._client.add_event_callback(self._handle_event, RoomMessageText)
        self._client.add_event_callback(self._handle_event, RoomMessageImage)
        self._client.add_event_callback(self._handle_event, RoomEncryptedImage)
        self._client.add_event_callback(self._handle_event, RoomMessageFile)
        self._client.add_event_callback(self._handle_event, RoomEncryptedFile)
        self._client.add_event_callback(self._handle_event, ReactionEvent)
        self._client.add_event_callback(self._handle_event, RedactionEvent)
        self._client.add_event_callback(self._handle_event, UnknownEvent)
        self._client.add_event_callback(self._handle_megolm, MegolmEvent)
        self._client.add_to_device_callback(self._handle_verification_start, KeyVerificationStart)
        self._client.add_to_device_callback(self._handle_verification_key, KeyVerificationKey)
        self._client.add_to_device_callback(self._handle_verification_mac, KeyVerificationMac)
        self._client.add_to_device_callback(self._handle_verification_cancel, KeyVerificationCancel)

        logger.info("Starting Matrix sync loop")
        while True:
            try:
                await self._client.sync_forever(timeout=30000, full_state=True)
            except LocalProtocolError as e:
                logger.warning("Sync protocol error (retrying): %s", e)
                continue

    # ── Generic event dispatch ───────────────────────────────

    async def _handle_event(self, room: MatrixRoom, event) -> None:
        if self.allowed_rooms and room.room_id not in self.allowed_rooms:
            return
        # State events (currently UnknownEvent) flow through regardless of
        # sender; timeline events from self are dropped to avoid self-notify.
        if not isinstance(event, UnknownEvent) and event.sender == self.user_id:
            return
        handler = self._handlers.get(type(event))
        if handler:
            await handler(room, event)

    # ── Per-type handlers ────────────────────────────────────

    async def _on_text(self, room: MatrixRoom, event: RoomMessageText) -> None:
        msg = Message.from_nio(room, event)
        logger.info("Received text in %s from %s", msg.room.name, msg.sender.name)
        if self._on_message:
            await self._on_message(msg)

    async def _on_media(self, room: MatrixRoom, event) -> None:
        msg = Message.from_nio(room, event)
        logger.info("Received media in %s from %s", msg.room.name, msg.sender.name)
        if self._on_message:
            await self._on_message(msg)

    async def _on_reaction(self, room: MatrixRoom, event: ReactionEvent) -> None:
        reaction = Reaction.from_nio(room, event)
        logger.info("Received reaction in %s from %s: %s",
                    reaction.room.name, reaction.sender.name, reaction.emoji)
        if self._on_message:
            await self._on_message(reaction)

    async def _on_redaction(self, room: MatrixRoom, event: RedactionEvent) -> None:
        redaction = Redaction.from_nio(room, event)
        logger.info("Received redaction of %s in %s from %s (%s)",
                    redaction.target_event_id, redaction.room.name, redaction.sender.name, redaction.reason)
        if self._on_message:
            await self._on_message(redaction)

    async def _on_unknown(self, room: MatrixRoom, event: UnknownEvent) -> None:
        if event.type == "m.room.pinned_events":
            await self._on_pinned_events(room, event)

    async def _on_pinned_events(self, room: MatrixRoom, event: UnknownEvent) -> None:
        # The pinned_events state event carries the full current list. Emit
        # a single RoomPins with that list; clients diff against their own
        # view if they want add/remove granularity. Suppress during initial
        # sync (room not yet in _synced_rooms) and for self-sent changes.
        new_pins = event.source.get("content", {}).get("pinned", []) or []
        if room.room_id not in self._synced_rooms:
            logger.info("pinned_events for %s during initial sync — %d ids, skipping emit", room.room_id, len(new_pins))
            return
        if event.sender == self.user_id:
            return
        room_pins = RoomPins(
            sender=Sender.from_nio(room, event.sender),
            room=Room.from_nio(room),
            pinned_event_ids=list(new_pins),
            sent_at=_format_ts(getattr(event, "server_timestamp", None)),
            verified=getattr(event, "verified", False),
        )
        logger.info("RoomPins in %s by %s: %d ids", room_pins.room.name, room_pins.sender.name, len(new_pins))
        if self._on_message:
            await self._on_message(room_pins)

    # ── Sync / verification / invite callbacks ───────────────

    def _trust_all_devices(self) -> None:
        for user_id in self._client.device_store.users:
            for device_id, olm_device in self._client.device_store[user_id].items():
                if not olm_device.verified:
                    self._client.verify_device(olm_device)

    async def _handle_sync(self, response: SyncResponse) -> None:
        self._trust_all_devices()
        for room_id in self._client.rooms:
            if room_id not in self._synced_rooms:
                self._synced_rooms.add(room_id)
                await self._client.joined_members(room_id)

    async def _handle_verification_start(self, event: KeyVerificationStart) -> None:
        logger.info("Verification request from %s (tx: %s)", event.sender, event.transaction_id)
        resp = await self._client.accept_key_verification(event.transaction_id)
        if isinstance(resp, ToDeviceError):
            logger.error("Failed to accept verification: %s", resp)

    async def _handle_verification_key(self, event: KeyVerificationKey) -> None:
        sas = self._client.key_verifications.get(event.transaction_id)
        if sas:
            emojis = sas.get_emoji()
            emoji_str = " ".join(f"{e[0]} ({e[1]})" for e in emojis)
            logger.info("🔐 VERIFY EMOJIS: %s", emoji_str)
            logger.info("Confirm these match in Element, then the bot will auto-confirm.")
            resp = await self._client.confirm_key_verification(event.transaction_id)
            if isinstance(resp, ToDeviceError):
                logger.error("Failed to confirm verification: %s", resp)

    async def _handle_verification_mac(self, event: KeyVerificationMac) -> None:
        logger.info("Verification complete (tx: %s)", event.transaction_id)

    async def _handle_verification_cancel(self, event: KeyVerificationCancel) -> None:
        logger.info("Verification cancelled (tx: %s): %s", event.transaction_id, event.reason)

    async def _handle_megolm(self, room: MatrixRoom, event: MegolmEvent) -> None:
        logger.warning("Undecryptable message in %s from %s", room.room_id, event.sender)

    async def _handle_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        if event.state_key != self.user_id:
            return
        logger.info("Accepting invite to %s from %s", room.room_id, event.sender)
        await self._client.join(room.room_id)

    # ── Writing ──────────────────────────────────────────────

    async def _room_send(self, room_id: str, message_type: str, content: dict) -> RoomSendResponse:
        """Send a room event, raising a readable error if matrix-nio returns
        a RoomSendError instead of the success response."""
        resp = await self._client.room_send(room_id, message_type, content)
        if not isinstance(resp, RoomSendResponse):
            raise RuntimeError(f"Matrix room_send failed: {resp}")
        return resp

    async def send_message(
        self,
        room_id: str,
        text: str,
        relation: MessageRelation | None = None,
    ) -> str:
        html = _MARKDOWN(text).strip()
        body, formatted, extra = (relation.apply(text, html) if relation else (text, html, {}))
        content: dict = {
            "msgtype": "m.text",
            "body": body,
            "format": "org.matrix.custom.html",
            "formatted_body": formatted,
            **extra,
        }
        resp = await self._room_send(room_id, "m.room.message", content)
        return resp.event_id

    async def redact_message(self, room_id: str, event_id: str, reason: str | None = None) -> None:
        resp = await self._client.room_redact(room_id, event_id, reason=reason)
        if not isinstance(resp, RoomRedactResponse):
            raise RuntimeError(f"Matrix room_redact failed: {resp}")

    async def send_reaction(self, room_id: str, event_id: str, emoji: str) -> None:
        await self._room_send(
            room_id, "m.reaction",
            {"m.relates_to": {"rel_type": "m.annotation", "event_id": event_id, "key": emoji}},
        )

    async def pin_message(self, room_id: str, event_id: str) -> None:
        await self._update_pinned(room_id, add=event_id)

    async def unpin_message(self, room_id: str, event_id: str) -> None:
        await self._update_pinned(room_id, remove=event_id)

    async def _update_pinned(
        self, room_id: str, add: str | None = None, remove: str | None = None
    ) -> None:
        # Fetch authoritative state before writing — other clients may have
        # edited the list since our last sync.
        current = await self.get_pinned_events(room_id)
        new_list = list(current)
        if add and add not in new_list:
            new_list.append(add)
        if remove and remove in new_list:
            new_list = [e for e in new_list if e != remove]
        if new_list == current:
            return
        resp = await self._client.room_put_state(
            room_id, "m.room.pinned_events", {"pinned": new_list},
        )
        if not isinstance(resp, RoomPutStateResponse):
            raise RuntimeError(f"Matrix room_put_state(pinned_events) failed: {resp}")

    async def get_pinned_events(self, room_id: str) -> list[str]:
        resp = await self._client.room_get_state_event(room_id, "m.room.pinned_events", "")
        if not isinstance(resp, RoomGetStateEventResponse):
            # Room has no pinned_events state event yet — treat as empty.
            return []
        return list(resp.content.get("pinned", []) or [])

    async def send_typing(self, room_id: str, typing: bool = True, timeout: int = 10000) -> None:
        await self._client.room_typing(room_id, typing, timeout=timeout)

    async def send_read_receipt(self, room_id: str, event_id: str) -> None:
        await self._client.room_read_markers(room_id, fully_read_event=event_id, read_event=event_id)

    def get_room_members(self, room_id: str) -> list[Sender]:
        """Return the current member list for a room from nio's synced state.
        Returns an empty list if the room isn't known to this client."""
        room = self._client.rooms.get(room_id)
        if room is None:
            return []
        return [Sender(id=uid, name=user.display_name) for uid, user in (room.users or {}).items()]

    async def download_attachment(
        self, server: str, media_id: str, *,
        k: str = "", iv: str = "", hash: str = "",
    ) -> bytes:
        """Download (and decrypt, if applicable) a Matrix attachment."""
        resp = await self._client.download(f"mxc://{server}/{media_id}")
        if k:
            return decrypt_attachment(resp.body, k, hash, iv)
        return resp.body

    async def get_user_profile(self, user_id: str, room_id: str | None = None) -> UserProfile:
        """Fetch display name, avatar, device-trust status, and (if a
        room_id is given) the user's power level in that room."""
        resp = await self._client.profile_get(user_id)
        if not isinstance(resp, ProfileGetResponse):
            raise RuntimeError(f"Matrix profile_get({user_id}) failed: {resp}")

        power_level: int | None = None
        if room_id:
            room = self._client.rooms.get(room_id)
            if room is not None and room.power_levels is not None:
                power_level = room.power_levels.get_user_level(user_id)

        return UserProfile(
            user_id=user_id,
            display_name=resp.displayname,
            avatar_url=resp.avatar_url,
            trust_status=self._compute_trust_status(user_id),
            power_level=power_level,
        )

    def _compute_trust_status(self, user_id: str) -> str:
        devices = list(self._client.device_store.active_user_devices(user_id))
        if not devices:
            return "unknown"
        if any(d.blacklisted for d in devices):
            return "blacklisted"
        if all(d.verified for d in devices):
            return "verified"
        if any(d.verified for d in devices):
            return "partial"
        return "unverified"

    async def get_message(self, room_id: str, event_id: str) -> Message:
        """Fetch a message by id and return it as a Message dataclass.

        Raises if the event isn't a message we can render — encrypted
        messages we lack the key for, reactions, redactions, state events."""
        resp = await self._client.room_get_event(room_id, event_id)
        event = getattr(resp, "event", None)
        if event is None:
            raise RuntimeError(f"Matrix room_get_event({event_id}) failed: {resp}")

        if isinstance(event, MegolmEvent):
            raise RuntimeError(f"Message {event_id} is encrypted and cannot be decrypted")

        room = self._client.rooms.get(room_id)
        if room is None:
            raise RuntimeError(f"Room {room_id} not in client state")

        if isinstance(event, (RoomMessageText, RoomMessageImage, RoomMessageFile,
                              RoomEncryptedImage, RoomEncryptedFile)):
            return Message.from_nio(room, event)
        raise RuntimeError(f"Event {event_id} is not a message (type={type(event).__name__})")

    async def _upload(
        self, data: bytes, content_type: str, filename: str, encrypt: bool = False
    ) -> tuple[str, dict | None]:
        """Upload bytes and return (mxc:// URL, decryption_info or None).

        With encrypt=True the bytes are encrypted before upload and the
        returned decryption_info carries the url/key/iv/hashes needed to
        decrypt — which is what encrypted-room events put in their `file`
        field."""
        resp, decryption_info = await self._client.upload(
            io.BytesIO(data),
            content_type=content_type,
            filename=filename,
            encrypt=encrypt,
        )
        content_uri = getattr(resp, "content_uri", None)
        if not content_uri:
            raise RuntimeError(f"Upload failed: {resp}")
        return content_uri, decryption_info

    async def upload_and_send_attachment(
        self,
        room_id: str,
        data: bytes,
        content_type: str,
        filename: str,
        caption: str | None = None,
    ) -> None:
        """Upload bytes and send to a room as either an image or a file.
        Auto-detects room encryption and adjusts the event shape accordingly.

        When a caption is present the message body carries the caption and
        `filename` is sent as a separate top-level field (per MSC2530). When
        there's no caption, body carries the filename (the legacy shape)."""
        room = self._client.rooms.get(room_id)
        encrypted = bool(room and room.encrypted)

        content_uri, decryption_info = await self._upload(data, content_type, filename, encrypt=encrypted)
        msgtype = "m.image" if content_type.startswith("image/") else "m.file"
        content: dict = {
            "msgtype": msgtype,
            "body": caption if caption else filename,
            "filename": filename,
            "info": {"mimetype": content_type, "size": len(data)},
        }
        if encrypted and decryption_info:
            # Encrypted-room shape: file carries url + encryption metadata,
            # no top-level url field.
            file_info = dict(decryption_info)
            file_info["url"] = content_uri
            content["file"] = file_info
        else:
            content["url"] = content_uri
        await self._room_send(room_id, "m.room.message", content)

    async def set_display_name(self, name: str) -> None:
        await self._client.set_displayname(name)

    async def upload_avatar(self, data: bytes, content_type: str, filename: str) -> None:
        """Upload bytes and set as the profile avatar. Profile content is
        public; never encrypt."""
        content_uri, _ = await self._upload(data, content_type, filename)
        await self._client.set_avatar(content_uri)

    async def close(self) -> None:
        await self._client.close()
