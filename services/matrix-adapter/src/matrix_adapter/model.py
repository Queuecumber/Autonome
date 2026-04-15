"""Matrix data model — all interaction with the homeserver lives here."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

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
    RoomEncryptedImage,
    RoomEncryptedFile,
    RoomMessageText,
    RoomMessageImage,
    RoomMessageFile,
    ReactionEvent,
    SyncResponse,
    ToDeviceError,
)

logger = logging.getLogger(__name__)


@dataclass
class Attachment:
    """An attachment on a Matrix message."""
    url: str
    content_type: str | None = None
    filename: str | None = None
    size: int | None = None
    content_base64: str | None = None


@dataclass
class Message:
    """An inbound Matrix message."""
    sender: str
    sender_name: str
    room_id: str
    room_name: str
    event_id: str
    encrypted: bool = False
    text: str | None = None
    attachments: list[Attachment] = field(default_factory=list)

    def to_event(self, source: str = "matrix") -> dict:
        metadata: dict = {
            "message_id": self.event_id,
            "sender": self.sender,
            "sender_name": self.sender_name,
            "room_id": self.room_id,
            "room_name": self.room_name,
            "encrypted": self.encrypted,
        }
        if self.attachments:
            metadata["attachments"] = [
                {"url": att.url, "content_type": att.content_type, "filename": att.filename}
                for att in self.attachments
            ]
        return {
            "source": source,
            "session_id": self.room_id,
            "text": self.text or "",
            "metadata": metadata,
        }


@dataclass
class Reaction:
    """A reaction to a Matrix message."""
    sender: str
    sender_name: str
    room_id: str
    event_id: str
    emoji: str
    target_event_id: str

    def to_event(self, source: str = "matrix") -> dict:
        return {
            "source": source,
            "session_id": self.room_id,
            "text": f"[reacted with {self.emoji} to {self.target_event_id}]",
            "metadata": {
                "type": "reaction",
                "sender": self.sender,
                "sender_name": self.sender_name,
                "room_id": self.room_id,
                "emoji": self.emoji,
                "target_event_id": self.target_event_id,
            },
        }


def _room_name(room: MatrixRoom) -> str:
    return room.display_name or room.canonical_alias or room.room_id


def _sender_name(room: MatrixRoom, sender: str) -> str:
    return room.user_name(sender) or sender


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
        self._on_message: Callable[[Message | Reaction], Awaitable[None]] | None = None
        self._encryption_info: dict[str, dict] = {}  # mxc_url -> {key, iv, hash}

    async def login(self) -> None:
        # Reuse saved credentials if available (avoids creating new devices)
        creds_path = Path(self._client.store_path) / "credentials.json" if self._client.store_path else None
        if creds_path and creds_path.exists():
            creds = json.loads(creds_path.read_text())
            self._client.access_token = creds["access_token"]
            self._client.user_id = creds["user_id"]
            self._client.device_id = creds["device_id"]
            self._client.load_store()
            logger.info(f"Restored session for {creds['user_id']} device {creds['device_id']}")
        elif self.access_token:
            self._client.access_token = self.access_token
            self._client.user_id = self.user_id
            self._client.device_id = self.device_id
            self._client.load_store()
            logger.info(f"Using provided access token for {self.user_id} device {self.device_id}")
        else:
            resp = await self._client.login(self.password, device_name="Autonome")
            if not isinstance(resp, LoginResponse):
                raise RuntimeError(f"Matrix login failed: {resp}")
            logger.info(f"Logged in as {self.user_id} device {resp.device_id}")
            # Save credentials so restarts reuse the same device
            if creds_path:
                creds_path.parent.mkdir(parents=True, exist_ok=True)
                creds_path.write_text(json.dumps({
                    "user_id": resp.user_id,
                    "device_id": resp.device_id,
                    "access_token": resp.access_token,
                }))

        await self._client.sync(timeout=10000)

        # Populate room member profiles so display names are available immediately
        for room_id in self._client.rooms:
            await self._client.joined_members(room_id)

        # Accept any pending invites from before callbacks were registered
        for room_id in list(self._client.invited_rooms.keys()):
            logger.info(f"Accepting pending invite to {room_id}")
            await self._client.join(room_id)

        self._trust_all_devices()
        logger.info(f"Ready: {len(self._client.rooms)} rooms")

    async def listen(self, on_message: Callable[[Message | Reaction], Awaitable[None]]) -> None:
        self._on_message = on_message

        self._client.add_response_callback(self._handle_sync, SyncResponse)
        self._client.add_event_callback(self._handle_invite, InviteMemberEvent)
        self._client.add_event_callback(self._handle_text, RoomMessageText)
        self._client.add_event_callback(self._handle_image, RoomMessageImage)
        self._client.add_event_callback(self._handle_image, RoomEncryptedImage)
        self._client.add_event_callback(self._handle_file, RoomMessageFile)
        self._client.add_event_callback(self._handle_file, RoomEncryptedFile)
        self._client.add_event_callback(self._handle_reaction, ReactionEvent)
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
                logger.warning(f"Sync protocol error (retrying): {e}")
                continue

    def _trust_all_devices(self) -> None:
        for user_id in self._client.device_store.users:
            for device_id, olm_device in self._client.device_store[user_id].items():
                if not olm_device.verified:
                    self._client.verify_device(olm_device)

    async def _handle_sync(self, response: SyncResponse) -> None:
        self._trust_all_devices()

    async def _handle_verification_start(self, event: KeyVerificationStart) -> None:
        logger.info(f"Verification request from {event.sender} (tx: {event.transaction_id})")
        resp = await self._client.accept_key_verification(event.transaction_id)
        if isinstance(resp, ToDeviceError):
            logger.error(f"Failed to accept verification: {resp}")

    async def _handle_verification_key(self, event: KeyVerificationKey) -> None:
        sas = self._client.key_verifications.get(event.transaction_id)
        if sas:
            emojis = sas.get_emoji()
            emoji_str = " ".join(f"{e[0]} ({e[1]})" for e in emojis)
            logger.info(f"🔐 VERIFY EMOJIS: {emoji_str}")
            logger.info("Confirm these match in Element, then the bot will auto-confirm.")
            resp = await self._client.confirm_key_verification(event.transaction_id)
            if isinstance(resp, ToDeviceError):
                logger.error(f"Failed to confirm verification: {resp}")

    async def _handle_verification_mac(self, event: KeyVerificationMac) -> None:
        logger.info(f"Verification complete (tx: {event.transaction_id})")

    async def _handle_verification_cancel(self, event: KeyVerificationCancel) -> None:
        logger.info(f"Verification cancelled (tx: {event.transaction_id}): {event.reason}")

    async def _handle_megolm(self, room: MatrixRoom, event: MegolmEvent) -> None:
        logger.warning(f"Undecryptable message in {room.room_id} from {event.sender}")

    async def _handle_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        if event.state_key != self.user_id:
            return
        logger.info(f"Accepting invite to {room.room_id} from {event.sender}")
        await self._client.join(room.room_id)

    def _should_process(self, room: MatrixRoom, sender: str) -> bool:
        if sender == self.user_id:
            return False
        if self.allowed_rooms and room.room_id not in self.allowed_rooms:
            return False
        return True

    async def _handle_text(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if not self._should_process(room, event.sender):
            return
        msg = Message(
            sender=event.sender,
            sender_name=_sender_name(room, event.sender),
            room_id=room.room_id,
            room_name=_room_name(room),
            event_id=event.event_id,
            encrypted=room.encrypted,
            text=event.body,
        )
        logger.info(f"Received text in {_room_name(room)} from {_sender_name(room, event.sender)}")
        if self._on_message:
            await self._on_message(msg)

    def _extract_media(self, event) -> tuple[str, str | None, int | None]:
        """Extract mxc URL, mimetype, and size from a media event. Cache encryption info."""
        content = getattr(event, "source", {}).get("content", {})
        info = content.get("info", {})

        # Encrypted media has content.file.url, unencrypted has content.url
        file_info = content.get("file")
        if file_info:
            url = file_info.get("url", "")
            key_info = file_info.get("key", {})
            self._encryption_info[url] = {
                "key": key_info.get("k", ""),
                "iv": file_info.get("iv", ""),
                "hash": file_info.get("hashes", {}).get("sha256", ""),
            }
        else:
            url = event.url or ""

        return url, info.get("mimetype"), info.get("size")

    async def _handle_image(self, room: MatrixRoom, event: RoomMessageImage) -> None:
        if not self._should_process(room, event.sender):
            return
        url, mimetype, size = self._extract_media(event)
        att = Attachment(
            url=url,
            content_type=mimetype,
            filename=event.body,
            size=size,
        )
        msg = Message(
            sender=event.sender,
            sender_name=_sender_name(room, event.sender),
            room_id=room.room_id,
            room_name=_room_name(room),
            event_id=event.event_id,
            encrypted=room.encrypted,
            attachments=[att],
        )
        logger.info(f"Received image in {_room_name(room)} from {_sender_name(room, event.sender)}")
        if self._on_message:
            await self._on_message(msg)

    async def _handle_file(self, room: MatrixRoom, event: RoomMessageFile) -> None:
        if not self._should_process(room, event.sender):
            return
        url, mimetype, size = self._extract_media(event)
        att = Attachment(
            url=url,
            content_type=mimetype,
            filename=event.body,
            size=size,
        )
        msg = Message(
            sender=event.sender,
            sender_name=_sender_name(room, event.sender),
            room_id=room.room_id,
            room_name=_room_name(room),
            event_id=event.event_id,
            encrypted=room.encrypted,
            attachments=[att],
        )
        logger.info(f"Received file in {_room_name(room)} from {_sender_name(room, event.sender)}")
        if self._on_message:
            await self._on_message(msg)

    async def _handle_reaction(self, room: MatrixRoom, event: ReactionEvent) -> None:
        if not self._should_process(room, event.sender):
            return
        relates_to = event.source.get("content", {}).get("m.relates_to", {})
        target_id = relates_to.get("event_id", "")
        emoji = relates_to.get("key", "")
        reaction = Reaction(
            sender=event.sender,
            sender_name=_sender_name(room, event.sender),
            room_id=room.room_id,
            event_id=event.event_id,
            emoji=emoji,
            target_event_id=target_id,
        )
        logger.info(f"Received reaction in {_room_name(room)} from {_sender_name(room, event.sender)}: {emoji}")
        if self._on_message:
            await self._on_message(reaction)

    # ── Writing ──────────────────────────────────────────────

    async def send_message(self, room_id: str, text: str) -> str:
        resp = await self._client.room_send(
            room_id,
            "m.room.message",
            {"msgtype": "m.text", "body": text},
        )
        return resp.event_id

    async def send_reaction(self, room_id: str, event_id: str, emoji: str) -> None:
        await self._client.room_send(
            room_id,
            "m.reaction",
            {
                "m.relates_to": {
                    "rel_type": "m.annotation",
                    "event_id": event_id,
                    "key": emoji,
                },
            },
        )

    async def send_typing(self, room_id: str, typing: bool = True, timeout: int = 10000) -> None:
        await self._client.room_typing(room_id, typing, timeout=timeout)

    async def send_read_receipt(self, room_id: str, event_id: str) -> None:
        await self._client.room_read_markers(room_id, fully_read_event=event_id, read_event=event_id)

    async def download_attachment(self, mxc_url: str) -> tuple[bytes, str | None]:
        resp = await self._client.download(mxc_url)
        data = resp.body
        content_type = getattr(resp, "content_type", None)

        # Decrypt if we have cached encryption info for this URL
        enc = self._encryption_info.pop(mxc_url, None)
        if enc and enc["key"]:
            from nio.crypto.attachments import decrypt_attachment
            data = decrypt_attachment(data, enc["key"], enc["hash"], enc["iv"])

        return data, content_type

    async def upload_and_send_image(self, room_id: str, data: bytes, content_type: str, filename: str) -> None:
        resp, _ = await self._client.upload(data, content_type=content_type, filename=filename)
        await self._client.room_send(
            room_id,
            "m.room.message",
            {
                "msgtype": "m.image",
                "url": resp.content_uri,
                "body": filename,
                "info": {"mimetype": content_type, "size": len(data)},
            },
        )

    async def set_display_name(self, name: str) -> None:
        await self._client.set_displayname(name)

    async def set_avatar(self, mxc_url: str) -> None:
        await self._client.set_avatar(mxc_url)

    async def close(self) -> None:
        await self._client.close()
