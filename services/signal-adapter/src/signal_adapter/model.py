"""Signal data model — all interaction with signal-cli lives here.

Provides domain-level abstractions for Signal messaging: messages, reactions,
attachments, receipts, typing indicators. Both reading (WebSocket) and writing
(REST API) go through this single client.
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Awaitable

import httpx
import websockets

logger = logging.getLogger(__name__)


@dataclass
class Attachment:
    """An attachment on a Signal message."""
    id: str
    content_type: str
    filename: str = "attachment"
    size: int = 0

    @property
    def is_image(self) -> bool:
        return self.content_type.startswith("image/")


@dataclass
class Reaction:
    """A reaction to a Signal message."""
    emoji: str
    sender: str
    target_author: str
    target_timestamp: int
    is_remove: bool = False


@dataclass
class ImageData:
    """A base64-encoded image fetched from an attachment."""
    type: str  # mime type
    data: str  # base64
    filename: str


@dataclass
class EventMetadata:
    """Metadata for a session manager event."""
    message_id: str = ""
    sender: str = ""
    type: str = ""  # "reaction" for reaction events
    emoji: str = ""
    target_timestamp: str = ""
    target_author: str = ""
    is_remove: bool = False
    images: list[ImageData] = field(default_factory=list)


@dataclass
class Event:
    """An event to push to the session manager."""
    source: str
    session_id: str
    text: str
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d: dict = {
            "source": self.source,
            "session_id": self.session_id,
            "text": self.text,
            "metadata": {},
        }
        # Only include non-default metadata fields
        m = self.metadata
        if m.message_id: d["metadata"]["message_id"] = m.message_id
        if m.sender: d["metadata"]["sender"] = m.sender
        if m.type: d["metadata"]["type"] = m.type
        if m.emoji: d["metadata"]["emoji"] = m.emoji
        if m.target_timestamp: d["metadata"]["target_timestamp"] = m.target_timestamp
        if m.target_author: d["metadata"]["target_author"] = m.target_author
        if m.is_remove: d["metadata"]["is_remove"] = m.is_remove
        if m.images:
            d["metadata"]["images"] = [
                {"type": img.type, "data": img.data, "filename": img.filename}
                for img in m.images
            ]
        return d


@dataclass
class Message:
    """An inbound Signal message."""
    sender: str
    timestamp: int
    text: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    reaction: Reaction | None = None


class SignalClient:
    """Unified client for signal-cli — reading and writing.

    Read side: WebSocket listener that yields Messages.
    Write side: REST API calls for sending messages, reactions, receipts, typing.
    """

    def __init__(self, signal_cli_url: str, account: str, allow_from: list[str] | None = None):
        self.signal_cli_url = signal_cli_url
        self.account = account
        self.allow_from = allow_from or []
        self._http = httpx.AsyncClient(timeout=60)

        ws_url = self.signal_cli_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ws_url}/v1/receive/{self.account}"

    # ── Reading ──────────────────────────────────────────────

    async def listen(self, on_message: Callable[[Message], Awaitable[None]]) -> None:
        """Connect to signal-cli WebSocket and deliver parsed Messages."""
        logger.info(f"Signal client connecting to {self.ws_url}")
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("Signal WebSocket connected")
                    async for raw in ws:
                        try:
                            envelope = json.loads(raw)
                            msg = self._parse_envelope(envelope)
                            if msg is not None:
                                await on_message(msg)
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON from signal-cli: {raw[:100]}")
                        except Exception as e:
                            logger.error(f"Error processing envelope: {e}")
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def _parse_envelope(self, envelope: dict) -> Message | None:
        """Parse a signal-cli envelope into a Message, or None if irrelevant."""
        env = envelope.get("envelope", {})
        sender = env.get("source")

        if self.allow_from and sender not in self.allow_from:
            return None

        data_msg = env.get("dataMessage")
        if not data_msg:
            return None

        timestamp = data_msg.get("timestamp", 0)

        # Reaction
        reaction_data = data_msg.get("reaction")
        if reaction_data:
            return Message(
                sender=sender,
                timestamp=timestamp,
                reaction=Reaction(
                    emoji=reaction_data.get("emoji", ""),
                    sender=sender,
                    target_author=reaction_data.get("targetAuthor", ""),
                    target_timestamp=reaction_data.get("targetSentTimestamp", 0),
                    is_remove=reaction_data.get("isRemove", False),
                ),
            )

        # Text + attachments
        text = data_msg.get("message", "")
        raw_attachments = data_msg.get("attachments", [])
        attachments = [
            Attachment(
                id=att.get("id", ""),
                content_type=att.get("contentType", ""),
                filename=att.get("fileName", "attachment"),
                size=att.get("size", 0),
            )
            for att in raw_attachments
            if att.get("id")
        ]

        if not text and not attachments:
            return None

        return Message(
            sender=sender,
            timestamp=timestamp,
            text=text,
            attachments=attachments,
        )

    async def fetch_attachment(self, attachment_id: str) -> bytes:
        """Download an attachment from signal-cli by ID."""
        resp = await self._http.get(
            f"{self.signal_cli_url}/v1/attachments/{attachment_id}",
        )
        resp.raise_for_status()
        return resp.content

    # ── Writing ──────────────────────────────────────────────

    async def send_message(self, recipient: str, text: str) -> None:
        """Send a text message."""
        await self._http.post(
            f"{self.signal_cli_url}/v2/send",
            json={
                "message": text,
                "number": self.account,
                "recipients": [recipient],
            },
        )

    async def send_attachment(
        self, recipient: str, file_path: str, mime_type: str, caption: str | None = None
    ) -> None:
        """Send a file attachment."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        data = base64.b64encode(path.read_bytes()).decode()
        await self._http.post(
            f"{self.signal_cli_url}/v2/send",
            json={
                "message": caption or "",
                "number": self.account,
                "recipients": [recipient],
                "base64_attachments": [data],
            },
        )

    async def send_reaction(
        self, recipient: str, emoji: str, target_author: str, timestamp: int
    ) -> None:
        """React to a message."""
        await self._http.post(
            f"{self.signal_cli_url}/v1/reactions/{self.account}",
            json={
                "reaction": emoji,
                "recipient": recipient,
                "target_author": target_author,
                "timestamp": timestamp,
            },
        )

    async def send_receipt(self, sender: str, timestamp: int) -> None:
        """Send a read receipt."""
        await self._http.post(
            f"{self.signal_cli_url}/v1/receipts/{self.account}",
            json={
                "receipt_type": "read",
                "recipient": sender,
                "timestamp": timestamp,
            },
        )

    async def set_typing(self, recipient: str, stop: bool = False) -> None:
        """Start or stop the typing indicator."""
        await self._http.put(
            f"{self.signal_cli_url}/v1/typing-indicator/{self.account}",
            json={
                "recipient": recipient,
                "stop": stop,
            },
        )

    async def close(self) -> None:
        await self._http.aclose()
