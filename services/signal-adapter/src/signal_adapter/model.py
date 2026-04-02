"""Signal data model — all interaction with signal-cli lives here.

Provides domain-level abstractions for Signal messaging: messages, reactions,
attachments, receipts, typing indicators. Both reading (WebSocket) and writing
(REST API) go through this single client.
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Awaitable

import httpx
import websockets

logger = logging.getLogger(__name__)


@dataclass
class Attachment:
    """An attachment on a Signal message. Content is lazy — fetch via SignalClient."""
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
class Message:
    """An inbound Signal message.

    This is the core data type for both inbound events and session history.
    The session manager receives this serialized as JSON.
    """
    sender: str
    timestamp: int
    text: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    reaction: Reaction | None = None
    # Resolved image content — populated by the adapter before pushing to session manager
    resolved_images: list[dict] = field(default_factory=list)

    def to_event(self, source: str = "signal") -> dict:
        """Serialize as a session manager event."""
        if self.reaction:
            r = self.reaction
            return {
                "source": source,
                "session_id": self.sender,
                "text": f"[reacted with {r.emoji} to message at {r.target_timestamp}]",
                "metadata": {
                    "type": "reaction",
                    "sender": self.sender,
                    "emoji": r.emoji,
                    "target_timestamp": str(r.target_timestamp),
                    "target_author": r.target_author,
                    "is_remove": r.is_remove,
                },
            }

        metadata: dict = {
            "message_id": str(self.timestamp),
            "sender": self.sender,
        }
        if self.resolved_images:
            metadata["images"] = self.resolved_images

        return {
            "source": source,
            "session_id": self.sender,
            "text": self.text or "[sent an image]",
            "metadata": metadata,
        }


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
