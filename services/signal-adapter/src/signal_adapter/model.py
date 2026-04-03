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
from typing import Awaitable, Callable
from urllib.parse import urlparse, urlunparse

import httpx
import websockets

logger = logging.getLogger(__name__)


@dataclass
class Attachment:
    """An attachment on a Signal message. Content is lazy — fetch via SignalClient."""
    id: str
    content_type: str | None = None
    filename: str | None = None
    size: int | None = None


@dataclass
class Message:
    """An inbound Signal message with optional attachments."""
    sender: str
    timestamp: int
    text: str | None = None
    attachments: list[Attachment] = field(default_factory=list)

    def to_event(self, source: str = "signal") -> dict:
        """Serialize as a session manager event."""
        metadata: dict = {
            "message_id": str(self.timestamp),
            "sender": self.sender,
        }
        if self.attachments:
            metadata["attachments"] = [
                {"id": att.id, "content_type": att.content_type, "filename": att.filename}
                for att in self.attachments
            ]
        return {
            "source": source,
            "session_id": self.sender,
            "text": self.text or "",
            "metadata": metadata,
        }


@dataclass
class Reaction:
    """A reaction to an existing Signal message."""
    sender: str
    emoji: str
    target_author: str
    target_timestamp: int
    is_remove: bool = False

    def to_event(self, source: str = "signal") -> dict:
        """Serialize as a session manager event."""
        return {
            "source": source,
            "session_id": self.sender,
            "text": f"[reacted with {self.emoji} to message at {self.target_timestamp}]",
            "metadata": {
                "type": "reaction",
                "sender": self.sender,
                "emoji": self.emoji,
                "target_timestamp": str(self.target_timestamp),
                "target_author": self.target_author,
                "is_remove": self.is_remove,
            },
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

        parsed = urlparse(self.signal_cli_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_url = urlunparse(parsed._replace(
            scheme=ws_scheme,
            path=f"/v1/receive/{self.account}",
        ))

    # ── Reading ──────────────────────────────────────────────

    async def listen(self, on_message: Callable[[Message | Reaction], Awaitable[None]]) -> None:
        """Connect to signal-cli WebSocket and deliver parsed Messages and Reactions."""
        logger.info(f"Signal client connecting to {self.ws_url}")
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("Signal WebSocket connected")
                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                            envelope = data.get("envelope")
                            if envelope is None:
                                continue
                            msg = self._parse_envelope(envelope)
                            if msg is not None:
                                await on_message(msg)
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON from signal-cli: {raw[:100]}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def _parse_envelope(self, envelope: dict) -> Message | Reaction | None:
        """Parse a signal-cli envelope into a Message, Reaction, or None."""
        sender = envelope.get("source")
        if not sender:
            return None

        if self.allow_from and sender not in self.allow_from:
            return None

        data_msg = envelope.get("dataMessage")
        if not data_msg:
            return None

        reaction_data = data_msg.get("reaction")
        if reaction_data:
            emoji = reaction_data.get("emoji")
            target_author = reaction_data.get("targetAuthor")
            target_timestamp = reaction_data.get("targetSentTimestamp")
            if not emoji or not target_author or not target_timestamp:
                return None
            return Reaction(
                sender=sender,
                emoji=emoji,
                target_author=target_author,
                target_timestamp=target_timestamp,
                is_remove=reaction_data.get("isRemove", False),
            )

        text = data_msg.get("message")
        timestamp = data_msg.get("timestamp")
        if not timestamp:
            return None
        raw_attachments = data_msg.get("attachments") or []
        attachments = [
            Attachment(
                id=att["id"],
                content_type=att.get("contentType"),
                filename=att.get("fileName"),
                size=att.get("size"),
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
