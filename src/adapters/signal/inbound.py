"""Signal inbound listener — bridges SignalClient to the session manager.

Listens for messages via SignalClient.listen(), normalizes them into
session manager events, and pushes them.
"""

import base64
import logging

import httpx

from adapters.signal.model import SignalClient, Message

logger = logging.getLogger(__name__)


class SignalInbound:
    """Receives messages from SignalClient and pushes events to the session manager."""

    def __init__(self, client: SignalClient, session_manager_url: str):
        self.client = client
        self.session_manager_url = session_manager_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=60)

    async def _on_message(self, msg: Message) -> None:
        """Convert a Message to a session manager event and push it."""
        sender = msg.sender
        timestamp = str(msg.timestamp)

        # Reaction event
        if msg.reaction:
            r = msg.reaction
            event = {
                "source": "signal",
                "session_id": sender,
                "text": f"[reacted with {r.emoji} to message at {r.target_timestamp}]",
                "metadata": {
                    "type": "reaction",
                    "sender": sender,
                    "emoji": r.emoji,
                    "target_timestamp": str(r.target_timestamp),
                    "target_author": r.target_author,
                    "is_remove": r.is_remove,
                },
            }
            logger.info(f"Received reaction from {sender}: {r.emoji}")
            await self._push_event(event)
            return

        # Text + attachment event
        text = msg.text
        metadata: dict = {
            "message_id": timestamp,
            "sender": sender,
        }

        # Fetch image attachments as base64
        image_data = []
        for att in msg.attachments:
            if att.is_image:
                try:
                    raw = await self.client.fetch_attachment(att.id)
                    b64 = base64.b64encode(raw).decode()
                    image_data.append({
                        "type": att.content_type,
                        "data": b64,
                        "filename": att.filename,
                    })
                    logger.info(f"Fetched image: {att.filename} ({att.content_type}, {len(raw)} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to fetch attachment {att.id}: {e}")
            else:
                text = f"{text}\n[Attachment: {att.filename} ({att.content_type})]" if text else f"[Attachment: {att.filename} ({att.content_type})]"

        if image_data:
            metadata["images"] = image_data

        if not text:
            text = "[sent an image]"

        event = {
            "source": "signal",
            "session_id": sender,
            "text": text,
            "metadata": metadata,
        }

        logger.info(f"Received message from {sender}: {text[:50]}{'...' if len(text) > 50 else ''}")
        await self._push_event(event)

    async def _push_event(self, event: dict) -> None:
        """Push an event to the session manager."""
        try:
            await self._http.post(
                f"{self.session_manager_url}/event",
                json=event,
            )
        except Exception as e:
            logger.error(f"Failed to push event to session manager: {e}")

    async def run(self) -> None:
        """Start listening — delegates to SignalClient.listen()."""
        await self.client.listen(self._on_message)

    async def close(self) -> None:
        await self._http.aclose()
