"""Signal inbound listener: connects to signal-cli WebSocket and pushes events to session manager."""

import asyncio
import base64
import json
import logging
from pathlib import Path

import httpx
import websockets

logger = logging.getLogger(__name__)


class SignalInbound:
    """Connects to signal-cli WebSocket for messages and pushes events to the session manager."""

    def __init__(
        self,
        signal_cli_url: str,
        session_manager_url: str,
        account: str,
        allow_from: list[str] | None = None,
    ):
        self.signal_cli_url = signal_cli_url.rstrip("/")
        self.session_manager_url = session_manager_url.rstrip("/")
        self.account = account
        self.allow_from = allow_from or []
        self._http = httpx.AsyncClient(timeout=60)

        ws_url = self.signal_cli_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ws_url}/v1/receive/{self.account}"

    async def _handle_envelope(self, envelope: dict) -> None:
        """Process a single signal-cli envelope and push event to session manager."""
        env = envelope.get("envelope", {})
        sender = env.get("source")

        if self.allow_from and sender not in self.allow_from:
            logger.debug(f"Ignoring message from unauthorized sender: {sender}")
            return

        data_msg = env.get("dataMessage")
        if not data_msg:
            return

        timestamp = str(data_msg.get("timestamp", ""))

        # Handle reactions
        reaction = data_msg.get("reaction")
        if reaction:
            event = {
                "source": "signal",
                "session_id": sender,
                "text": f"[reacted with {reaction.get('emoji', '?')} to message at {reaction.get('targetSentTimestamp', '?')}]",
                "metadata": {
                    "type": "reaction",
                    "sender": sender,
                    "emoji": reaction.get("emoji", ""),
                    "target_timestamp": str(reaction.get("targetSentTimestamp", "")),
                    "target_author": reaction.get("targetAuthor", ""),
                    "is_remove": reaction.get("isRemove", False),
                },
            }
            logger.info(f"Received reaction from {sender}: {reaction.get('emoji')}")
            await self._push_event(event)
            return

        # Handle text messages (with optional attachments)
        text = data_msg.get("message", "")
        attachments = data_msg.get("attachments", [])

        # Skip if no text and no attachments
        if not text and not attachments:
            return

        metadata: dict = {
            "message_id": timestamp,
            "sender": sender,
        }

        # Process attachments — build content parts for multimodal
        image_data = []
        for att in attachments:
            content_type = att.get("contentType", "")
            file_path = att.get("file")
            filename = att.get("fileName", "attachment")

            if file_path and content_type.startswith("image/"):
                # Read image file and base64 encode for vision
                try:
                    path = Path(file_path)
                    if path.exists():
                        b64 = base64.b64encode(path.read_bytes()).decode()
                        image_data.append({
                            "type": content_type,
                            "data": b64,
                            "filename": filename,
                        })
                        logger.info(f"Processed image attachment: {filename} ({content_type})")
                except Exception as e:
                    logger.warning(f"Failed to read attachment {file_path}: {e}")
            elif file_path:
                # Non-image attachment — just note it in text
                text = f"{text}\n[Attachment: {filename} ({content_type})]" if text else f"[Attachment: {filename} ({content_type})]"

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
        """Connect to signal-cli WebSocket and stream messages."""
        logger.info(f"Signal inbound connecting to {self.ws_url}")
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("Signal WebSocket connected")
                    async for raw_msg in ws:
                        try:
                            envelope = json.loads(raw_msg)
                            await self._handle_envelope(envelope)
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON message from signal-cli: {raw_msg[:100]}")
                        except Exception as e:
                            logger.error(f"Error processing envelope: {e}")
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def close(self) -> None:
        await self._http.aclose()
