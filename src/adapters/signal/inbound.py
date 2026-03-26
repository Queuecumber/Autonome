"""Signal inbound listener: connects to signal-cli WebSocket and pushes events to session manager."""

import asyncio
import json
import logging

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

        # Convert http:// to ws:// for WebSocket
        ws_url = self.signal_cli_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ws_url}/v1/receive/{self.account}"

    async def _handle_envelope(self, envelope: dict) -> None:
        """Process a single signal-cli envelope and push event to session manager."""
        data_msg = envelope.get("envelope", {}).get("dataMessage")
        if not data_msg or not data_msg.get("message"):
            return

        sender = envelope["envelope"].get("source")
        if self.allow_from and sender not in self.allow_from:
            logger.debug(f"Ignoring message from unauthorized sender: {sender}")
            return

        text = data_msg["message"]
        timestamp = str(data_msg.get("timestamp", ""))

        event = {
            "source": "signal",
            "session_id": sender,
            "text": text,
            "metadata": {
                "message_id": timestamp,
                "sender": sender,
            },
        }

        logger.info(f"Received message from {sender}: {text[:50]}...")

        # Send read receipt
        await self._send_receipt(sender, int(timestamp) if timestamp else 0)

        # Start typing indicator
        await self._set_typing(sender, True)

        try:
            await self._http.post(
                f"{self.session_manager_url}/event",
                json=event,
            )
        except Exception as e:
            logger.error(f"Failed to push event to session manager: {e}")
        finally:
            # Stop typing indicator (agent's send_message will also implicitly stop it)
            await self._set_typing(sender, False)

    async def _send_receipt(self, sender: str, timestamp: int) -> None:
        """Send a read receipt for a message."""
        if not timestamp:
            return
        try:
            await self._http.post(
                f"{self.signal_cli_url}/v1/receipts/{self.account}",
                json={
                    "receipt_type": "read",
                    "target_author": sender,
                    "timestamps": [timestamp],
                },
            )
        except Exception as e:
            logger.debug(f"Failed to send read receipt: {e}")

    async def _set_typing(self, recipient: str, enabled: bool) -> None:
        """Send typing indicator start/stop."""
        try:
            await self._http.put(
                f"{self.signal_cli_url}/v1/typing-indicator/{self.account}",
                json={
                    "recipient": recipient,
                    "stop": not enabled,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to set typing indicator: {e}")

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
