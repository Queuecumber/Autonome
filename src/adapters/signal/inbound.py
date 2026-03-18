"""Signal inbound listener: polls signal-cli and pushes events to session manager."""

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


class SignalInbound:
    """Polls signal-cli for messages and pushes events to the session manager."""

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

    async def poll_messages(self) -> None:
        try:
            resp = await self._http.get(
                f"{self.signal_cli_url}/v1/receive/{self.account}",
            )
            resp.raise_for_status()
            messages = resp.json()
        except Exception as e:
            logger.error(f"Failed to poll signal-cli: {e}")
            return

        for envelope in messages:
            data_msg = envelope.get("envelope", {}).get("dataMessage")
            if not data_msg or not data_msg.get("message"):
                continue

            sender = envelope["envelope"].get("source")
            if self.allow_from and sender not in self.allow_from:
                logger.debug(f"Ignoring message from unauthorized sender: {sender}")
                continue

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

            try:
                await self._http.post(
                    f"{self.session_manager_url}/event",
                    json=event,
                )
            except Exception as e:
                logger.error(f"Failed to push event to session manager: {e}")

    async def run(self, poll_interval: float = 1.0) -> None:
        logger.info(f"Signal inbound listener started. Account: {self.account}")
        while True:
            await self.poll_messages()
            await asyncio.sleep(poll_interval)

    async def close(self) -> None:
        await self._http.aclose()
