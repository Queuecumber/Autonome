"""Signal adapter: bridges signal-cli JSON-RPC to LiteLLM with session management."""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from agent_platform.config import build_mcp_tool_declarations
from agent_platform.session import SessionManager
from adapters.signal.channel_mcp import staged_attachments

logger = logging.getLogger(__name__)


class SignalAdapter:
    def __init__(
        self,
        config: dict,
        litellm_url: str,
        session_dir: Path,
    ):
        self.config = config
        self.litellm_url = litellm_url.rstrip("/")

        channel_config = config["channels"]["signal"]
        self.signal_cli_url = channel_config["signal_cli"].rstrip("/")
        self.account = channel_config["account"]
        self.allow_from = channel_config.get("allow_from", [])

        session_config = config.get("session", {})
        max_tokens = session_config.get("max_history_tokens", 100000)
        self.session = SessionManager(store_dir=session_dir, max_history_tokens=max_tokens)

        self.mcp_tools = build_mcp_tool_declarations(config)
        self.model = "main"

        self.heartbeat_prompt = config.get("heartbeat", {}).get(
            "prompt", "Check HEARTBEAT.md"
        )

        self._http = httpx.AsyncClient(timeout=300)

    async def handle_message(self, sender: str, text: str, message_id: str | None = None) -> str | None:
        """Process an incoming message and return the agent's response."""
        if self.allow_from and sender not in self.allow_from:
            logger.warning(f"Ignoring message from unauthorized sender: {sender}")
            return None

        conversation_id = sender

        # Load session history
        history = self.session.load_truncated(contact=sender, model=self.config["model"]["model"])

        # Build the new user message
        user_msg = {"role": "user", "content": text}

        # Build chat completion request
        messages = history + [user_msg]
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": self.mcp_tools,
        }

        # Call LiteLLM
        try:
            resp = await self._http.post(
                f"{self.litellm_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            logger.error(f"LiteLLM request failed: {e}")
            return f"Error: {e}"

        assistant_msg = result["choices"][0]["message"]
        assistant_text = assistant_msg.get("content", "")

        # Save the exchange to session.
        # Note: LiteLLM handles the full tool execution loop server-side and
        # returns only the final response. Intermediate tool_call/tool_result
        # messages are not available to us. We save the final text exchange.
        self.session.append(sender, [user_msg, {"role": "assistant", "content": assistant_text}])

        # Drain staged attachments
        attachments = staged_attachments.drain(conversation_id)

        # Send response via signal-cli
        await self._send_signal_message(sender, assistant_text, attachments)

        # Clean up attachment files after successful delivery
        for att in attachments:
            try:
                Path(att["file_path"]).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean up attachment {att['file_path']}: {e}")

        return assistant_text

    async def handle_heartbeat(self) -> str | None:
        """Process a heartbeat: send heartbeat prompt to agent, deliver response."""
        if not self.allow_from:
            logger.warning("No contacts configured for heartbeat delivery")
            return None

        contact = self.allow_from[0]
        return await self.handle_message(contact, f"[HEARTBEAT] {self.heartbeat_prompt}")

    async def _send_signal_message(
        self, recipient: str, text: str, attachments: list[dict] | None = None
    ) -> None:
        """Send a message (and optional attachments) via signal-cli REST API."""
        payload: dict[str, Any] = {
            "message": text,
            "number": self.account,
            "recipients": [recipient],
        }
        if attachments:
            payload["base64_attachments"] = []
            for att in attachments:
                file_path = Path(att["file_path"])
                if file_path.exists():
                    data = base64.b64encode(file_path.read_bytes()).decode()
                    payload["base64_attachments"].append(data)

        try:
            await self._http.post(
                f"{self.signal_cli_url}/v2/send",
                json=payload,
            )
        except Exception as e:
            logger.error(f"Failed to send Signal message: {e}")

    async def poll_messages(self) -> None:
        """Poll signal-cli for new messages and process them."""
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
            text = data_msg["message"]
            timestamp = str(data_msg.get("timestamp", ""))

            await self.handle_message(sender, text, message_id=timestamp)

    async def run(self, poll_interval: float = 1.0) -> None:
        """Main loop: poll signal-cli for messages."""
        logger.info(f"Signal adapter started. Account: {self.account}")
        while True:
            await self.poll_messages()
            await asyncio.sleep(poll_interval)

    async def close(self) -> None:
        await self._http.aclose()
