"""Signal outbound MCP server: tools for sending messages, attachments, reactions, receipts, typing."""

import base64
import logging
from pathlib import Path

import httpx
from fastmcp import FastMCP

logger = logging.getLogger(__name__)


class SignalSender:
    """Handles sending messages and signals via signal-cli REST API."""

    def __init__(self, signal_cli_url: str, account: str):
        self.signal_cli_url = signal_cli_url.rstrip("/")
        self.account = account
        self._http = httpx.AsyncClient(timeout=60)

    async def send_text(self, recipient: str, text: str) -> None:
        await self._http.post(
            f"{self.signal_cli_url}/v2/send",
            json={
                "message": text,
                "number": self.account,
                "recipients": [recipient],
            },
        )

    async def send_file(
        self, recipient: str, file_path: str, mime_type: str, caption: str | None = None
    ) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        data = base64.b64encode(path.read_bytes()).decode()
        payload = {
            "message": caption or "",
            "number": self.account,
            "recipients": [recipient],
            "base64_attachments": [data],
        }
        await self._http.post(f"{self.signal_cli_url}/v2/send", json=payload)

        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")

    async def send_receipt(self, sender: str, timestamp: int) -> None:
        await self._http.post(
            f"{self.signal_cli_url}/v1/receipts/{self.account}",
            json={
                "receipt_type": "read",
                "target_author": sender,
                "timestamps": [timestamp],
            },
        )

    async def set_typing(self, recipient: str, stop: bool = False) -> None:
        await self._http.put(
            f"{self.signal_cli_url}/v1/typing-indicator/{self.account}",
            json={
                "recipient": recipient,
                "stop": stop,
            },
        )

    async def close(self) -> None:
        await self._http.aclose()


def create_signal_mcp(sender: SignalSender) -> FastMCP:
    """Create the Signal outbound MCP server."""
    mcp = FastMCP("signal", instructions=(
        "Signal messaging. Use these tools to communicate with users on Signal. "
        "You must call send_message to deliver responses — text you generate without "
        "calling send_message is not seen by anyone. The recipient is the phone number "
        "from the event metadata. "
        "When you receive a message, send a read_receipt to acknowledge it, then "
        "start the typing_indicator before composing your response."
    ))

    @mcp.tool
    async def send_message(recipient: str, text: str) -> str:
        """Send a text message to a recipient on Signal. Automatically stops typing indicator."""
        try:
            await sender.set_typing(recipient, stop=True)
        except Exception:
            pass
        try:
            await sender.send_text(recipient, text)
            return f"Sent message to {recipient}"
        except Exception as e:
            return f"Error sending message: {e}"

    @mcp.tool
    async def send_attachment(
        recipient: str, file_path: str, mime_type: str, caption: str | None = None
    ) -> str:
        """Send a file attachment to a recipient on Signal."""
        try:
            await sender.send_file(recipient, file_path, mime_type, caption)
            return f"Sent {file_path} to {recipient}"
        except Exception as e:
            return f"Error sending attachment: {e}"

    @mcp.tool
    async def react(message_id: str, emoji: str) -> str:
        """React to a message with an emoji."""
        return f"Reacted with {emoji} to {message_id} (stub — signal-cli wiring TODO)"

    @mcp.tool
    async def read_receipt(message_sender: str, message_timestamp: int) -> str:
        """Send a read receipt for a message. Call this when you've read a message."""
        try:
            await sender.send_receipt(message_sender, message_timestamp)
            return f"Read receipt sent to {message_sender}"
        except Exception as e:
            return f"Error sending read receipt: {e}"

    @mcp.tool
    async def typing_indicator(recipient: str, stop: bool = False) -> str:
        """Show or hide the typing indicator. Call with stop=False before composing, stop=True when done."""
        try:
            await sender.set_typing(recipient, stop=stop)
            status = "stopped" if stop else "started"
            return f"Typing indicator {status} for {recipient}"
        except Exception as e:
            return f"Error setting typing indicator: {e}"

    return mcp
