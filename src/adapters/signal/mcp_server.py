"""Signal interface — MCP tools + inbound event forwarding over SignalClient.

This is the interface layer between the Signal data model and the rest of the
platform. Outbound: MCP tools for the agent. Inbound: listens to the client's
message stream and pushes events to the session manager.
"""

import base64
import logging

import httpx
from fastmcp import FastMCP

from adapters.signal.model import SignalClient, Message

logger = logging.getLogger(__name__)


class SignalInterface:
    """Full Signal interface: MCP tools (outbound) + event forwarding (inbound)."""

    def __init__(self, client: SignalClient, session_manager_url: str):
        self.client = client
        self.session_manager_url = session_manager_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=60)
        self.mcp = self._create_mcp()

    def _create_mcp(self) -> FastMCP:
        client = self.client

        mcp = FastMCP("signal", instructions=(
            "Signal messaging. Use these tools to communicate with users on Signal. "
            "You must call send_message to deliver responses — text you generate without "
            "calling send_message is not seen by anyone. The recipient is the phone number "
            "from the event metadata. "
            "When you receive a message, send a read_receipt to acknowledge it, then "
            "start the typing_indicator before composing your response."
        ))

        # ── Tools ────────────────────────────────────────────

        @mcp.tool
        async def send_message(recipient: str, text: str) -> str:
            """Send a text message to a recipient on Signal. Automatically stops typing indicator."""
            try:
                await client.set_typing(recipient, stop=True)
            except Exception:
                pass
            try:
                await client.send_message(recipient, text)
                return f"Sent message to {recipient}"
            except Exception as e:
                return f"Error sending message: {e}"

        @mcp.tool
        async def send_attachment(
            recipient: str, file_path: str, mime_type: str, caption: str | None = None
        ) -> str:
            """Send a file attachment to a recipient on Signal."""
            try:
                await client.send_attachment(recipient, file_path, mime_type, caption)
                return f"Sent {file_path} to {recipient}"
            except Exception as e:
                return f"Error sending attachment: {e}"

        @mcp.tool
        async def react(
            recipient: str, emoji: str, target_author: str, message_timestamp: int
        ) -> str:
            """React to a message with an emoji. target_author is who sent the message, message_timestamp identifies which message."""
            try:
                await client.send_reaction(recipient, emoji, target_author, message_timestamp)
                return f"Reacted with {emoji}"
            except Exception as e:
                return f"Error reacting: {e}"

        @mcp.tool
        async def read_receipt(message_sender: str, message_timestamp: int) -> str:
            """Send a read receipt for a message. Call this when you've read a message."""
            try:
                await client.send_receipt(message_sender, message_timestamp)
                return f"Read receipt sent to {message_sender}"
            except Exception as e:
                return f"Error sending read receipt: {e}"

        @mcp.tool
        async def typing_indicator(recipient: str, stop: bool = False) -> str:
            """Show or hide the typing indicator. Call with stop=False before composing, stop=True when done."""
            try:
                await client.set_typing(recipient, stop=stop)
                status = "stopped" if stop else "started"
                return f"Typing indicator {status} for {recipient}"
            except Exception as e:
                return f"Error setting typing indicator: {e}"

        # ── Resources ────────────────────────────────────────

        @mcp.resource("signal://attachments/{attachment_id}")
        async def get_attachment(attachment_id: str) -> bytes:
            """Retrieve a Signal attachment by ID."""
            return await client.fetch_attachment(attachment_id)

        return mcp

    # ── Inbound event forwarding ─────────────────────────

    async def _on_message(self, msg: Message) -> None:
        """Convert a Message to a session manager event and push it."""
        sender = msg.sender
        timestamp = str(msg.timestamp)

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
        try:
            await self._http.post(
                f"{self.session_manager_url}/event",
                json=event,
            )
        except Exception as e:
            logger.error(f"Failed to push event to session manager: {e}")

    # ── Lifecycle ────────────────────────────────────────

    async def run_inbound(self) -> None:
        """Start listening for inbound messages."""
        await self.client.listen(self._on_message)

    async def run_mcp(self, host: str = "0.0.0.0", port: int = 8100) -> None:
        """Start the MCP server."""
        await self.mcp.run_async(transport="http", host=host, port=port)

    async def close(self) -> None:
        await self._http.aclose()
