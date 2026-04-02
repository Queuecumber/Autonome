"""Signal adapter — MCP interface + inbound event forwarding.

This is both the interface layer and the application entrypoint for the Signal
adapter. It creates the SignalClient, registers MCP tools over it, and runs
the inbound listener + MCP server.

    python -m signal_adapter.mcp_server
"""

import asyncio
import base64
import logging
import os

import httpx
from fastmcp import FastMCP

from signal_adapter.model import SignalClient, Message, Event, EventMetadata, ImageData

logger = logging.getLogger(__name__)

# ── Client and state (created in main()) ─────────────────

client: SignalClient
session_manager_url: str
_http: httpx.AsyncClient

mcp = FastMCP("signal", instructions=(
    "Signal messaging. Use these tools to communicate with users on Signal. "
    "You must call send_message to deliver responses — text you generate without "
    "calling send_message is not seen by anyone. The recipient is the phone number "
    "from the event metadata. "
    "When you receive a message, send a read_receipt to acknowledge it, then "
    "start the typing_indicator before composing your response."
))


# ── Tools ────────────────────────────────────────────────

@mcp.tool
async def send_message(recipient: str, text: str) -> None:
    """Send a text message to a recipient on Signal. Automatically stops typing indicator."""
    try:
        await client.set_typing(recipient, stop=True)
    except Exception:
        pass
    await client.send_message(recipient, text)


@mcp.tool
async def send_attachment(
    recipient: str, file_path: str, mime_type: str, caption: str | None = None
) -> None:
    """Send a file attachment to a recipient on Signal."""
    await client.send_attachment(recipient, file_path, mime_type, caption)


@mcp.tool
async def react(
    recipient: str, emoji: str, target_author: str, message_timestamp: int
) -> None:
    """React to a message with an emoji. target_author is who sent the message, message_timestamp identifies which message."""
    await client.send_reaction(recipient, emoji, target_author, message_timestamp)


@mcp.tool
async def read_receipt(message_sender: str, message_timestamp: int) -> None:
    """Send a read receipt for a message. Call this when you've read a message."""
    await client.send_receipt(message_sender, message_timestamp)


@mcp.tool
async def typing_indicator(recipient: str, stop: bool = False) -> None:
    """Show or hide the typing indicator. Call with stop=False before composing, stop=True when done."""
    await client.set_typing(recipient, stop=stop)


# ── Resources ────────────────────────────────────────────

@mcp.resource("signal://attachments/{attachment_id}")
async def get_attachment(attachment_id: str) -> bytes:
    """Retrieve a Signal attachment by ID."""
    return await client.fetch_attachment(attachment_id)


# ── Inbound event forwarding ─────────────────────────────

async def on_message(msg: Message) -> None:
    """Convert a Message to a structured Event and push it to the session manager."""
    sender = msg.sender
    timestamp = str(msg.timestamp)

    if msg.reaction:
        r = msg.reaction
        event = Event(
            source="signal",
            session_id=sender,
            text=f"[reacted with {r.emoji} to message at {r.target_timestamp}]",
            metadata=EventMetadata(
                type="reaction",
                sender=sender,
                emoji=r.emoji,
                target_timestamp=str(r.target_timestamp),
                target_author=r.target_author,
                is_remove=r.is_remove,
            ),
        )
        logger.info(f"Received reaction from {sender}: {r.emoji}")
    else:
        text = msg.text
        images: list[ImageData] = []

        for att in msg.attachments:
            if att.is_image:
                try:
                    raw = await client.fetch_attachment(att.id)
                    images.append(ImageData(
                        type=att.content_type,
                        data=base64.b64encode(raw).decode(),
                        filename=att.filename,
                    ))
                    logger.info(f"Fetched image: {att.filename} ({att.content_type}, {len(raw)} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to fetch attachment {att.id}: {e}")
            else:
                suffix = f"[Attachment: {att.filename} ({att.content_type})]"
                text = f"{text}\n{suffix}" if text else suffix

        if not text:
            text = "[sent an image]"

        event = Event(
            source="signal",
            session_id=sender,
            text=text,
            metadata=EventMetadata(
                message_id=timestamp,
                sender=sender,
                images=images,
            ),
        )
        logger.info(f"Received message from {sender}: {text[:50]}{'...' if len(text) > 50 else ''}")

    try:
        await _http.post(f"{session_manager_url}/event", json=event.to_dict())
    except Exception as e:
        logger.error(f"Failed to push event to session manager: {e}")


# ── Entrypoint ───────────────────────────────────────────

async def main():
    global client, session_manager_url, _http

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    signal_cli_url = os.environ.get("SIGNAL_CLI_URL", "http://localhost:8080")
    session_manager_url = os.environ.get("SESSION_MANAGER_URL", "http://localhost:5000")
    account = os.environ.get("SIGNAL_ACCOUNT", "")
    allow_from = os.environ.get("ALLOW_FROM", "").split(",") if os.environ.get("ALLOW_FROM") else []
    mcp_port = int(os.environ.get("CHANNEL_MCP_PORT", "8100"))

    client = SignalClient(
        signal_cli_url=signal_cli_url,
        account=account,
        allow_from=allow_from,
    )
    _http = httpx.AsyncClient(timeout=60)

    try:
        await asyncio.gather(
            client.listen(on_message),
            mcp.run_async(transport="http", host="0.0.0.0", port=mcp_port),
        )
    finally:
        await _http.aclose()
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
