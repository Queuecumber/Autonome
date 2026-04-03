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

from signal_adapter.model import SignalClient, Message, Reaction

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
    "start the typing_indicator before composing your response. "
    "When a message has attachments, the metadata includes attachment IDs. "
    "To view an attachment, call get_attachment with the attachment ID."
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
    recipient: str, data: str, caption: str | None = None
) -> None:
    """Send a file attachment to a recipient on Signal. Data is base64-encoded file content."""
    await client.send_attachment(recipient, base64.b64decode(data), caption)


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


@mcp.tool
async def get_attachment(attachment_id: str) -> dict:
    """Fetch a Signal attachment by ID. Returns attachment with content_type and content_base64."""
    att = await client.fetch_attachment(attachment_id)
    return {
        "id": att.id,
        "content_type": att.content_type,
        "content_base64": att.content_base64,
    }


@mcp.tool
async def update_profile(
    name: str | None = None, about: str | None = None
) -> None:
    """Update the Signal profile name and/or about text."""
    await client.update_profile(name=name, about=about)


@mcp.tool
async def update_profile_avatar(attachment_id: str) -> None:
    """Set the Signal profile avatar from a Signal attachment ID. Send yourself the image first, then use its attachment ID."""
    att = await client.fetch_attachment(attachment_id)
    avatar = base64.b64decode(att.content_base64) if att.content_base64 else None
    if not avatar:
        raise ValueError("Attachment has no content")
    await client.update_profile(avatar=avatar)


# ── Inbound event forwarding ─────────────────────────────

async def on_message(msg: Message | Reaction) -> None:
    """Push a message or reaction to the session manager."""
    logger.info(f"Received: {msg}")
    try:
        await _http.post(f"{session_manager_url}/event", json=msg.to_event())
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
