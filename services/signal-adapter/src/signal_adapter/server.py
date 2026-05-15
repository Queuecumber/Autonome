"""Signal adapter — MCP interface + inbound event forwarding.

This is both the interface layer and the application entrypoint for the Signal
adapter. It creates the SignalClient, registers MCP tools over it, and runs
the inbound listener + MCP server.

    python -m signal_adapter.server
"""

import asyncio
import base64
import logging
import os

import httpx
from fastmcp import FastMCP
from fastmcp.resources import ResourceContent, ResourceResult
from pydantic import Base64Bytes

from signal_adapter.model import SignalClient, Message, Reaction

logger = logging.getLogger(__name__)

# ── Client and state (created in main()) ─────────────────

client: SignalClient
session_manager_url: str
_http: httpx.AsyncClient  # timeout must exceed LLM response time

mcp = FastMCP("signal", instructions=(
    "Signal messaging. Use these tools to communicate with users on Signal. "
    "You must call send_message to deliver responses — text you generate without "
    "calling send_message is not seen by anyone. The recipient is the phone number "
    "from the event metadata. "
    "When you receive a message, send a read_receipt to acknowledge it, then "
    "start the typing_indicator before composing your response. "
    "When a message has attachments, the metadata includes `signal:///attachment/{id}` "
    "URIs. These are MCP resources — read them with the platform's `resources_read` "
    "tool, or pass them as the binary argument to any tool that accepts attachments; "
    "the platform resolves them transparently."
))


# ── Tools ────────────────────────────────────────────────

@mcp.tool
async def send_message(recipient: str, text: str) -> None:
    """Send a text message on Signal.

    Stops the typing indicator automatically.

    Args:
        recipient: A phone number in E.164 format (e.g. `+15551234567`)
            or a Signal group ID — usually from incoming event metadata.
        text: The message body.

    Raises:
        RuntimeError: If signal-cli rejects the send.
    """
    try:
        await client.set_typing(recipient, stop=True)
    except Exception:
        pass
    await client.send_message(recipient, text)


@mcp.tool
async def send_attachment(
    recipient: str, data: Base64Bytes, caption: str | None = None
) -> None:
    """Send a file attachment on Signal.

    Args:
        recipient: Phone number (E.164) or group ID.
        data: The attachment.
        caption: Optional message alongside the attachment.

    Raises:
        RuntimeError: If signal-cli fails to send.
    """
    await client.send_attachment(recipient, data, caption)


@mcp.tool
async def react(
    recipient: str, emoji: str, target_author: str, message_timestamp: int
) -> None:
    """React to a Signal message with an emoji.

    Signal identifies messages by sender + timestamp rather than an event ID.

    Args:
        recipient: The conversation (phone number or group ID) where the
            original message was sent.
        emoji: The emoji to react with.
        target_author: Sender of the message being reacted to.
        message_timestamp: Timestamp of the message being reacted to.

    Raises:
        RuntimeError: If signal-cli fails to send the reaction.
    """
    await client.send_reaction(recipient, emoji, target_author, message_timestamp)


@mcp.tool
async def read_receipt(message_sender: str, message_timestamp: int) -> None:
    """Send a read receipt for a Signal message.

    Conventionally sent when you've read an incoming message, before
    composing a response.

    Args:
        message_sender: Sender of the message being acknowledged.
        message_timestamp: Timestamp of the message being acknowledged.
    """
    await client.send_receipt(message_sender, message_timestamp)


@mcp.tool
async def typing_indicator(recipient: str, stop: bool = False) -> None:
    """Show or hide the typing indicator on Signal.

    `send_message` automatically stops the indicator on send.

    Args:
        recipient: Phone number (E.164) or group ID.
        stop: Hide the indicator instead of showing it.
    """
    await client.set_typing(recipient, stop=stop)


@mcp.resource("signal:///attachment/{attachment_id}")
async def signal_attachment_resource(attachment_id: str) -> ResourceResult:
    """Serve Signal attachments as MCP resources.

    Returns a `ResourceResult` carrying the attachment bytes and signal-cli's
    declared content_type. Falls back to `application/octet-stream` when
    signal-cli didn't report one.

    Args:
        attachment_id: The attachment id from incoming event metadata.
    """
    att = await client.fetch_attachment(attachment_id)
    data = base64.b64decode(att.content_base64)
    return ResourceResult([ResourceContent(data, mime_type=att.content_type or "application/octet-stream")])


@mcp.tool
async def update_profile(
    name: str | None = None, about: str | None = None
) -> None:
    """Update this Signal account's profile.

    Args:
        name: New display name. Omit to leave unchanged.
        about: New status/bio text. Omit to leave unchanged.

    Raises:
        RuntimeError: If signal-cli fails to update the profile.
    """
    await client.update_profile(name=name, about=about)


@mcp.tool
async def update_profile_avatar(avatar: Base64Bytes) -> None:
    """Set this Signal account's profile avatar.

    Args:
        avatar: The new profile picture.

    Raises:
        RuntimeError: If signal-cli fails to upload the avatar.
    """
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
    _http = httpx.AsyncClient(timeout=600)

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
