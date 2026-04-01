"""Signal interface — MCP tools + inbound event forwarding over SignalClient.

This is the interface layer between the Signal data model and the rest of the
platform. Outbound: MCP tools for the agent. Inbound: listens to the client's
message stream and pushes events to the session manager.

The client must be set via init() before tools are called.
"""

import base64
import logging

import httpx
from fastmcp import FastMCP

from adapters.signal.model import SignalClient, Message

logger = logging.getLogger(__name__)

# ── Module state (set by init()) ────────────────────────

_client: SignalClient | None = None
_session_manager_url: str = ""
_http: httpx.AsyncClient | None = None

mcp = FastMCP("signal", instructions=(
    "Signal messaging. Use these tools to communicate with users on Signal. "
    "You must call send_message to deliver responses — text you generate without "
    "calling send_message is not seen by anyone. The recipient is the phone number "
    "from the event metadata. "
    "When you receive a message, send a read_receipt to acknowledge it, then "
    "start the typing_indicator before composing your response."
))


def init(client: SignalClient, session_manager_url: str) -> None:
    """Initialize the module with a SignalClient and session manager URL."""
    global _client, _session_manager_url, _http
    _client = client
    _session_manager_url = session_manager_url
    _http = httpx.AsyncClient(timeout=60)


def _get_client() -> SignalClient:
    assert _client is not None, "Signal MCP server not initialized — call init() first"
    return _client


# ── Tools ────────────────────────────────────────────────

@mcp.tool
async def send_message(recipient: str, text: str) -> str:
    """Send a text message to a recipient on Signal. Automatically stops typing indicator."""
    client = _get_client()
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
        await _get_client().send_attachment(recipient, file_path, mime_type, caption)
        return f"Sent {file_path} to {recipient}"
    except Exception as e:
        return f"Error sending attachment: {e}"


@mcp.tool
async def react(
    recipient: str, emoji: str, target_author: str, message_timestamp: int
) -> str:
    """React to a message with an emoji. target_author is who sent the message, message_timestamp identifies which message."""
    try:
        await _get_client().send_reaction(recipient, emoji, target_author, message_timestamp)
        return f"Reacted with {emoji}"
    except Exception as e:
        return f"Error reacting: {e}"


@mcp.tool
async def read_receipt(message_sender: str, message_timestamp: int) -> str:
    """Send a read receipt for a message. Call this when you've read a message."""
    try:
        await _get_client().send_receipt(message_sender, message_timestamp)
        return f"Read receipt sent to {message_sender}"
    except Exception as e:
        return f"Error sending read receipt: {e}"


@mcp.tool
async def typing_indicator(recipient: str, stop: bool = False) -> str:
    """Show or hide the typing indicator. Call with stop=False before composing, stop=True when done."""
    try:
        await _get_client().set_typing(recipient, stop=stop)
        status = "stopped" if stop else "started"
        return f"Typing indicator {status} for {recipient}"
    except Exception as e:
        return f"Error setting typing indicator: {e}"


# ── Resources ────────────────────────────────────────────

@mcp.resource("signal://attachments/{attachment_id}")
async def get_attachment(attachment_id: str) -> bytes:
    """Retrieve a Signal attachment by ID."""
    return await _get_client().fetch_attachment(attachment_id)


# ── Inbound event forwarding ─────────────────────────────

async def on_message(msg: Message) -> None:
    """Convert a Message to a session manager event and push it."""
    client = _get_client()
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
        await _push_event(event)
        return

    text = msg.text
    metadata: dict = {
        "message_id": timestamp,
        "sender": sender,
    }

    image_data = []
    for att in msg.attachments:
        if att.is_image:
            try:
                raw = await client.fetch_attachment(att.id)
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
    await _push_event(event)


async def _push_event(event: dict) -> None:
    assert _http is not None, "Signal MCP server not initialized"
    try:
        await _http.post(f"{_session_manager_url}/event", json=event)
    except Exception as e:
        logger.error(f"Failed to push event to session manager: {e}")


# ── Lifecycle ────────────────────────────────────────────

async def run_inbound() -> None:
    """Start listening for inbound messages."""
    await _get_client().listen(on_message)


async def run_mcp(host: str = "0.0.0.0", port: int = 8100) -> None:
    """Start the MCP server."""
    await mcp.run_async(transport="http", host=host, port=port)


async def close() -> None:
    if _http:
        await _http.aclose()
