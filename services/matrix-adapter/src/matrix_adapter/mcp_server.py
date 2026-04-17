"""Matrix adapter — MCP interface + inbound event forwarding."""

import asyncio
import base64
import logging
import os

import filetype
import httpx
from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from matrix_adapter.model import MatrixClient, Message, Reaction

logger = logging.getLogger(__name__)

client: MatrixClient
session_manager_url: str
_http: httpx.AsyncClient

mcp = FastMCP("matrix", instructions=(
    "Matrix messaging. Use these tools to communicate with users on Matrix. "
    "You must call send_message to deliver responses — text you generate without "
    "calling send_message is not seen by anyone. The room_id from the event "
    "metadata identifies where to send your response. "
    "When you receive a message, send a read_receipt to acknowledge it, then "
    "start the typing_indicator before composing your response. "
    "When a message has attachments, the metadata includes attachment URLs. "
    "To view an attachment, call get_attachment with the mxc:// URL."
))


@mcp.tool
async def send_message(room_id: str, text: str) -> None:
    """Send a text message to a Matrix room. Automatically stops typing indicator."""
    try:
        await client.send_typing(room_id, typing=False)
    except Exception:
        pass
    await client.send_message(room_id, text)


@mcp.tool
async def react(room_id: str, event_id: str, emoji: str) -> None:
    """React to a message with an emoji."""
    await client.send_reaction(room_id, event_id, emoji)


@mcp.tool
async def read_receipt(room_id: str, event_id: str) -> None:
    """Send a read receipt for a message."""
    await client.send_read_receipt(room_id, event_id)


@mcp.tool
async def typing_indicator(room_id: str, stop: bool = False) -> None:
    """Show or hide the typing indicator."""
    await client.send_typing(room_id, typing=not stop)


@mcp.tool
async def get_attachment(mxc_url: str) -> ImageContent | TextContent:
    """Fetch a Matrix attachment by mxc:// URL. Images are returned as ImageContent."""
    data, _ = await client.download_attachment(mxc_url)
    kind = filetype.guess(data)
    if kind and kind.mime.startswith("image/"):
        return ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType=kind.mime)
    return TextContent(type="text", text=f"[attachment: {kind.mime if kind else 'unknown'}, {len(data)} bytes]")


@mcp.tool
async def send_attachment(room_id: str, data: bytes, filename: str, content_type: str = "application/octet-stream") -> None:
    """Send a file attachment to a Matrix room."""
    if content_type.startswith("image/"):
        await client.upload_and_send_image(room_id, data, content_type, filename)
    else:
        resp, _ = await client._client.upload(data, content_type=content_type, filename=filename)
        await client._client.room_send(
            room_id, "m.room.message",
            {"msgtype": "m.file", "url": resp.content_uri, "body": filename},
        )


@mcp.tool
async def update_profile(display_name: str | None = None, avatar: bytes | None = None) -> None:
    """Update the Matrix profile. Set display_name and/or avatar (image bytes)."""
    if display_name is not None:
        await client.set_display_name(display_name)
    if avatar is not None:
        kind = filetype.guess(avatar)
        content_type = kind.mime if kind else "image/png"
        resp, _ = await client._client.upload(avatar, content_type=content_type, filename="avatar")
        await client.set_avatar(resp.content_uri)


# ── Inbound event forwarding ─────────────────────────────

async def on_message(msg: Message | Reaction) -> None:
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

    homeserver = os.environ.get("MATRIX_HOMESERVER", "http://localhost:8008")
    user_id = os.environ.get("MATRIX_USER_ID", "")
    device_id = os.environ.get("MATRIX_DEVICE_ID", "AUTONOME")
    password = os.environ.get("MATRIX_PASSWORD", "")
    access_token = os.environ.get("MATRIX_ACCESS_TOKEN", "")
    allowed_rooms = os.environ.get("MATRIX_ALLOWED_ROOMS", "").split(",") if os.environ.get("MATRIX_ALLOWED_ROOMS") else []
    session_manager_url = os.environ.get("SESSION_MANAGER_URL", "http://localhost:5000")
    mcp_port = int(os.environ.get("CHANNEL_MCP_PORT", "8200"))

    client = MatrixClient(
        homeserver=homeserver,
        user_id=user_id,
        device_id=device_id,
        password=password if password else None,
        access_token=access_token if access_token else None,
        allowed_rooms=allowed_rooms if allowed_rooms else None,
    )
    _http = httpx.AsyncClient(timeout=600)

    await client.login()

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
