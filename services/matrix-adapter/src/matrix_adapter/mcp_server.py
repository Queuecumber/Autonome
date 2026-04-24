"""Matrix adapter — MCP interface + inbound event forwarding."""

import asyncio
import base64
import io
import json
import logging
import os

import exifread
import filetype
import httpx
from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from pydantic import Base64Bytes

from matrix_adapter.model import MatrixClient, Message, Reaction


def _exif_summary(data: bytes) -> dict | None:
    """Extract a small, useful subset of EXIF from image bytes.
    Returns None if no EXIF present or parsing fails."""
    try:
        tags = exifread.process_file(io.BytesIO(data), details=False)
    except Exception:
        return None
    if not tags:
        return None

    out: dict = {}

    dt = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    if dt:
        out["datetime"] = str(dt)

    make = tags.get("Image Make")
    model = tags.get("Image Model")
    if make or model:
        out["camera"] = " ".join(str(x) for x in (make, model) if x).strip()

    lat = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")
    if lat and lon and lat_ref and lon_ref:
        try:
            def _to_deg(ratio, ref):
                d, m, s = [float(r.num) / float(r.den) for r in ratio.values]
                decimal = d + m / 60 + s / 3600
                return -decimal if str(ref) in ("S", "W") else decimal
            out["gps"] = {"lat": round(_to_deg(lat, lat_ref), 6), "lon": round(_to_deg(lon, lon_ref), 6)}
        except Exception:
            pass

    w = tags.get("EXIF ExifImageWidth") or tags.get("Image ImageWidth")
    h = tags.get("EXIF ExifImageLength") or tags.get("Image ImageLength")
    if w and h:
        out["dimensions"] = f"{w}x{h}"

    software = tags.get("Image Software")
    if software:
        out["software"] = str(software)

    return out or None

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
    except Exception as e:
        # Typing indicator is best-effort — never let it block the actual send.
        logger.debug(f"stop-typing before send_message failed: {e!r}")
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
async def get_room_members(room_id: str) -> list[dict]:
    """List the people currently in a Matrix room.

    Each event already carries a `member_count` in its metadata — use that
    as your first signal for how public a room is. Call this tool to see
    who specifically is there when:

    - You're about to share anything private or personal and the room has
      more than two members
    - Someone new joined the conversation and you don't recognize them
    - A member_count you noticed before has changed

    Returns [{id, name}, ...]. Large rooms are truncated to 50 entries
    with a `_truncated` marker appended."""
    members = client.get_room_members(room_id)
    out: list[dict] = [{"id": m.id, "name": m.name} for m in members[:50]]
    if len(members) > 50:
        out.append({"id": "_truncated", "name": f"{len(members) - 50} more not shown"})
    return out


@mcp.tool
async def get_attachment(mxc_url: str) -> list[ImageContent | TextContent]:
    """Fetch a Matrix attachment by mxc:// URL.

    For images, returns the image content plus (when present) an EXIF
    summary with the fields most likely to tell you something useful:
    datetime the photo was taken, camera make/model, GPS coordinates,
    dimensions, and editing software. Photos from phones and most
    cameras carry this; screenshots and many forwarded images don't.

    Non-image attachments return a single text block describing the
    file."""
    data, _ = await client.download_attachment(mxc_url)
    kind = filetype.guess(data)
    if kind and kind.mime.startswith("image/"):
        result: list[ImageContent | TextContent] = [
            ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType=kind.mime),
        ]
        exif = _exif_summary(data)
        if exif:
            result.append(TextContent(type="text", text=json.dumps(exif)))
        return result
    return [TextContent(type="text", text=f"[attachment: {kind.mime if kind else 'unknown'}, {len(data)} bytes]")]


@mcp.tool
async def send_attachment(
    room_id: str,
    data: Base64Bytes,
    filename: str,
    content_type: str = "application/octet-stream",
    caption: str | None = None,
) -> None:
    """Send a file attachment to a Matrix room.

    `filename` is the filename the recipient will see in their client
    (e.g. `photo.jpg`, `report.pdf`). Don't put a caption here.

    `caption` is an optional message to go with the attachment — the thing
    you'd type alongside the image. Leave it off if the attachment speaks
    for itself."""
    await client.upload_and_send_attachment(room_id, data, content_type, filename, caption=caption)


@mcp.tool
async def update_profile(display_name: str | None = None, avatar: Base64Bytes | None = None) -> None:
    """Update the Matrix profile. Set display_name and/or avatar (image bytes)."""
    if display_name is not None:
        await client.set_display_name(display_name)
    if avatar is not None:
        kind = filetype.guess(avatar)
        if not kind:
            raise ValueError("Could not identify avatar image type")
        await client.upload_avatar(avatar, kind.mime, f"avatar.{kind.extension}")


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
