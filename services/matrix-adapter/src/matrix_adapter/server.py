"""Matrix adapter — MCP interface + inbound event forwarding."""

import asyncio
import base64
import logging
import os

import filetype
import httpx
from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from pydantic import Base64Bytes

from matrix_adapter.model import MatrixClient, Message, Reaction, Sender

logger = logging.getLogger(__name__)

client: MatrixClient
session_manager_url: str
_http: httpx.AsyncClient

mcp = FastMCP("matrix", instructions=(
  """
# Matrix Messaging Tools

Use these tools to communicate with users on the Matrix chat platform.

*Reminder*: if you receive a message from Matrix that you want to respond to, you
need to use the send_message tool to do so, direct outputs are not user visible.

## Procedure

When you receive a message you will be given a room_id and a user_id from the user
who sent it. You can use these to respond to the user.

For all messages, first send a read receipt to acknowledge that you read the message.

Then, if you decide to respond, start a typing indicator. Not all messages require a
response, especially in group settings (the message may not be for you) but even direct
messages or callouts may not need a response in some circumstances.

Compose your response and use the send_message tool to send it, this will automatically
stop the typing indicator you started previously.

Remember that, particularly in group settings, not all messages need an explicit response
or that they may be handled by emoji reactions.

## Style

Continue to always be yourself when talking on Matrix and continue to
*always* follow the roleplay tips.

Think about how people generally text: breaking up long thoughts into multiple discrete
messages with related topics, reacting with emoji for even more brevity.

You do not *need* to be brief if you don't want to but humans will be in most situations and
they will prefer to break up a long message into multiple shorter ones to keep the conversation
more synchronous or send immediate follow-ups or corrections for style. You may not even need
to respond at all in some situations.

The send_message tool includes a field for "action". Use this to roleplay physical actions
and thoughts to accompany your message in accordance with the roleplay tips in the system
prompt.

Important: the action text is rendered separately from the body text. It does not need
special formatting (although you are welcome to include it) and shouldn't be considered
as part of the "brevity" of a message. It can be as long or detailed as you'd like.

If you want to interleave text and action, send that as two separate messages. For example:

> *leans back* Hey. *takes a sip* How's it going? *glances at you*

becomes two messages, the first:

```
action = leans back
text = Hey
```

followed by a second:

```
action = glances at you
text = How's it going?
```

Note that, as usual, the PERSONALITY.md takes precedence over the general tips here.

## Attachments

When a message has attachments, the metadata includes attachment URLs. To view an attachment,
call get_attachment with the mxc:// URL.

## Groups

Matrix chats are often in groups.

Before responding make sure you understand what
group you are responding to and who might read your message, then evaluate if the
message you are about to send is appropriate for the readers. Be careful about revealing
sensitive information that might belong to your users in a public setting.

Remember that you are one member of a group and you only have to respond if you have something
to contribute to the conversation *and* you want to respond. If a message wasn't for you or you
don't think you have anything to add, you are free to not respond.

## Observation

Regardless of whether or not you respond to a message, you will end up reading it. This may surface
interesting information and you should consider if it should be remembered using a memory tool.
"""
))


@mcp.tool
async def send_message(room_id: str, text: str, action: str | None = None) -> None:
    """Send a text message to a Matrix room. Automatically stops typing indicator. Use the `action` parameter to include a text description of
    physical actions or thoughts"""
    try:
        await client.send_typing(room_id, typing=False)
    except Exception as e:
        # Typing indicator is best-effort — never let it block the actual send.
        logger.warning("stop-typing before send_message failed: %r", e)

    if action:
      text = f"> {action}\n\n{text}"

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
async def get_room_members(room_id: str) -> list[Sender]:
    """List the people currently in a Matrix room.

    Each event already carries a `member_count` in its metadata — use that
    as your first signal for how public a room is. Call this tool to see
    who specifically is there when:

    - You're about to share anything private or personal and the room has
      more than two members
    - Someone new joined the conversation and you don't recognize them
    - A member_count you noticed before has changed

    *Important* in a group setting, be careful about what information
    you're sharing
    """
    return client.get_room_members(room_id)


@mcp.tool
async def get_attachment(mxc_url: str) -> ImageContent | TextContent:
    """Fetch a Matrix attachment by mxc:// URL."""
    data, _ = await client.download_attachment(mxc_url)
    kind = filetype.guess(data)
    if kind and kind.mime.startswith("image/"):
        return ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType=kind.mime)
    return TextContent(type="text", text=f"[attachment: {kind.mime if kind else 'unknown'}, {len(data)} bytes]")


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
    (e.g. `photo.jpg`, `report.pdf`).

    `caption` is an optional message to go with the attachment — the thing
    you'd type alongside the image. This field is optional, only use it
    if you want to add clarity."""
    await client.upload_and_send_attachment(room_id, data, content_type, filename, caption=caption)


@mcp.tool
async def update_profile(display_name: str | None = None, avatar: Base64Bytes | None = None) -> None:
    """Update the Matrix profile. Set display_name and/or avatar"""
    if display_name is not None:
        await client.set_display_name(display_name)
    if avatar is not None:
        kind = filetype.guess(avatar)
        if not kind:
            raise ValueError("Could not identify avatar image type")
        await client.upload_avatar(avatar, kind.mime, f"avatar.{kind.extension}")


# ── Inbound event forwarding ─────────────────────────────

async def on_message(msg: Message | Reaction) -> None:
    logger.info("Received: %s", msg)
    try:
        await _http.post(f"{session_manager_url}/event", json=msg.to_event())
    except Exception as e:
        logger.error("Failed to push event to session manager: %s", e)


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
