"""Matrix adapter — MCP interface + inbound event forwarding."""

import asyncio
import logging
import os

import filetype
import httpx
from fastmcp import FastMCP
from fastmcp.resources import ResourceContent, ResourceResult
from matrix_adapter.model import (
    MatrixClient,
    Message,
    MessageRelation,
    Pin,
    Reaction,
    Redaction,
    Sender,
    UserProfile,
)
from pydantic import Base64Bytes

logger = logging.getLogger(__name__)

client: MatrixClient
session_manager_url: str
_http: httpx.AsyncClient
_event_queue: asyncio.Queue

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

When a message has attachments, the metadata includes `mxc://...` URIs. These are MCP
resources — read them with `resources_read` to see the content.

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

## Redaction

User's may "redact" or delete a message, when this happens you will recieve a redaction event the references
the message ID of the message which was redacted. You will still have the original message in your memory
but you should note that the message was redacted and refrain from mentioning it unless there is a good
reason to do otherwise.

## Edits

User's may edit a message, in this case you will receive a message containing the new (replaced) content
of the message as well as an additional "relation" field that will show the event_id of the message which
is edited and a relation_type of `m.replace`. As usual you may have all "versions" of an edited message in
context however you should treat the most recent version as authoritative unless there is a good reason to do otherwise.

## Threads and Replies

Users may organize responses into threads. Similar to edits, you will receive a new message
with new content and a "relation" field. This will show the event_id of the *thread root* along with
a relation_type of `m.thread`. You can reply to threads by including this same *thread root* event id
in your message using the `related_event_id` field and specifying the `m.thread` relationship type.

You should keep in mind which messages are organized into which threads in order to keep track of
the different information directly related to each thread. Threads will typically be focused around
a particular topic or discussion. Be careful not to respond to the wrong thread when composing a reply.

Users may also reply directly to a message to clarify what they are referring to. If this
happens the new message will have a relation_type of `m.in_reply_to` along with the event_id of a
previous message. You can do the same by specifying the event id of the message you want to reply
directly to in the `related_event_id` field and specifying the same `m.in_reply_to` relationship
type.

Good practice is to prefer threads for group messages and replies for direct messages if the context
isn't obvious but be flexible and go with the flow. If something makes sense as a threaded response or
a reply or if it just seems like the user wants to go in a different direction, you can bend the rules.

## Pinned Messages

When someone pins a message in a room you will receive a `pin` event with the event id
of the newly pinned message; when someone unpins one, you will receive an `unpin` event with
the same shape. Each room message you receive also carries the current pinned list in its metadata
as `pinned_event_ids`.

The pinned messages themselves may already be in your conversation history. If not, fetch
them with `get_message(room_id, event_id)`. Treat pins as a signal that the message matters
to the room — worth remembering when it looks like a rule, an announcement, or context the
room expects everyone to know.

You can pin and unpin messages yourself with the `pin_message` and `unpin_message` tools.
Use this when something is genuinely worth keeping at the top of the room — a decision, a
recurring reference, a rule. Don't pin casually; the pinned list is shared room state and
everyone sees what you put there.



"""
))


@mcp.tool
async def send_message(room_id: str, text: str, action: str | None = None, related_event_id: str | None = None, relationship_type: str | None = None) -> str:
    """Send a text message to a Matrix room.

    Stops the typing indicator automatically. Markdown in `text` and
   `action` renders as HTML in supporting clients.

   The message you are going to send *could* be related to another event (for example a thread
   or a reply). To manage this relationship use the related_event_id and the relationship_type
   arguments. Refer to the server instructions for more details.

   Some common relationship types:
    m.thread: reply in a thread, the related_event_id should be the *root* of the thread
    m.in_reply_to: reply to a single message, calling out the message you are replying to for clarity

    Args:
        room_id: The Matrix room (e.g. `!abc:example.com`) — usually from
            an incoming event's metadata.
        text: The message body.
        action: Optional roleplay action (gesture, expression, thought)
            rendered separately from the body.
        related_event_id: Optional event id of a previous event that is related to this message
        relationship_type: Optional type of the relationship between this event and the previous event

    Returns:
      The event_id of the message which was sent

    Raises:
        RuntimeError: If the homeserver rejects the send.
    """
    try:
        await client.send_typing(room_id, typing=False)
    except Exception as e:
        # Typing indicator is best-effort — never let it block the actual send.
        logger.warning("stop-typing before send_message failed: %r", e)

    if action:
        text = f"> {action}\n\n{text}"

    if (related_event_id is None) != (relationship_type is None):
        raise ValueError("related_event_id and relationship_type must both be set or both omitted")

    relation = (
        MessageRelation(related_event_id=related_event_id, relation_type=relationship_type)
        if related_event_id else None
    )
    return await client.send_message(room_id, text, relation=relation)


@mcp.tool
async def redact_message(room_id: str, event_id: str, reason: str | None = None) -> None:
    """Redact (delete) a message.

    Use to redact a message by ID, this should be the ID of a message which you have sent.

    Args:
        room_id: The matrix room.
        event_id: The event ID of the message to redact.
        reason: Optional reason for why the message is being redacted.

    Raises:
        RuntimeError: If the homeserver rejects the redaction.
    """
    await client.redact_message(room_id, event_id, reason)


@mcp.tool
async def edit_message(room_id: str, event_id: str, text: str, action: str | None = None) -> str:
    """Edit a message.

    Replaces the contents of a previous message you sent with the new (optional action) and text.

    Args:
        room_id: The matrix room.
        event_id: The ID of the message to edit.
        text: The new text.
        action: Optional new action.

    Returns:
        The event_id of the new message which replaces the old message.

    Raises:
        RuntimeError: If the homeserver rejects the edit.
    """
    try:
        await client.send_typing(room_id, typing=False)
    except Exception as e:
        # Typing indicator is best-effort — never let it block the actual send.
        logger.warning("stop-typing before edit_message failed: %r", e)

    if action:
        text = f"> {action}\n\n{text}"

    relation = MessageRelation(related_event_id=event_id, relation_type="m.replace")
    return await client.send_message(room_id, text, relation=relation)


@mcp.tool
async def react(room_id: str, event_id: str, emoji: str) -> None:
    """React to a message with an emoji.

    Args:
        room_id: The Matrix room.
        event_id: The message being reacted to (from `message_id` in
            an incoming event's metadata).
        emoji: The emoji to react with.

    Raises:
        RuntimeError: If the homeserver rejects the reaction.
    """
    await client.send_reaction(room_id, event_id, emoji)


@mcp.tool
async def pin_message(room_id: str, event_id: str) -> None:
    """Pin a message in a Matrix room.

    Adds `event_id` to the room's pinned-messages list.

    Args:
        room_id: The Matrix room.
        event_id: The message to pin.

    Raises:
        RuntimeError: If the homeserver rejects the change (e.g. insufficient
            power level).
    """
    await client.pin_message(room_id, event_id)


@mcp.tool
async def unpin_message(room_id: str, event_id: str) -> None:
    """Unpin a message in a Matrix room.

    Removes `event_id` from the room's pinned-messages list.

    Args:
        room_id: The Matrix room.
        event_id: The message to unpin.

    Raises:
        RuntimeError: If the homeserver rejects the change (e.g. insufficient
            power level).
    """
    await client.unpin_message(room_id, event_id)


@mcp.tool
async def read_receipt(room_id: str, event_id: str) -> None:
    """Send a read receipt for a message.

    Marks a message as read in this user's view of the room. Conventionally
    sent before composing a response so the sender sees acknowledgment.

    Args:
        room_id: The Matrix room.
        event_id: The message being acknowledged.
    """
    await client.send_read_receipt(room_id, event_id)


@mcp.tool
async def typing_indicator(room_id: str, stop: bool = False) -> None:
    """Show or hide the typing indicator in a Matrix room.

    Other users see this user's name as "typing…" while active.
    `send_message` automatically stops the indicator on send.

    Args:
        room_id: The Matrix room.
        stop: Hide the indicator instead of showing it.
    """
    await client.send_typing(room_id, typing=not stop)


@mcp.tool
async def get_message(room_id: str, event_id: str) -> Message:
    """Fetch a Matrix message by id.

    Args:
        room_id: The Matrix room the message lives in.
        event_id: The event id to look up.

    Returns:
        The message with its sender, room, text, attachments, and relation.

    Raises:
        RuntimeError: If the homeserver can't find the event, the event
            isn't a message (e.g. a reaction or state event), or the message
            is encrypted and we don't have the decryption key.
    """
    return await client.get_message(room_id, event_id)


@mcp.tool
async def get_user_profile(user_id: str, room_id: str | None = None) -> UserProfile:
    """Look up a Matrix user's profile.

    Use this when you see a user_id you don't recognize, when you want to
    show what someone looks like (avatar), or when reasoning about who has
    moderation authority in a room. `trust_status` reflects the verification
    state of the user's devices — useful when deciding how much to trust a
    message at face value.

    Args:
        user_id: The Matrix user id (e.g. `@alice:example.com`).
        room_id: Optional. If provided, includes the user's power level in
            that room.

    Returns:
        Profile with `display_name`, `avatar_url` (an `mxc://` URI you can
        read with `resources_read` to see the image), `trust_status` (one of
        `verified`, `partial`, `unverified`, `blacklisted`, `unknown`), and
        `power_level` if a room was given.

    Raises:
        RuntimeError: If the homeserver rejects the lookup.
    """
    return await client.get_user_profile(user_id, room_id)


@mcp.tool
async def get_room_members(room_id: str) -> list[Sender]:
    """List the people currently in a Matrix room.

    Each event already carries a `member_count` in its metadata — use that
    as your first signal for how public a room is. Call this tool to see
    who specifically is there when:

    - You're about to share anything private or personal and the room has
      more than two members
    - Someone new joined the conversation and you don't recognize them
    - A `member_count` you noticed before has changed

    *Important*: in a group setting, be careful about what information
    you're sharing.

    Args:
        room_id: The Matrix room.

    Returns:
        One entry per member with `id` and `name`.
    """
    return client.get_room_members(room_id)


@mcp.resource("mxc://{server}/{media_id}{?k,iv,hash,mime}")
async def mxc_resource(
    server: str, media_id: str,
    k: str = "", iv: str = "", hash: str = "", mime: str = "",
) -> ResourceResult:
    """Serve `mxc://...` URIs as MCP resources."""
    data = await client.download_attachment(server, media_id, k=k, iv=iv, hash=hash)
    return ResourceResult([ResourceContent(data, mime_type=mime or "application/octet-stream")])


@mcp.tool
async def send_attachment(
    room_id: str,
    data: Base64Bytes,
    filename: str,
    content_type: str = "application/octet-stream",
    caption: str | None = None,
) -> None:
    """Send a file attachment to a Matrix room.

    Args:
        room_id: The Matrix room.
        data: The attachment.
        filename: Name the recipient will see in their client (e.g.
            `photo.jpg`, `report.pdf`). Don't put a caption here.
        content_type: MIME type of the attachment. Default
            `application/octet-stream`.
        caption: Optional message displayed alongside the attachment.
            Leave off if the attachment speaks for itself.

    Raises:
        RuntimeError: If upload or send fails (size limit, auth, etc.).
    """
    await client.upload_and_send_attachment(room_id, data, content_type, filename, caption=caption)


@mcp.tool
async def update_profile(display_name: str | None = None, avatar: Base64Bytes | None = None) -> None:
    """Update this Matrix account's profile.

    Either or both fields can be set. Profile changes are visible to
    everyone in shared rooms.

    Args:
        display_name: New display name. Omit to leave unchanged.
        avatar: New profile picture. Omit to leave unchanged.

    Raises:
        ValueError: If the avatar isn't a recognizable image format.
        RuntimeError: If the homeserver rejects the update.
    """
    if display_name is not None:
        await client.set_display_name(display_name)
    if avatar is not None:
        kind = filetype.guess(avatar)
        if not kind:
            raise ValueError("Could not identify avatar image type")
        await client.upload_avatar(avatar, kind.mime, f"avatar.{kind.extension}")


# ── Inbound event forwarding ─────────────────────────────

async def on_message(msg: Message | Reaction | Redaction | Pin) -> None:
    logger.info("Received: %s", msg)
    await _event_queue.put(msg.to_event())


async def _forward_events() -> None:
    """Drain queued events to session-manager. Retries each event with
    exponential backoff (1s–60s) until it sticks — so a session-manager
    blip doesn't drop messages."""
    while True:
        payload = await _event_queue.get()
        backoff = 1
        while True:
            try:
                await _http.post(f"{session_manager_url}/event", json=payload)
                break
            except Exception as e:
                logger.warning("Event delivery failed (%s); retry in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
        _event_queue.task_done()


# ── Entrypoint ───────────────────────────────────────────

async def main():
    global client, session_manager_url, _http, _event_queue

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
    _event_queue = asyncio.Queue()

    await client.login()

    try:
        await asyncio.gather(
            client.listen(on_message),
            mcp.run_async(transport="http", host="0.0.0.0", port=mcp_port),
            _forward_events(),
        )
    finally:
        await _http.aclose()
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
