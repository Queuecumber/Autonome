"""Signal MCP server — tool interface over SignalClient.

Exposes domain-level actions (send message, react, receipts, typing)
as MCP tools for the agent to call.
"""

from fastmcp import FastMCP

from adapters.signal.model import SignalClient


def create_mcp(client: SignalClient) -> FastMCP:
    """Create the Signal MCP server with tools and resources backed by the client."""
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

    # ── Resources ─────────────────────────────────────────

    @mcp.resource("signal://attachments/{attachment_id}")
    async def get_attachment(attachment_id: str) -> bytes:
        """Retrieve a Signal attachment by ID."""
        return await client.fetch_attachment(attachment_id)

    return mcp
