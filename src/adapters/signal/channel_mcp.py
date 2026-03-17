"""Signal channel MCP server: attachments, reactions, typing indicator."""

from collections import defaultdict

from fastmcp import FastMCP


class StagedAttachments:
    """In-memory attachment staging queue. Safe for single-threaded async (asyncio)."""

    def __init__(self):
        self._queues: dict[str, list[dict[str, str]]] = defaultdict(list)

    def stage(self, conversation_id: str, file_path: str, mime_type: str) -> None:
        self._queues[conversation_id].append({
            "file_path": file_path,
            "mime_type": mime_type,
        })

    def drain(self, conversation_id: str) -> list[dict[str, str]]:
        items = self._queues.pop(conversation_id, [])
        return items


# Global instance — shared between MCP tools and the adapter's message loop
staged_attachments = StagedAttachments()

# Callback for immediate sends — set by the adapter at startup
_send_callback = None


def set_send_callback(callback) -> None:
    """Register a callback for immediate attachment sends. Called by adapter."""
    global _send_callback
    _send_callback = callback


def create_channel_mcp(heartbeat_handler=None) -> FastMCP:
    """Create the Signal channel MCP server with all tools.

    Args:
        heartbeat_handler: async callable that processes a heartbeat.
            If provided, a POST /heartbeat custom route is registered.
    """
    mcp = FastMCP("signal-channel")

    if heartbeat_handler is not None:
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        @mcp.custom_route("/heartbeat", methods=["POST"])
        async def heartbeat_route(request: Request) -> JSONResponse:
            result = await heartbeat_handler()
            return JSONResponse({"status": "ok", "response": result})

    @mcp.tool
    def stage_attachment(conversation_id: str, file_path: str, mime_type: str) -> str:
        """Queue a file to send with the next text response. Multiple files can be staged."""
        staged_attachments.stage(conversation_id, file_path, mime_type)
        return f"Staged {file_path} for conversation {conversation_id}"

    @mcp.tool
    async def send_attachment(
        conversation_id: str, file_path: str, mime_type: str, caption: str | None = None
    ) -> str:
        """Send a file immediately to the current conversation."""
        if _send_callback is None:
            return "Error: send callback not registered"
        await _send_callback(conversation_id, file_path, mime_type, caption)
        return f"Sent {file_path} to conversation {conversation_id}"

    @mcp.tool
    async def react(conversation_id: str, message_id: str, emoji: str) -> str:
        """React to a message with an emoji."""
        if _send_callback is None:
            return "Error: send callback not registered"
        return f"Reacted with {emoji} to {message_id} (stub — signal-cli wiring TODO)"

    @mcp.tool
    async def set_typing(conversation_id: str, enabled: bool) -> str:
        """Show or hide the typing indicator."""
        status = "started" if enabled else "stopped"
        return f"Typing {status} for {conversation_id} (stub — signal-cli wiring TODO)"

    return mcp
