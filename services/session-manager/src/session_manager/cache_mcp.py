"""MCP server hosting the session-manager binary cache as resources.

Tools across the platform that produce binary content (images, files,
audio) get their bytes persisted into the BinaryStore and exposed here
under `pointer://...` URIs. The agent receives these URIs in tool output
metadata. To re-view a binary later, the agent calls the `read_binary`
tool with the pointer URI.

Runs in-process alongside the orchestrator. Shares its BinaryStore
instance via the `binary_store` module-level reference, set by main.py
at startup."""

import base64
import logging

from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from session_manager.binaries import BinaryStore

logger = logging.getLogger(__name__)

POINTER_PREFIX = "pointer://"

# Set by main.py before the server starts.
binary_store: BinaryStore | None = None

mcp = FastMCP("session", instructions=(
    "Session-manager binary cache. When other tools return binary content "
    "(images, files, audio) they show up here as `pointer://...` URIs in "
    "tool result metadata. Use `read_binary` to re-load a binary into your "
    "current input — useful when an image scrolled out of context and you "
    "want to look at it again. Pointers can also be passed as arguments to "
    "tools that accept binary content; the orchestrator resolves them "
    "transparently before dispatching."
))


def _strip_prefix(pointer: str) -> str:
    return pointer[len(POINTER_PREFIX):] if pointer.startswith(POINTER_PREFIX) else pointer


@mcp.tool
async def read_binary(pointer: str) -> ImageContent | TextContent:
    """Re-load a previously stored binary into your input.

    For images: returns the image content so you can see it again.
    For other types: returns a text descriptor (the bytes stay on disk).
    Pointers look like `pointer://5-photo.jpg` and appear in tool result
    metadata. The `pointer://` prefix is optional."""
    assert binary_store is not None, "cache_mcp.binary_store not initialized"
    name = _strip_prefix(pointer)
    content, mime = binary_store.load(name)
    if mime.startswith("image/"):
        return ImageContent(
            type="image",
            data=base64.b64encode(content).decode(),
            mimeType=mime,
        )
    return TextContent(
        type="text",
        text=f"[binary {name} ({mime}, {len(content)} bytes) — non-visual]",
    )


@mcp.resource("pointer://{name}")
async def read_resource(name: str) -> bytes:
    """MCP resource view of the binary cache. Reading returns raw bytes;
    fastmcp infers the mime type from the content. The `read_binary` tool
    is the LLM-facing entrypoint — this resource decorator exists for
    protocol completeness so resources/list and resources/read work for
    any client speaking MCP natively."""
    assert binary_store is not None, "cache_mcp.binary_store not initialized"
    content, _ = binary_store.load(name)
    return content
