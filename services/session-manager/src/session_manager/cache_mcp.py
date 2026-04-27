"""MCP server hosting the session-manager binary cache as resources.

Tools across the platform that produce binary content (images, files,
audio) get their bytes persisted into the BinaryStore and exposed here
under `pointer://...` URIs. Other clients access them via the standard
MCP resource protocol (resources/list, resources/read); the orchestrator
bridges those verbs through to the LLM as tool calls.

Runs in-process alongside the orchestrator. Shares its BinaryStore
instance via the `binary_store` module-level reference, set by main.py
at startup."""

import logging

from fastmcp import FastMCP

from session_manager.binaries import BinaryStore

logger = logging.getLogger(__name__)

POINTER_PREFIX = "pointer://"

# Set by main.py before the server starts.
binary_store: BinaryStore | None = None

mcp = FastMCP("session", instructions=(
    "Session-manager binary cache. Tools that produce binary content "
    "(images, files, audio) persist bytes here and surface `pointer://...` "
    "URIs in their results. Read a pointer with the orchestrator's "
    "resources_read tool; pass pointers as arguments to other tools that "
    "accept binary content (the orchestrator resolves them transparently)."
))


@mcp.resource("pointer://{name}")
async def read_resource(name: str) -> bytes:
    """MCP resource view of the binary cache. Reading returns raw bytes;
    fastmcp infers the mime type from the content."""
    assert binary_store is not None, "cache_mcp.binary_store not initialized"
    content, _ = binary_store.load(name)
    return content
