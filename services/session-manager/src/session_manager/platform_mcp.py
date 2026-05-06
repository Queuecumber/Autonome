"""Orchestrator-internal MCP server."""

import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fastmcp import FastMCP
from mcp.types import EmbeddedResource

from session_manager.binaries import BinaryStore

if TYPE_CHECKING:
    from session_manager.orchestrator import SessionOrchestrator

logger = logging.getLogger(__name__)

POINTER_PREFIX = "pointer://"

# Set by main.py before the server starts.
binary_store: BinaryStore | None = None
orchestrator: "SessionOrchestrator | None" = None

mcp = FastMCP("session", instructions=(
    "Platform-internal resources and bridge tools. Use "
    "`resources_read` to load any URI (pointer://, mxc://, etc.) into "
    "context, or pass URIs as arguments to other tools that accept "
    "binary content (the platform resolves them transparently)."
))


# ── pointer:// cache ─────────────────────────────────────


@mcp.resource("pointer://{name}")
async def read_pointer(name: str) -> bytes:
    """Read a binary by pointer name from the platform's content cache.

    Args:
        name: The pointer's name component (everything after `pointer://`).

    Returns:
        Raw bytes. The content type is inferred from the bytes by fastmcp.

    Raises:
        FileNotFoundError: If no content matches `name` (it may have been
            garbage-collected).
    """
    assert binary_store is not None, "platform_mcp.binary_store not initialized"
    content, _ = binary_store.load(name)
    return content


# ── Resource bridge tools ────────────────────────────────


@mcp.tool
async def resources_list() -> str:
    """List concrete resources currently available across all MCP servers.

    Aggregates each connected server's `resources/list`. Use to discover
    what's addressable right now before reading. For families of resources
    addressable by template (e.g. `pointer://{name}`), use
    `resources_template_list` instead.

    Returns:
        A JSON array of `{server, uri, name, description, mimeType}` entries.
    """
    assert orchestrator is not None, "platform_mcp.orchestrator not initialized"
    out: list[dict[str, Any]] = []
    for conn in orchestrator.mcp_connections.values():
        try:
            resources = await conn.list_resources()
        except Exception as e:
            logger.debug("resources_list: %s failed: %r", conn.name, e)
            continue
        for r in resources:
            out.append({
                "server": conn.name,
                "uri": str(getattr(r, "uri", "")),
                "name": getattr(r, "name", None),
                "description": getattr(r, "description", None),
                "mimeType": getattr(r, "mimeType", None),
            })
    return json.dumps(out)


@mcp.tool
async def resources_template_list() -> str:
    """List resource URI templates exposed across all MCP servers.

    Templates describe families of addressable resources (e.g.
    `pointer://{name}`, `mxc://{server}/{media_id}`). Use to learn which
    schemes are available and how to construct concrete URIs for them.

    Returns:
        A JSON array of `{server, uriTemplate, name, description, mimeType}`
        entries.
    """
    assert orchestrator is not None, "platform_mcp.orchestrator not initialized"
    out: list[dict[str, Any]] = []
    for conn in orchestrator.mcp_connections.values():
        try:
            templates = await conn.list_resource_templates()
        except Exception as e:
            logger.debug("resources_template_list: %s failed: %r", conn.name, e)
            continue
        for t in templates:
            out.append({
                "server": conn.name,
                "uriTemplate": getattr(t, "uriTemplate", None),
                "name": getattr(t, "name", None),
                "description": getattr(t, "description", None),
                "mimeType": getattr(t, "mimeType", None),
            })
    return json.dumps(out)


@mcp.tool
async def resources_read(uri: str) -> list[EmbeddedResource]:
    """Read a resource by URI and load its content into your input.

    Use when you actually need the bytes in context — e.g. viewing an
    image, reading a document. When you only need to forward a binary to
    another tool, that tool will accept the pointer URI directly so
    no read is necessary.

    Args:
        uri: Full resource URI (e.g. `pointer://5-photo.jpg`,
            `mxc://server/abc`). Can come from `resources_list` or be
            built from a template returned by `resources_template_list`.

    Returns:
        The resource's content. Images come back so you can see them,
        text comes back as text, other binaries come back as descriptors.

    Raises:
        ValueError: If the URI has no scheme, or no MCP server is
            registered for that scheme.
    """
    assert orchestrator is not None, "platform_mcp.orchestrator not initialized"
    scheme = urlparse(uri).scheme.lower()
    if not scheme:
        raise ValueError(f"URI has no scheme: {uri!r}")
    conn = orchestrator._scheme_to_mcp.get(scheme)
    if conn is None:
        raise ValueError(f"No MCP server registered for scheme {scheme!r}")
    contents = await conn.read_resource(uri)
    return [EmbeddedResource(type="resource", resource=c) for c in contents]
