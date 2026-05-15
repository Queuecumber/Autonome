"""Orchestrator-internal MCP server."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from fastmcp import FastMCP
from fastmcp.resources import ResourceContent, ResourceResult
from mcp.types import EmbeddedResource

from session_manager.binaries import BinaryStore

if TYPE_CHECKING:
    from session_manager.orchestrator import SessionOrchestrator

logger = logging.getLogger(__name__)

POINTER_PREFIX = "pointer://"

# Set by main.py before the server starts.
binary_store: BinaryStore | None = None
orchestrator: "SessionOrchestrator | None" = None


@dataclass
class ResourceEntry:
    server: str
    uri: str
    name: str | None = None
    description: str | None = None
    mimeType: str | None = None


@dataclass
class ResourceTemplateEntry:
    server: str
    uriTemplate: str
    name: str | None = None
    description: str | None = None
    mimeType: str | None = None

mcp = FastMCP("session", instructions=(
    "Platform-internal resources and bridge tools. Use "
    "`resources_read` to load any MCP resource (pointer://, mxc://, etc.) into "
    "context."
))


# ── pointer:// cache ─────────────────────────────────────


def _require_orchestrator() -> "SessionOrchestrator":
    if orchestrator is None:
        raise RuntimeError("platform_mcp.orchestrator not initialized")
    return orchestrator


def _require_binary_store() -> BinaryStore:
    if binary_store is None:
        raise RuntimeError("platform_mcp.binary_store not initialized")
    return binary_store


@mcp.resource("pointer://{name}")
async def read_pointer(name: str) -> ResourceResult:
    """Read a binary by pointer name from the platform's content cache.

    Args:
        name: The pointer's name component (everything after `pointer://`).

    Returns:
        A `ResourceResult` with the stored mime, recovered by the BinaryStore
        at load time.

    Raises:
        FileNotFoundError: If no content matches `name` (it may have been
            garbage-collected).
    """
    content, mime = _require_binary_store().load(name)
    return ResourceResult([ResourceContent(content, mime_type=mime or "application/octet-stream")])


# ── Resource bridge tools ────────────────────────────────


@mcp.tool
async def resources_list(scheme: str | None = None) -> list[ResourceEntry]:
    """List concrete resources currently available across all MCP servers.

    Aggregates each connected server's `resources/list`. Use to discover
    what's addressable right now before reading. For families of resources
    addressable by template (e.g. `pointer://{name}`), use
    `resources_template_list` instead.

    Args:
        scheme: If set, only return resources whose URI uses this scheme
            (e.g. `mxc`, `workspace`).

    Returns:
        One `ResourceEntry` per resource.
    """
    orch = _require_orchestrator()
    out: list[ResourceEntry] = []
    for conn in orch.mcp_connections.values():
        for r in await conn.list_resources():
            uri = str(getattr(r, "uri", ""))
            if scheme and urlparse(uri).scheme.lower() != scheme.lower():
                continue
            out.append(ResourceEntry(
                server=conn.name,
                uri=uri,
                name=getattr(r, "name", None),
                description=getattr(r, "description", None),
                mimeType=getattr(r, "mimeType", None),
            ))
    return out


@mcp.tool
async def resources_template_list() -> list[ResourceTemplateEntry]:
    """List resource URI templates exposed across all MCP servers.

    Templates describe families of addressable resources (e.g.
    `pointer://{name}`, `mxc://{server}/{media_id}`). Use to learn which
    schemes are available and how to construct concrete URIs for them.

    Returns:
        One `ResourceTemplateEntry` per template.
    """
    orch = _require_orchestrator()
    out: list[ResourceTemplateEntry] = []
    for conn in orch.mcp_connections.values():
        for t in await conn.list_resource_templates():
            out.append(ResourceTemplateEntry(
                server=conn.name,
                uriTemplate=getattr(t, "uriTemplate", ""),
                name=getattr(t, "name", None),
                description=getattr(t, "description", None),
                mimeType=getattr(t, "mimeType", None),
            ))
    return out


@mcp.tool
async def resources_read(uri: str) -> list[EmbeddedResource]:
    """Read a resource by URI and load its content into context.

    Use when you actually need the bytes — e.g. viewing an image, reading
    a document.

    Args:
        uri: Full resource URI (e.g. `pointer://5-photo.jpg`,
            `mxc://server/abc`). Can come from `resources_list` or be
            built from a template returned by `resources_template_list`.

    Returns:
        The resource's content. Images come back so you can see them,
        text comes back as text.

    Raises:
        ValueError: If no MCP server is registered for the URI's scheme.
    """
    orch = _require_orchestrator()
    scheme = urlparse(uri).scheme.lower()
    conn = orch._scheme_to_mcp.get(scheme)
    if conn is None:
        raise ValueError(f"No MCP server registered for scheme {scheme!r}")
    contents = await conn.read_resource(uri)
    return [EmbeddedResource(type="resource", resource=c) for c in contents]
