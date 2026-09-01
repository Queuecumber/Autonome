"""MCP connection management — tool discovery, execution, and content conversion."""

import asyncio
import base64
import copy
import io
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx
import exifread
import exifread.utils
import jsonpath
import jsonref
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import METHOD_NOT_FOUND

from session_manager.binaries import BinaryStore

logger = logging.getLogger(__name__)


def _is_method_not_found(err: McpError) -> bool:
    """Did the server reject the call as an unimplemented method?"""
    return getattr(getattr(err, "error", None), "code", None) == METHOD_NOT_FOUND


try:  # 1.x transport: takes headers (and auth/timeout) directly
    from mcp.client.streamable_http import (  # type: ignore[attr-defined]
        streamablehttp_client as _legacy_http_client,
    )
except ImportError:  # pragma: no cover - removed in 2.x
    _legacy_http_client = None

try:  # 2.x transport: configuration moved onto a caller-supplied httpx client
    from mcp.client.streamable_http import (  # type: ignore[attr-defined]
        streamable_http_client as _modern_http_client,
    )
except ImportError:  # pragma: no cover - absent in older 1.x
    _modern_http_client = None


@asynccontextmanager
async def _open_transport(url: str, headers: dict[str, str] | None):
    """Open a streamable-HTTP transport across both SDK generations.

    These are not two spellings of one function. 1.x takes `headers`
    directly; 2.x dropped that parameter and expects a preconfigured httpx
    client instead. Aliasing one to the other silently drops auth, so both
    shapes are handled explicitly.
    """
    if _legacy_http_client is not None:
        async with _legacy_http_client(url, headers=headers) as transport:
            yield transport
        return

    if _modern_http_client is None:  # pragma: no cover - neither available
        raise RuntimeError("mcp SDK exposes no streamable-HTTP client")

    client = httpx.AsyncClient(headers=headers or {})
    try:
        async with _modern_http_client(url, http_client=client) as transport:
            yield transport
    finally:
        await client.aclose()

POINTER_PREFIX = "pointer://"

# Mirrors workspace_fs/server.py's _is_text_type — kept duplicated to avoid
# coupling MCP servers to each other. If this list grows, sync both copies.
_TEXT_TYPES = {"application/json", "application/xml", "application/yaml", "application/x-yaml"}


def _is_text_type(mime: str) -> bool:
    return mime.startswith("text/") or mime in _TEXT_TYPES


# Select every schema node whose format marks it as binary content.
_BINARY_FINDER = jsonpath.compile(
    "$..[?(@.format == 'binary' or @.format == 'byte' or @.format == 'base64')]"
)

_BINARY_PARAM_DESCRIPTION = (
    "Resource URI (e.g. 'pointer://5-photo.jpg', 'mxc://server/abc') or "
    "raw base64 bytes."
)

URIResolver = Callable[[str], Awaitable[bytes]]


def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP tool to OpenAI Responses API function tool format."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema or {},
    }


@dataclass
class BinaryParam:
    """A binary-typed parameter in a tool schema.

    Holds the JSONPointer into the (inlined) schema where the binary node
    lives, and a cached JSONPath that translates that pointer into a
    selector over runtime tool-call arguments.
    """
    schema_pointer: jsonpath.JSONPointer

    @cached_property
    def args_matcher(self) -> jsonpath.JSONPath:
        """Translate the schema pointer into a JSONPath over args.

        Strips schema-only segments (properties, anyOf, oneOf, and the
        numeric variant indices they imply) and replaces container-shape
        keywords (items, additionalProperties) with `*` to match every
        runtime element.
        """
        parts = ["$"]
        for part in self.schema_pointer.parts:
            if part in ("properties", "anyOf", "oneOf"):
                continue
            if part.isnumeric():
                continue
            if part in ("items", "additionalProperties"):
                parts.append("*")
            else:
                parts.append(part)
        return jsonpath.compile(".".join(parts))


def inline_refs(schema: dict) -> dict:
    """Resolve every `$ref`, drop `$defs`, and break shared object identity
    so downstream JSONPath traversals don't dedupe ref-reuse sites.

    jsonref.replace_refs(proxies=False) still reuses the same Python dict
    for every ref to a given def; a JSON round-trip forces each occurrence
    to become its own dict.

    Self-referential schemas expand infinitely here and aren't supported.
    """
    resolved = json.loads(json.dumps(jsonref.replace_refs(schema, proxies=False)))
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
    return resolved


def rewrite_binary_params(schema: dict) -> list[BinaryParam]:
    """Find every binary-string node in the schema, rewrite each in place
    (strip the binary `format`, add a URI-or-bytes usage description), and
    return a list of BinaryParams for dispatch-time resolution.

    Expects refs already resolved via inline_refs().
    """
    params: list[BinaryParam] = []
    for match in _BINARY_FINDER.finditer(schema):
        patch = (
            jsonpath.JSONPatch()
            .replace(match.pointer().join("format"), "string")
            .add(match.pointer().join("description"), _BINARY_PARAM_DESCRIPTION)
        )
        patch.apply(schema)
        params.append(BinaryParam(match.pointer()))
    return params


async def resolve_uri_args(
    args: dict, params: list[BinaryParam], resolver: URIResolver,
) -> dict:
    """Replace URI-shaped values in binary-typed argument positions with
    the resolver's bytes (base64-encoded). Non-URI values pass through.
    The resolver raises if the URI's scheme isn't registered."""
    if not params:
        return args
    args = copy.deepcopy(args)
    for param in params:
        for match in param.args_matcher.finditer(args):
            val = match.value
            if not isinstance(val, str):
                continue
            scheme = urlparse(val).scheme
            if not scheme:
                continue
            content = await resolver(val)
            encoded = base64.b64encode(content).decode()
            jsonpath.JSONPatch().replace(match.pointer(), encoded).apply(args)
    return args


def _save_and_describe(store: BinaryStore, data_b64: str, mime_type: str) -> dict | None:
    """Persist base64 content and return the pointer metadata, or None on failure."""
    try:
        raw = base64.b64decode(data_b64)
        pointer_id = store.save(raw, mime_type)
        return {
            "id": f"{POINTER_PREFIX}{pointer_id}",
            "content_type": mime_type,
            "size": len(raw),
        }
    except Exception as e:
        logger.warning("Failed to persist binary (%s) to BinaryStore: %s", mime_type, e)
        return None


def _pointer_text(pointer: dict) -> dict:
    return {"type": "input_text", "text": json.dumps({"pointer": pointer}, ensure_ascii=False)}


def _describe_binary(data_b64: str, mime_type: str, store: BinaryStore | None) -> dict:
    """Persist bytes and return an input_text carrying pointer JSON. If no
    store is available the shape still carries content_type with no id,
    signaling 'binary present, no pointer to reference it by.'"""
    pointer = _save_and_describe(store, data_b64, mime_type) if store else None
    return _pointer_text(pointer or {"content_type": mime_type})


def _exif_summary(data: bytes) -> dict | None:
    """Extract a useful subset of EXIF from image bytes. Returns None if no
    EXIF is present. exifread supports JPEG/TIFF/HEIC/RAW; PNG and WebP
    have no EXIF, so this returns None on those (which is fine)."""
    tags = exifread.process_file(io.BytesIO(data), details=False)
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

    coords = exifread.utils.get_gps_coords(tags)
    if coords:
        lat, lon = coords
        out["gps"] = {"lat": round(lat, 6), "lon": round(lon, 6)}

    w = tags.get("EXIF ExifImageWidth") or tags.get("Image ImageWidth")
    h = tags.get("EXIF ExifImageLength") or tags.get("Image ImageLength")
    if w and h:
        out["dimensions"] = f"{w}x{h}"

    software = tags.get("Image Software")
    if software:
        out["software"] = str(software)

    return out or None


def mcp_content_to_openai(content_blocks: list, store: BinaryStore | None = None) -> list[dict]:
    """Convert MCP content blocks to OpenAI Responses API message content parts.

    Every binary gets persisted to the BinaryStore and produces an input_text
    part carrying the pointer JSON. Images additionally produce an input_image
    part so the model can see the bytes, and an EXIF summary text part when
    metadata is present.
    """
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append({"type": "input_text", "text": block.text})

        elif block.type == "image":
            parts.append(_describe_binary(block.data, block.mimeType, store))
            raw = base64.b64decode(block.data) if isinstance(block.data, str) else block.data
            exif = _exif_summary(raw)
            if exif:
                parts.append({"type": "input_text", "text": json.dumps(exif)})
            parts.append({
                "type": "input_image",
                # `detail` is not optional here: input_image without it fails
                # request validation outright.
                "detail": "auto",
                "image_url": f"data:{block.mimeType};base64,{block.data}",
            })

        elif block.type == "audio":
            # No audio variant in the Responses input schema — the pointer is
            # all we can hand back, same as video.
            parts.append(_describe_binary(block.data, block.mimeType, store))

        elif block.type == "resource":
            resource = getattr(block, "resource", None)
            blob = getattr(resource, "blob", None)
            text = getattr(resource, "text", None)
            mime = getattr(resource, "mimeType", None) or "application/octet-stream"
            if blob is not None:
                if mime.startswith("image/"):
                    raw = base64.b64decode(blob)
                    exif = _exif_summary(raw)
                    if exif:
                        parts.append({"type": "input_text", "text": json.dumps(exif)})
                    parts.append({
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": f"data:{mime};base64,{blob}",
                    })
                elif mime.startswith("video/"):
                    # No model host we target accepts video, so hand back the
                    # pointer rather than the bytes — she still learns a video
                    # arrived and can re-fetch it by URI.
                    parts.append(_describe_binary(blob, mime, store))
                elif mime.startswith("audio/"):
                    parts.append(_describe_binary(blob, mime, store))
                elif _is_text_type(mime):
                    raw = base64.b64decode(blob)
                    parts.append({"type": "input_text", "text": raw.decode("utf-8")})
                else:
                    raise ValueError(f"Cannot inline resource of type {mime!r}; pass the URI to a tool that handles it instead")
            elif text is not None:
                parts.append({"type": "input_text", "text": text})
            else:
                parts.append({"type": "input_text", "text": str(block)})

        else:
            parts.append({"type": "input_text", "text": str(block)})
    return parts



_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: str, where: str) -> str:
    """Substitute ${VAR} references from the environment.

    Raises on a missing variable rather than leaving the placeholder in
    place. A literal "${HA_TOKEN}" sent as a bearer token surfaces much
    later as an opaque 401 from the server, a long way from the mistake.
    """
    def replace(match: re.Match) -> str:
        name = match.group(1)
        try:
            return os.environ[name]
        except KeyError:
            raise ValueError(
                f"{where}: environment variable {name!r} is not set") from None

    return _ENV_REF.sub(replace, value)


def parse_server_spec(name: str, spec: Any) -> tuple[str, dict[str, str] | None]:
    """Normalize one `mcp_servers` entry into (url, headers).

    Accepts the plain-URL shorthand or a mapping::

        memory: http://memory-mcp:8001/mcp
        home_assistant:
          url: http://homeassistant.local:8123/mcp
          headers:
            Authorization: "Bearer ${HA_TOKEN}"

    Header values expand ${VAR} from the environment, so tokens come from
    a secret at runtime instead of being written into the config file.
    """
    if isinstance(spec, str):
        return spec, None
    if not isinstance(spec, dict):
        raise ValueError(
            f"mcp_servers.{name}: expected a URL string or a mapping, "
            f"got {type(spec).__name__}")

    url = spec.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError(f"mcp_servers.{name}: missing required 'url'")

    raw_headers = spec.get("headers") or {}
    if not isinstance(raw_headers, dict):
        raise ValueError(f"mcp_servers.{name}.headers: expected a mapping")

    headers = {
        str(k): _expand_env(str(v), f"mcp_servers.{name}.headers.{k}")
        for k, v in raw_headers.items()
    }
    return url, headers or None


class MCPConnection:
    """Manages a persistent connection to an MCP server."""

    def __init__(self, name: str, url: str, prefix: str = "aptool",
                 headers: dict[str, str] | None = None):
        self.name = name
        self.url = url
        # Never logged: these carry bearer tokens and API keys.
        self.headers = dict(headers) if headers else None
        self.prefix = prefix
        self.session: ClientSession | None = None
        self.capabilities: Any = None
        self.tools: list[dict] = []
        self.binary_params: dict[str, list[BinaryParam]] = {}
        self.instructions: str = ""
        self._original_names: dict[str, str] = {}
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._error: BaseException | None = None
        self._task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Start the connection task and wait until ready or failed."""
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.name}")
        await self._ready.wait()
        if self._error:
            raise self._error

    async def _run(self) -> None:
        """Run the connection lifecycle in an isolated task."""
        try:
            async with _open_transport(self.url, self.headers) as transport:
                read, write = transport[0], transport[1]
                async with ClientSession(read, write) as session:
                    self.session = session
                    init_result = await session.initialize()

                    self.capabilities = getattr(init_result, "capabilities", None)
                    self.instructions = getattr(init_result, "instructions", "") or ""

                    result = await session.list_tools()
                    self.tools = []
                    self.binary_params: dict[str, list[BinaryParam]] = {}
                    for t in result.tools:
                        openai_tool = mcp_tool_to_openai(t)
                        original_name = openai_tool["name"]
                        prefixed_name = f"{self.prefix}-{self.name}-{original_name}"
                        self._original_names[prefixed_name] = original_name
                        openai_tool["name"] = prefixed_name
                        schema = inline_refs(openai_tool.get("parameters") or {})
                        openai_tool["parameters"] = schema
                        params = rewrite_binary_params(schema)
                        if params:
                            self.binary_params[prefixed_name] = params
                            logger.debug("  %s: %d binary param(s) → pointer", prefixed_name, len(params))
                        self.tools.append(openai_tool)

                    logger.info("MCP [%s]: connected, %d tools, instructions=%s",
                                self.name, len(self.tools), "yes" if self.instructions else "no")
                    for t in self.tools:
                        logger.debug("  - %s", t["name"])

                    self._ready.set()
                    await self._shutdown.wait()
        except BaseException as e:
            self._error = e
            self._ready.set()

    async def call_tool(self, prefixed_name: str, arguments: str | dict) -> list:
        """Execute a tool call and return raw MCP content blocks."""
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name} not connected")

        original_name = self._original_names.get(prefixed_name, prefixed_name)
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        result = await self.session.call_tool(original_name, args)
        return result.content

    @property
    def supports_resources(self) -> bool:
        """Whether the server advertised the resources capability.

        Resources are optional in the protocol — plenty of servers expose
        tools only, and asking them to enumerate resources is an error.
        """
        return getattr(self.capabilities, "resources", None) is not None

    async def list_resources(self) -> list:
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name} not connected")
        if not self.supports_resources:
            return []
        try:
            result = await self.session.list_resources()
        except McpError as e:
            if not _is_method_not_found(e):
                raise
            logger.debug("MCP [%s]: resources/list not implemented", self.name)
            return []
        return list(result.resources or [])

    async def list_resource_templates(self) -> list:
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name} not connected")
        if not self.supports_resources:
            return []
        try:
            result = await self.session.list_resource_templates()
        except McpError as e:
            # A server can advertise `resources` and still not implement the
            # templates half — the two are separate methods.
            if not _is_method_not_found(e):
                raise
            logger.debug("MCP [%s]: resources/templates/list not implemented", self.name)
            return []
        return list(result.resourceTemplates or [])

    async def read_resource(self, uri: str) -> list:
        """Read a resource and return its contents as MCP ResourceContents."""
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name} not connected")
        result = await self.session.read_resource(uri)
        return list(result.contents or [])

    async def close(self) -> None:
        self._shutdown.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except BaseException:
                pass
