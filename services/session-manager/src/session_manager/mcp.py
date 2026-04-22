"""MCP connection management — tool discovery, execution, and content conversion."""

import asyncio
import base64
import copy
import json
import logging
from dataclasses import dataclass
from functools import cached_property

import jsonpath
import jsonref
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from session_manager.binaries import BinaryStore

logger = logging.getLogger(__name__)

POINTER_PREFIX = "pointer://"

# Select every schema node whose format marks it as binary content.
_BINARY_FINDER = jsonpath.compile(
    "$..[?(@.format == 'binary' or @.format == 'byte' or @.format == 'base64')]"
)

_POINTER_DESCRIPTION = (
    f"Pointer to a stored binary (e.g. '{POINTER_PREFIX}5-photo.jpg'). "
    "Do not pass raw bytes or base64."
)


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
    (strip the binary `format`, add a pointer-usage description), and
    return a list of BinaryParams for dispatch-time resolution.

    Expects refs already resolved via inline_refs().
    """
    params: list[BinaryParam] = []
    for match in _BINARY_FINDER.finditer(schema):
        patch = (
            jsonpath.JSONPatch()
            .replace(match.pointer().join("format"), "string")
            .add(match.pointer().join("description"), _POINTER_DESCRIPTION)
        )
        patch.apply(schema)
        params.append(BinaryParam(match.pointer()))
    return params


def resolve_pointer_args(args: dict, params: list[BinaryParam], store: BinaryStore) -> dict:
    """For each BinaryParam, find matching positions in args and replace
    any `pointer://...` strings with their base64-encoded bytes. Non-pointer
    values pass through untouched. Expired pointers raise loudly."""
    if not params:
        return args
    args = copy.deepcopy(args)
    for param in params:
        for match in param.args_matcher.finditer(args):
            val = match.value
            if not isinstance(val, str) or not val.startswith(POINTER_PREFIX):
                continue
            pointer_id = val[len(POINTER_PREFIX):]
            try:
                content, _ = store.load(pointer_id)
            except FileNotFoundError:
                raise ValueError(
                    f"Pointer {pointer_id!r} not found — it may have been "
                    "garbage-collected. Ask for the binary to be re-produced."
                )
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
        logger.warning(f"Failed to persist binary ({mime_type}) to BinaryStore: {e}")
        return None


def _pointer_text(pointer: dict) -> dict:
    return {"type": "input_text", "text": json.dumps({"pointer": pointer}, ensure_ascii=False)}


def _describe_binary(data_b64: str, mime_type: str, store: BinaryStore | None) -> dict:
    """Persist bytes and return an input_text carrying pointer JSON. If no
    store is available the shape still carries content_type with no id,
    signaling 'binary present, no pointer to reference it by.'"""
    pointer = _save_and_describe(store, data_b64, mime_type) if store else None
    return _pointer_text(pointer or {"content_type": mime_type})


def mcp_content_to_openai(content_blocks: list, store: BinaryStore | None = None) -> list[dict]:
    """Convert MCP content blocks to OpenAI Responses API message content parts.

    Every binary gets persisted to the BinaryStore and produces an input_text
    part carrying the pointer JSON. Images additionally produce an input_image
    part so the model can see the bytes.
    """
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append({"type": "input_text", "text": block.text})

        elif block.type == "image":
            parts.append(_describe_binary(block.data, block.mimeType, store))
            parts.append({
                "type": "input_image",
                "image_url": f"data:{block.mimeType};base64,{block.data}",
            })

        elif block.type == "audio":
            parts.append(_describe_binary(block.data, block.mimeType, store))

        elif block.type == "resource":
            resource = getattr(block, "resource", None)
            blob = getattr(resource, "blob", None)
            text = getattr(resource, "text", None)
            mime = getattr(resource, "mimeType", None) or "application/octet-stream"
            if blob is not None:
                parts.append(_describe_binary(blob, mime, store))
            elif text is not None:
                parts.append({"type": "input_text", "text": text})
            else:
                parts.append({"type": "input_text", "text": str(block)})

        else:
            parts.append({"type": "input_text", "text": str(block)})
    return parts



class MCPConnection:
    """Manages a persistent connection to an MCP server."""

    def __init__(self, name: str, url: str, prefix: str = "aptool"):
        self.name = name
        self.url = url
        self.prefix = prefix
        self.session: ClientSession | None = None
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
            async with streamablehttp_client(self.url) as transport:
                read, write = transport[0], transport[1]
                async with ClientSession(read, write) as session:
                    self.session = session
                    init_result = await session.initialize()

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
                            logger.info(f"  {prefixed_name}: {len(params)} binary param(s) → pointer")
                        self.tools.append(openai_tool)

                    logger.info(f"MCP [{self.name}]: connected, {len(self.tools)} tools, instructions={'yes' if self.instructions else 'no'}")
                    for t in self.tools:
                        logger.info(f"  - {t['name']}")

                    self._ready.set()
                    await self._shutdown.wait()
        except BaseException as e:
            self._error = e
            self._ready.set()

    async def call_tool(
        self, prefixed_name: str, arguments: str, store: BinaryStore | None = None
    ) -> list:
        """Execute a tool call and return raw MCP content blocks.

        If a BinaryStore is provided and this tool has binary params, any
        pointer:// strings in arguments are resolved to base64 bytes first.
        """
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name} not connected")

        original_name = self._original_names.get(prefixed_name, prefixed_name)
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        params = self.binary_params.get(prefixed_name, [])
        if params and store is not None:
            args = resolve_pointer_args(args, params, store)
        result = await self.session.call_tool(original_name, args)
        return result.content

    async def close(self) -> None:
        self._shutdown.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except BaseException:
                pass
