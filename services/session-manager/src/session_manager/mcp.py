"""MCP connection management — tool discovery, execution, and content conversion."""

import asyncio
import base64
import json
import logging

import jsonref
from jsonpath_ng import Fields
from jsonpath_ng import parse as jsonpath_parse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from session_manager.binaries import BinaryStore

logger = logging.getLogger(__name__)

POINTER_PREFIX = "pointer://"
BINARY_FORMATS = {"binary", "byte", "base64"}
_ALL_NODES = jsonpath_parse("$..*")


def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP tool to OpenAI Responses API function tool format."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema or {},
    }


def _rewrite_to_pointer(node: dict) -> None:
    """Rewrite a binary-string schema node in place as a pointer-typed string."""
    node.pop("format", None)
    existing = node.get("description", "")
    node["description"] = (
        f"{existing} Pointer to a stored binary "
        f"(e.g. '{POINTER_PREFIX}5-photo.jpg'). "
        f"Do not pass raw bytes or base64."
    ).strip()


def _enclosing_property_name(match) -> str | None:
    """Walk up the match's context chain and return the field name whose
    parent is a `properties` dict — i.e., the user-visible argument name.
    Returns None if no such ancestor exists."""
    cur = match
    while cur is not None and cur.context is not None:
        parent_path = cur.context.path
        if isinstance(parent_path, Fields) and parent_path.fields == ("properties",):
            if isinstance(cur.path, Fields) and len(cur.path.fields) == 1:
                return cur.path.fields[0]
        cur = cur.context
    return None


def rewrite_binary_params(schema: dict) -> set[str]:
    """Find every binary-string schema node, rewrite each in place as a
    pointer-typed string, and return the set of argument property names
    that may carry a pointer.

    Matches on the `format` field itself (jsonpath-ng's $..* descends into
    arrays but doesn't yield dict list-elements as top-level matches, so we
    pivot off the leaf scalar and look up to the containing dict).

    Expects refs already resolved and `$defs` removed — see inline_refs().
    """
    names: set[str] = set()
    for match in _ALL_NODES.find(schema):
        # Leaf `format: <binary-variant>` match?
        if not isinstance(match.value, str) or match.value not in BINARY_FORMATS:
            continue
        if not (isinstance(match.path, Fields) and match.path.fields == ("format",)):
            continue
        parent_dict = match.context.value
        if not isinstance(parent_dict, dict) or parent_dict.get("type") != "string":
            continue
        _rewrite_to_pointer(parent_dict)
        name = _enclosing_property_name(match.context)
        if name:
            names.add(name)
    return names


def inline_refs(schema: dict) -> dict:
    """Resolve every `$ref` and drop `$defs`, returning a plain dict.

    Fully materializes (proxies=False) so downstream code can mutate the
    schema freely. Self-referential schemas will expand infinitely and
    aren't supported here.
    """
    resolved = jsonref.replace_refs(schema, proxies=False)
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
    return resolved


def resolve_pointer_args(args: dict, names: set[str], store: BinaryStore) -> dict:
    """Walk args; whenever we're under a key whose schema was binary-typed,
    replace every `pointer://...` string we find with its base64-encoded
    bytes. Non-pointer values pass through. Expired pointers raise loudly.
    """
    if not names:
        return args

    def resolve_deep(value):
        """Value is nested under a binary-typed key — resolve any pointer."""
        if isinstance(value, str):
            if value.startswith(POINTER_PREFIX):
                pointer = value[len(POINTER_PREFIX):]
                try:
                    content, _ = store.load(pointer)
                except FileNotFoundError:
                    raise ValueError(
                        f"Pointer {pointer!r} not found — it may have been garbage-collected. "
                        "Ask for the binary to be re-produced."
                    )
                return base64.b64encode(content).decode()
            return value
        if isinstance(value, list):
            return [resolve_deep(item) for item in value]
        if isinstance(value, dict):
            return {k: resolve_deep(v) for k, v in value.items()}
        return value

    def walk(node):
        if isinstance(node, dict):
            return {k: (resolve_deep(v) if k in names else walk(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(args)


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
    """Persist bytes and return an input_text pointer part, or a textual
    placeholder if no store is available."""
    pointer = _save_and_describe(store, data_b64, mime_type) if store else None
    if pointer:
        return _pointer_text(pointer)
    return {"type": "input_text", "text": f"[binary: {mime_type}]"}


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
        self.binary_param_names: dict[str, set[str]] = {}
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
                    self.binary_param_names: dict[str, set[str]] = {}
                    for t in result.tools:
                        openai_tool = mcp_tool_to_openai(t)
                        original_name = openai_tool["name"]
                        prefixed_name = f"{self.prefix}-{self.name}-{original_name}"
                        self._original_names[prefixed_name] = original_name
                        openai_tool["name"] = prefixed_name
                        schema = inline_refs(openai_tool.get("parameters") or {})
                        openai_tool["parameters"] = schema
                        names = rewrite_binary_params(schema)
                        if names:
                            self.binary_param_names[prefixed_name] = names
                            logger.info(f"  {prefixed_name}: binary params → pointer ({sorted(names)})")
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
        names = self.binary_param_names.get(prefixed_name, set())
        if names and store is not None:
            args = resolve_pointer_args(args, names, store)
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
