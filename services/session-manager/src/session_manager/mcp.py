"""MCP connection management — tool discovery, execution, and content conversion."""

import asyncio
import base64
import copy
import json
import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from session_manager.binaries import BinaryStore

logger = logging.getLogger(__name__)


def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP tool to OpenAI Responses API function tool format."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema or {},
    }


def rewrite_binary_params(schema: dict) -> list[tuple[str, ...]]:
    """In-place rewrite of `{type: string, format: byte|binary}` fields to
    pointer-typed strings. Returns the paths of rewritten fields so the
    orchestrator can resolve them before dispatch.

    Handles nested anyOf/oneOf/allOf (e.g. Optional[bytes] → {anyOf:[
    {type:string, format:binary}, {type:null}]}). Each matching variant
    gets rewritten in place and the outer path is recorded once.
    """
    paths: list[tuple[str, ...]] = []

    def rewrite_node(node: dict) -> bool:
        """Rewrite a binary-string node in place. Returns True if matched."""
        if node.get("type") == "string" and node.get("format") in ("byte", "binary"):
            node.pop("format", None)
            existing = node.get("description", "")
            node["description"] = (
                (existing + " " if existing else "")
                + "Pointer (filename) to a stored binary from a prior tool result. "
                  "Do not pass raw bytes or base64."
            ).strip()
            return True
        return False

    def walk(node: dict, path: tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            return
        if rewrite_node(node):
            paths.append(path)
            return
        matched_here = False
        for combinator in ("anyOf", "oneOf", "allOf"):
            variants = node.get(combinator)
            if isinstance(variants, list):
                for variant in variants:
                    if isinstance(variant, dict) and rewrite_node(variant):
                        matched_here = True
        if matched_here:
            paths.append(path)
            return
        for key, child in (node.get("properties") or {}).items():
            walk(child, path + (key,))
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, path + ("[]",))

    walk(schema, ())
    return paths


def resolve_pointer_args(args: dict, paths: list[tuple[str, ...]], store: BinaryStore) -> dict:
    """Walk args and replace pointers at the given paths with base64-encoded bytes."""
    if not paths:
        return args
    args = copy.deepcopy(args)

    def sub(node, path: tuple[str, ...]):
        if not path:
            if isinstance(node, str):
                try:
                    content, _ = store.load(node)
                except FileNotFoundError:
                    return node
                return base64.b64encode(content).decode()
            return node
        head, *rest = path
        if head == "[]" and isinstance(node, list):
            return [sub(item, tuple(rest)) for item in node]
        if isinstance(node, dict) and head in node:
            node[head] = sub(node[head], tuple(rest))
        return node

    for path in paths:
        sub(args, path)
    return args


def _save_and_describe(store: BinaryStore, data_b64: str, mime_type: str) -> dict | None:
    """Persist base64 content and return the pointer metadata, or None on failure."""
    try:
        raw = base64.b64decode(data_b64)
        pointer_id = store.save(raw, mime_type)
        return {"id": pointer_id, "content_type": mime_type, "size": len(raw)}
    except Exception as e:
        logger.warning(f"Failed to persist binary ({mime_type}) to BinaryStore: {e}")
        return None


def mcp_content_to_openai(content_blocks: list, store: BinaryStore | None = None) -> list[dict]:
    """Convert MCP content blocks to OpenAI Responses API message content parts.

    TextContent  → input_text
    ImageContent → input_image with a `_pointer` sidecar (model-consumable)
    AudioContent → input_text carrying the pointer JSON (not model-consumable today)
    Other        → input_text with a string fallback

    Non-text content is persisted to the BinaryStore when one is provided.
    """
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append({"type": "input_text", "text": block.text})
        elif block.type == "image":
            part: dict = {
                "type": "input_image",
                "image_url": f"data:{block.mimeType};base64,{block.data}",
            }
            if store is not None:
                pointer = _save_and_describe(store, block.data, block.mimeType)
                if pointer:
                    part["_pointer"] = pointer
            parts.append(part)
        elif block.type == "audio":
            pointer = _save_and_describe(store, block.data, block.mimeType) if store else None
            if pointer:
                parts.append({"type": "input_text", "text": json.dumps({"pointer": pointer}, ensure_ascii=False)})
            else:
                parts.append({"type": "input_text", "text": f"[audio: {block.mimeType}]"})
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
        self.binary_param_paths: dict[str, list[tuple[str, ...]]] = {}
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
                    self.binary_param_paths: dict[str, list[tuple[str, ...]]] = {}
                    for t in result.tools:
                        openai_tool = mcp_tool_to_openai(t)
                        original_name = openai_tool["name"]
                        prefixed_name = f"{self.prefix}-{self.name}-{original_name}"
                        self._original_names[prefixed_name] = original_name
                        openai_tool["name"] = prefixed_name
                        paths = rewrite_binary_params(openai_tool.get("parameters") or {})
                        if paths:
                            self.binary_param_paths[prefixed_name] = paths
                            logger.info(f"  {prefixed_name}: {len(paths)} binary param(s) → pointer")
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
        pointer strings in arguments are resolved to base64 bytes first.
        """
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name} not connected")

        original_name = self._original_names.get(prefixed_name, prefixed_name)
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        paths = self.binary_param_paths.get(prefixed_name, [])
        if paths and store is not None:
            args = resolve_pointer_args(args, paths, store)
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
