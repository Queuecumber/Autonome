"""MCP connection management — tool discovery, execution, and content conversion."""

import asyncio
import json
import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP tool to OpenAI Responses API function tool format."""
    schema = dict(tool.inputSchema) if tool.inputSchema else {}
    if "type" not in schema:
        schema["type"] = "object"
    if "properties" not in schema:
        schema["properties"] = {}
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": schema,
    }


def mcp_content_to_openai(content_blocks: list) -> list[dict]:
    """Convert MCP content blocks to OpenAI Responses API message content parts.

    TextContent → text part
    ImageContent → input_image part (data URI)
    AudioContent → text placeholder (no OpenAI audio in tool results yet)
    """
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            parts.append({
                "type": "input_image",
                "image_url": f"data:{block.mimeType};base64,{block.data}",
            })
        elif block.type == "audio":
            parts.append({"type": "text", "text": f"[audio: {block.mimeType}]"})
        else:
            parts.append({"type": "text", "text": str(block)})
    return parts



class MCPConnection:
    """Manages a persistent connection to an MCP server.

    Each connection runs in its own asyncio task to isolate anyio cancel scopes
    from the streamable HTTP transport.
    """

    def __init__(self, name: str, url: str, prefix: str = "aptool"):
        self.name = name
        self.url = url
        self.prefix = prefix
        self.session: ClientSession | None = None
        self.tools: list[dict] = []
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
                    for t in result.tools:
                        openai_tool = mcp_tool_to_openai(t)
                        original_name = openai_tool["name"]
                        prefixed_name = f"{self.prefix}-{self.name}-{original_name}"
                        self._original_names[prefixed_name] = original_name
                        openai_tool["name"] = prefixed_name
                        self.tools.append(openai_tool)

                    logger.info(f"MCP [{self.name}]: connected, {len(self.tools)} tools, instructions={'yes' if self.instructions else 'no'}")
                    for t in self.tools:
                        logger.info(f"  - {t['name']}")

                    self._ready.set()
                    await self._shutdown.wait()
        except BaseException as e:
            self._error = e
            self._ready.set()

    async def call_tool(self, prefixed_name: str, arguments: str) -> list:
        """Execute a tool call and return raw MCP content blocks."""
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name} not connected")

        original_name = self._original_names.get(prefixed_name, prefixed_name)
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        result = await self.session.call_tool(original_name, args)
        return result.content

    async def close(self) -> None:
        self._shutdown.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, BaseException):
                pass
