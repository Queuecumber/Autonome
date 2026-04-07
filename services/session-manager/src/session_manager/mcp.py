"""MCP connection management — tool discovery, execution, and content conversion."""

import json
import logging
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP tool to OpenAI function tool format."""
    schema = dict(tool.inputSchema) if tool.inputSchema else {}
    if "type" not in schema:
        schema["type"] = "object"
    if "properties" not in schema:
        schema["properties"] = {}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def mcp_content_to_openai(content_blocks: list) -> list[dict]:
    """Convert MCP content blocks to OpenAI message content parts.

    TextContent → text part
    ImageContent → image_url part (data URI)
    AudioContent → text placeholder (no OpenAI audio in tool results yet)
    """
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
            })
        elif block.type == "audio":
            parts.append({"type": "text", "text": f"[audio: {block.mimeType}]"})
        else:
            parts.append({"type": "text", "text": str(block)})
    return parts


def content_for_history(content_blocks: list) -> str:
    """Extract text-only summary from MCP content blocks for session history.

    Binary content (images, audio) is replaced with a placeholder.
    """
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "image":
            parts.append(f"[image: {block.mimeType}]")
        elif block.type == "audio":
            parts.append(f"[audio: {block.mimeType}]")
        else:
            parts.append(str(block))
    return "\n".join(parts)


class MCPConnection:
    """Manages a persistent connection to an MCP server."""

    def __init__(self, name: str, url: str, prefix: str = "aptool"):
        self.name = name
        self.url = url
        self.prefix = prefix
        self.session: ClientSession | None = None
        self.tools: list[dict] = []
        self.instructions: str = ""
        self._original_names: dict[str, str] = {}
        self._exit_stack = AsyncExitStack()

    async def connect(self) -> None:
        """Establish connection, get server instructions, and discover tools."""
        transport = await self._exit_stack.enter_async_context(streamablehttp_client(self.url))
        read, write = transport[0], transport[1]
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        init_result = await self.session.initialize()

        self.instructions = getattr(init_result, "instructions", "") or ""

        result = await self.session.list_tools()
        self.tools = []
        for t in result.tools:
            openai_tool = mcp_tool_to_openai(t)
            original_name = openai_tool["function"]["name"]
            prefixed_name = f"{self.prefix}-{self.name}-{original_name}"
            self._original_names[prefixed_name] = original_name
            openai_tool["function"]["name"] = prefixed_name
            self.tools.append(openai_tool)

        logger.info(f"MCP [{self.name}]: connected, {len(self.tools)} tools, instructions={'yes' if self.instructions else 'no'}")
        for t in self.tools:
            logger.info(f"  - {t['function']['name']}")

    async def call_tool(self, prefixed_name: str, arguments: str) -> list:
        """Execute a tool call and return raw MCP content blocks."""
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name} not connected")

        original_name = self._original_names.get(prefixed_name, prefixed_name)
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        result = await self.session.call_tool(original_name, args)
        return result.content

    async def close(self) -> None:
        await self._exit_stack.aclose()
