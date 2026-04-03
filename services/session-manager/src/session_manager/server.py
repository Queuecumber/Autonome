"""Session manager: central orchestrator that receives events and drives LLM calls.

Uses the OpenAI Python SDK for model calls and connects to MCP servers
directly for tool discovery and execution.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_platform.session import SessionManager

logger = logging.getLogger(__name__)


def _mcp_tool_to_openai(tool) -> dict:
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


def _strip_binary_content(tool_result: dict) -> dict:
    """Strip base64 image data from a tool result before saving to history."""
    content = tool_result.get("content")
    if isinstance(content, list):
        # Multimodal content — replace image blocks with references
        stripped = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                stripped.append({"type": "text", "text": "[image content stripped from history]"})
            else:
                stripped.append(block)
        return {**tool_result, "content": stripped}
    if isinstance(content, str) and len(content) > 10000:
        # Large text content (likely base64) — check if it's a JSON blob with base64
        try:
            data = json.loads(content)
            if isinstance(data, dict) and data.get("content_base64"):
                data["content_base64"] = "[stripped from history]"
                return {**tool_result, "content": json.dumps(data)}
        except (json.JSONDecodeError, KeyError):
            pass
    return tool_result


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
        self._context = None

    async def connect(self) -> None:
        """Establish connection, get server instructions, and discover tools."""
        self._context = streamablehttp_client(self.url)
        read, write, _ = await self._context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        init_result = await self.session.initialize()

        if hasattr(init_result, "instructions") and init_result.instructions:
            self.instructions = init_result.instructions

        result = await self.session.list_tools()
        self.tools = []
        for t in result.tools:
            openai_tool = _mcp_tool_to_openai(t)
            original_name = openai_tool["function"]["name"]
            prefixed_name = f"{self.prefix}-{self.name}-{original_name}"
            self._original_names[prefixed_name] = original_name
            openai_tool["function"]["name"] = prefixed_name
            self.tools.append(openai_tool)

        logger.info(f"MCP [{self.name}]: connected, {len(self.tools)} tools, instructions={'yes' if self.instructions else 'no'}")
        for t in self.tools:
            logger.info(f"  - {t['function']['name']}")

    async def call_tool(self, prefixed_name: str, arguments: str) -> str:
        """Execute a tool call and return the result as a string."""
        if self.session is None:
            return f"Error: MCP server {self.name} not connected"

        original_name = self._original_names.get(prefixed_name, prefixed_name)

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            result = await self.session.call_tool(original_name, args)
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts) if parts else ""
        except Exception as e:
            logger.error(f"MCP [{self.name}] tool {original_name} failed: {e}")
            return f"Error: {e}"

    async def close(self) -> None:
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._context:
            await self._context.__aexit__(None, None, None)


class SessionOrchestrator:
    """Receives events from adapters, maintains session history, drives LLM calls."""

    def __init__(self, config: dict, session_dir: Path):
        self.config = config

        model_config = config["model"]
        self.model = model_config["model"]
        self.reasoning_effort = model_config.get("reasoning_effort")

        self.llm = AsyncOpenAI(
            api_key=model_config.get("api_key", ""),
            base_url=model_config.get("api_base"),
            default_headers=model_config.get("extra_headers"),
            timeout=300,
        )

        session_config = config.get("session", {})
        max_tokens = session_config.get("max_history_tokens", 100000)
        self.session = SessionManager(store_dir=session_dir, max_history_tokens=max_tokens)

        self.workspace_dir = Path(config.get("workspace", "./workspace"))

        self.heartbeat_prompt = config.get("heartbeat", {}).get(
            "prompt", "Check HEARTBEAT.md"
        )

        channels = config.get("channels", {})
        signal_config = channels.get("signal", {})
        allow_from = signal_config.get("allow_from", [])
        self.heartbeat_source = "signal"
        self.heartbeat_session_id = allow_from[0] if allow_from else "system"

        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

        self.mcp_connections: dict[str, MCPConnection] = {}
        self.openai_tools: list[dict] = []
        self._tool_to_mcp: dict[str, MCPConnection] = {}

        self.max_tool_iterations = 20

    async def connect_mcp_servers(self, mcp_urls: dict[str, str]) -> None:
        """Connect to all MCP servers and discover tools."""
        for name, url in mcp_urls.items():
            conn = MCPConnection(name, url)
            try:
                await conn.connect()
                self.mcp_connections[name] = conn
                for tool in conn.tools:
                    tool_name = tool["function"]["name"]
                    self.openai_tools.append(tool)
                    self._tool_to_mcp[tool_name] = conn
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name} at {url}: {e}")

        logger.info(f"Connected to {len(self.mcp_connections)} MCP servers, {len(self.openai_tools)} tools total")

    def _get_lock(self, channel: str, session_id: str) -> asyncio.Lock:
        key = (channel, session_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _build_system_prompt(self) -> str:
        """Build system prompt from AGENTS.md + MCP server instructions."""
        parts = []

        agents_path = self.workspace_dir / "AGENTS.md"
        if agents_path.exists():
            parts.append(agents_path.read_text())

        server_docs = []
        for conn in self.mcp_connections.values():
            if conn.instructions:
                tool_names = ", ".join(t["function"]["name"] for t in conn.tools)
                server_docs.append(f"### {conn.name}\n{conn.instructions}\nTools: {tool_names}")
        if server_docs:
            parts.append("## Available Tool Servers\n\n" + "\n\n".join(server_docs))

        return "\n\n".join(parts)

    async def _execute_tool_call(self, tool_call) -> dict:
        """Execute a single tool call via the appropriate MCP server."""
        func = tool_call.function
        tool_name = func.name
        conn = self._tool_to_mcp.get(tool_name)

        if conn is None:
            result_text = f"Error: unknown tool '{tool_name}'"
        else:
            result_text = await conn.call_tool(tool_name, func.arguments)

        # Check if tool result contains image data — pass as multimodal content
        try:
            result_data = json.loads(result_text)
            if isinstance(result_data, dict) and result_data.get("content_type", "").startswith("image/"):
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": [
                        {"type": "text", "text": f"Attachment {result_data.get('id', '')}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{result_data['content_type']};base64,{result_data['content_base64']}"},
                        },
                    ],
                }
        except (json.JSONDecodeError, KeyError):
            pass

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        }

    async def handle_event(self, event: dict[str, Any]) -> str | None:
        """Process an inbound event from any adapter."""
        source = event["source"]
        session_id = event["session_id"]
        text = event.get("text", "")
        metadata = event.get("metadata", {})

        async with self._get_lock(source, session_id):
            # Load session history, filter and sanitize
            raw_history = self.session.load_truncated(
                channel=source,
                session_id=session_id,
                model=self.model,
            )
            history = []
            for msg in raw_history:
                if msg.get("role") not in ("user", "assistant", "tool", "system"):
                    continue
                if not msg.get("content"):
                    if msg.get("tool_calls"):
                        msg["content"] = "(calling tools)"
                    elif msg.get("role") == "assistant":
                        msg["content"] = "(no text response)"
                for key in ("reasoning_content", "thinking_blocks", "provider_specific_fields", "function_call"):
                    msg.pop(key, None)
                # Strip broken/empty image_url blocks from multimodal content
                content = msg.get("content")
                if isinstance(content, list):
                    msg["content"] = [
                        block for block in content
                        if not (isinstance(block, dict) and block.get("type") == "image_url")
                    ] or "(image stripped)"
                history.append(msg)

            # Build user message with metadata context + current timestamp
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            context_prefix = f"[{source} | time={now}]"
            if metadata:
                context_prefix = f"[{source} | time={now} | {json.dumps(metadata)}]"
            enriched_msg = {"role": "user", "content": f"{context_prefix} {text}"}
            stored_msg = {"role": "user", "content": text}

            # Build system prompt (AGENTS.md + MCP server instructions)
            system_content = self._build_system_prompt()
            system_msg = [{"role": "system", "content": system_content}] if system_content else []
            messages = system_msg + history + [enriched_msg]

            # Build API call kwargs
            call_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if self.reasoning_effort:
                call_kwargs["reasoning_effort"] = self.reasoning_effort
            if self.openai_tools:
                call_kwargs["tools"] = self.openai_tools

            logger.info(f"Calling LLM: {len(messages)} messages, {len(self.openai_tools)} tools")

            # Tool execution loop
            all_new_messages = [stored_msg]
            for iteration in range(self.max_tool_iterations):
                try:
                    response = await self.llm.chat.completions.create(**call_kwargs)
                except Exception as e:
                    logger.error(f"LLM call failed: {e}")
                    return None

                choice = response.choices[0]
                assistant_msg = choice.message

                logger.info(f"LLM response (iter {iteration}): finish_reason={choice.finish_reason}")

                # If the model wants to call tools
                if choice.finish_reason == "tool_calls" or (assistant_msg.tool_calls and len(assistant_msg.tool_calls) > 0):
                    assistant_dict = assistant_msg.model_dump()
                    # Bedrock requires non-empty content even on tool_call messages
                    if not assistant_dict.get("content"):
                        assistant_dict["content"] = "(calling tools)"
                    messages.append(assistant_dict)
                    all_new_messages.append(assistant_dict)

                    for tool_call in assistant_msg.tool_calls:
                        logger.info(f"  Tool call: {tool_call.function.name}({tool_call.function.arguments[:100]})")
                        tool_result = await self._execute_tool_call(tool_call)
                        logger.info(f"  Result: {str(tool_result['content'])[:200]}")
                        messages.append(tool_result)
                        all_new_messages.append(_strip_binary_content(tool_result))

                    call_kwargs["messages"] = messages
                    continue

                # No tool calls — final response
                assistant_text = assistant_msg.content or ""
                if assistant_text:
                    all_new_messages.append({"role": "assistant", "content": assistant_text})

                self.session.append(source, session_id, all_new_messages)

                logger.info(f"Final response: {assistant_text[:200]}")
                return assistant_text

            logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached")
            self.session.append(source, session_id, all_new_messages)
            return None

    async def handle_heartbeat(self) -> str | None:
        """Process a heartbeat routed to the primary contact's session."""
        event = {
            "source": self.heartbeat_source,
            "session_id": self.heartbeat_session_id,
            "text": f"[HEARTBEAT] {self.heartbeat_prompt}",
            "metadata": {},
        }
        return await self.handle_event(event)

    async def close(self) -> None:
        for conn in self.mcp_connections.values():
            await conn.close()


def create_app(orchestrator: SessionOrchestrator) -> Starlette:
    """Create the HTTP application for the session manager."""

    async def event_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        result = await orchestrator.handle_event(body)
        if result is None:
            return JSONResponse(
                {"status": "error", "response": None},
                status_code=502,
            )
        return JSONResponse({"status": "ok", "response": result})

    async def heartbeat_endpoint(request: Request) -> JSONResponse:
        result = await orchestrator.handle_heartbeat()
        if result is None:
            return JSONResponse(
                {"status": "error", "response": None},
                status_code=502,
            )
        return JSONResponse({"status": "ok", "response": result})

    return Starlette(routes=[
        Route("/event", event_endpoint, methods=["POST"]),
        Route("/heartbeat", heartbeat_endpoint, methods=["POST"]),
    ])
