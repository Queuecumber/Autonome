"""Session manager: central orchestrator that receives events and drives LLM calls."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from session_manager.mcp import MCPConnection, mcp_content_to_openai
from session_manager.session import SessionManager

logger = logging.getLogger(__name__)

# TODO: make configurable, iterate on prompting
SYSTEM_PROMPT = """\
# Your Environment

You are running on the Agent Platform. You interact with the world through \
MCP tools — there is no shell, no bash, no direct file access. Everything \
you do happens through tool calls.

Your available tools are provided automatically. Use them directly by name.

## Responding to Messages

Your text output is NOT delivered to anyone. The ONLY way to communicate is \
by calling the appropriate send_message tool with the recipient and your \
message text. If you do not call send_message, the user will not see any \
response from you.

You are free to use your text output as an internal molologue, to help you \
reason or remember things in the direct conversation context. You can think of \
this as your private thoughts on whats happening, use this capability however \
you wish. Remember, your text output will not be delivered to the user, you \
must use the appropriate send_message tool to communicate with the user. The \
text output is for you and you alone.

## Message Context

When you receive a message, it includes metadata with the sender, message ID, \
and timestamp. Use the sender to reply and the message ID for reactions/receipts.

## Every Session

Before doing anything else:

1. Read your identity files (SOUL.md, etc.) — this is who you are
2. Read USER.md — this is who you're helping
3. Read recent daily memories for context
4. Read your global memory
5. DO NOT read private files unless specifically asked

Don't ask permission. Just do it.

## Memory

You wake up fresh each session. Your memory tools are your continuity:

- Daily notes — raw logs of what happened, stored by date
- Global memory — curated long-term index

Capture what matters. Decisions, context, things to remember.

Periodically review recent daily memories and update global memory with \
what's worth keeping long-term.

## Safety

- Don't exfiltrate private data
- Don't modify workspace files without good reason
- When in doubt, ask

## Style

- Keep responses concise. This is chat, not an essay.
- One or two sentences is usually enough.
- Only go long when the topic genuinely needs it.
- You're a person in a conversation, not a report generator.
"""


def _prepare_for_history(msg: dict) -> dict:
    """Prepare a message for session history — strip binary, ensure non-empty content."""
    msg = dict(msg)  # shallow copy

    if not msg.get("content"):
        if msg.get("tool_calls"):
            msg["content"] = "(calling tools)"
        elif msg.get("role") == "assistant":
            msg["content"] = "(no text response)"

    # Replace binary content blocks with text placeholders
    content = msg.get("content")
    if isinstance(content, list):
        msg["content"] = [
            {"type": "text", "text": "[image]"} if (isinstance(b, dict) and b.get("type") == "image_url") else b
            for b in content
        ] or "(stripped)"

    return msg


def _prepare_for_model(msg: dict) -> dict | None:
    """Prepare a history message for sending to the model. Returns None to skip."""
    role = msg.get("role")
    if role not in ("user", "assistant", "tool", "system"):
        return None

    msg = {k: v for k, v in msg.items()
           if k not in ("reasoning_content", "thinking_blocks", "provider_specific_fields", "function_call")}

    if not msg.get("content"):
        if msg.get("tool_calls"):
            msg["content"] = "(calling tools)"
        elif role == "assistant":
            msg["content"] = "(no text response)"

    # Strip any leftover image blocks from history
    content = msg.get("content")
    if isinstance(content, list):
        msg["content"] = [
            b for b in content
            if not (isinstance(b, dict) and b.get("type") == "image_url")
        ] or "(image stripped)"

    return msg


class SessionOrchestrator:
    """Receives events from adapters, maintains session history, drives LLM calls."""

    def __init__(self, config: dict, session_dir: Path):
        self.config = config

        model_config = config.get("model", {})
        self.model = model_config.get("model", "")
        self.reasoning_effort = model_config.get("reasoning_effort")

        # OpenAI SDK reads OPENAI_API_KEY and OPENAI_BASE_URL from env by default
        self.llm = AsyncOpenAI(
            default_headers=model_config.get("extra_headers"),
            timeout=300,
        )

        session_config = config.get("session", {})
        max_tokens = session_config.get("max_history_tokens", 100000)
        self.session = SessionManager(store_dir=session_dir, max_history_tokens=max_tokens)

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
        """Build system prompt from base instructions + MCP server instructions."""
        parts = [SYSTEM_PROMPT]

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
            return {"role": "tool", "tool_call_id": tool_call.id, "content": f"Error: unknown tool '{tool_name}'"}

        content_blocks = await conn.call_tool(tool_name, func.arguments)
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": mcp_content_to_openai(content_blocks),
        }

    async def handle_event(self, event: dict[str, Any]) -> str | None:
        """Process an inbound event from any adapter."""
        source = event["source"]
        session_id = event["session_id"]
        text = event.get("text", "")
        metadata = event.get("metadata", {})

        async with self._get_lock(source, session_id):
            # Load session history, prepare for model
            raw_history = self.session.load_truncated(source, session_id)
            history = [m for msg in raw_history if (m := _prepare_for_model(msg)) is not None]

            # Build user message with metadata context + current timestamp (local time with tz)
            now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            meta = f" | {json.dumps(metadata)}" if metadata else ""
            enriched_msg = {"role": "user", "content": f"[{source} | time={now}{meta}] {text}"}
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
                    if not assistant_dict.get("content"):
                        assistant_dict["content"] = "(calling tools)"
                    messages.append(assistant_dict)
                    all_new_messages.append(_prepare_for_history(assistant_dict))

                    for tool_call in assistant_msg.tool_calls:
                        logger.info(f"  Tool call: {tool_call.function.name}({tool_call.function.arguments[:100]})")
                        tool_result = await self._execute_tool_call(tool_call)
                        logger.info(f"  Result: {str(tool_result['content'])[:200]}")
                        messages.append(tool_result)
                        all_new_messages.append(_prepare_for_history(tool_result))

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

    async def close(self) -> None:
        for conn in self.mcp_connections.values():
            await conn.close()
