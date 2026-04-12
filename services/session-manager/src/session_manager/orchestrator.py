"""Session manager: central orchestrator that receives events and drives LLM calls.

Uses the OpenAI Responses API with streaming for model calls.
Supports interruption: a new message cancels in-progress generation and
provides the partial response as context for the next turn.
"""

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

You are free to use your text output as an internal monologue, to help you \
reason or remember things in the direct conversation context. You can think of \
this as your private thoughts on whats happening, use this capability however \
you wish. Remember, your text output will not be delivered to the user, you \
must use the appropriate send_message tool to communicate with the user. The \
text output is for you and you alone.

## Message Context

When you receive a message, it includes metadata with the sender, message ID, \
and timestamp. Use the sender to reply and the message ID for reactions/receipts.

## Interruptions

If you see an [interrupted] marker on a previous turn, it means you were \
generating a response when the user sent a new message. The marker shows what \
you had composed so far (tool calls, partial text). Use this context to decide \
whether your interrupted response is still relevant or should be abandoned.

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


def _prepare_for_history(item: dict) -> dict:
    """Prepare an output item for session history — strip binary content."""
    item = dict(item)
    content = item.get("content")
    if isinstance(content, list):
        item["content"] = [
            {"type": "text", "text": "[image]"} if (isinstance(b, dict) and b.get("type") in ("image_url", "input_image")) else b
            for b in content
        ] or "(stripped)"
    return item


def _describe_interrupted(completed_items: list) -> str:
    """Build a human-readable summary of what the model had generated before interruption."""
    parts = []
    for item in completed_items:
        item_type = getattr(item, "type", None)
        if item_type == "function_call":
            try:
                args = json.loads(item.arguments)
                args_str = json.dumps(args, ensure_ascii=False)
            except (json.JSONDecodeError, AttributeError):
                args_str = getattr(item, "arguments", "")
            parts.append(f"tool call: {item.name}({args_str})")
        elif item_type == "message":
            for content in getattr(item, "content", []):
                if hasattr(content, "text") and content.text:
                    parts.append(f"text: {content.text}")
    return "; ".join(parts) if parts else ""


class SessionOrchestrator:
    """Receives events from adapters, maintains session history, drives LLM calls."""

    def __init__(self, config: dict, session_dir: Path):
        self.config = config

        model_config = config.get("model", {})
        self.model = model_config.get("model", "")
        self.reasoning_effort = model_config.get("reasoning_effort")

        self.llm = AsyncOpenAI(
            default_headers=model_config.get("extra_headers"),
            timeout=300,
        )

        session_config = config.get("session", {})
        max_tokens = session_config.get("max_history_tokens", 100000)
        self.session = SessionManager(store_dir=session_dir, max_history_tokens=max_tokens)

        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._cancel: dict[tuple[str, str], asyncio.Event] = {}

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
                    tool_name = tool["name"]
                    self.openai_tools.append(tool)
                    self._tool_to_mcp[tool_name] = conn
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                logger.error(f"Failed to connect to MCP server {name} at {url}: {e}")

        logger.info(f"Connected to {len(self.mcp_connections)} MCP servers, {len(self.openai_tools)} tools total")

    def _get_lock(self, channel: str, session_id: str) -> asyncio.Lock:
        key = (channel, session_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _build_instructions(self) -> str:
        """Build instructions from base prompt + MCP server instructions."""
        parts = [SYSTEM_PROMPT]

        server_docs = []
        for conn in self.mcp_connections.values():
            if conn.instructions:
                tool_names = ", ".join(t["name"] for t in conn.tools)
                server_docs.append(f"### {conn.name}\n{conn.instructions}\nTools: {tool_names}")
        if server_docs:
            parts.append("## Available Tool Servers\n\n" + "\n\n".join(server_docs))

        return "\n\n".join(parts)

    async def _execute_tool_call(self, call_id: str, name: str, arguments: str) -> tuple[dict, list[dict]]:
        """Execute a tool call.

        Returns (function_call_output, image_items):
          - function_call_output: the output item with text content
          - image_items: user messages with image_url content for the model to see
        """
        conn = self._tool_to_mcp.get(name)

        if conn is None:
            return {"type": "function_call_output", "call_id": call_id, "output": f"Error: unknown tool '{name}'"}, []

        content_blocks = await conn.call_tool(name, arguments)
        openai_parts = mcp_content_to_openai(content_blocks)

        text_parts = []
        image_items = []
        for part in openai_parts:
            if part.get("type") == "text":
                text_parts.append(part["text"])
            elif part.get("type") == "input_image":
                text_parts.append("[image attached]")
                image_items.append({"role": "user", "content": [part]})

        output = {"type": "function_call_output", "call_id": call_id, "output": "\n".join(text_parts)}
        return output, image_items

    async def _stream_response(self, call_kwargs: dict, cancel: asyncio.Event):
        """Stream an LLM response, collecting completed items.

        Returns (response, completed_items):
          - On normal completion: (Response, [all output items])
          - On interruption: (None, [items completed before cancel])
        """
        completed_items = []
        response = None

        async for event in await self.llm.responses.create(**call_kwargs, stream=True):
            if cancel.is_set():
                logger.info("Stream interrupted by new message")
                return None, completed_items

            event_type = getattr(event, "type", None)
            if event_type == "response.output_item.done":
                completed_items.append(event.item)
            elif event_type == "response.completed":
                response = event.response

        return response, completed_items

    async def handle_event(self, event: dict[str, Any]) -> str | None:
        """Process an inbound event from any adapter."""
        source = event["source"]
        session_id = event["session_id"]
        text = event.get("text", "")
        metadata = event.get("metadata", {})
        key = (source, session_id)

        # Signal any in-progress work for this session to cancel
        if key in self._cancel:
            logger.info(f"Interrupting in-progress response for {key}")
            self._cancel[key].set()

        async with self._get_lock(source, session_id):
            cancel = asyncio.Event()
            self._cancel[key] = cancel

            try:
                return await self._process_event(
                    source, session_id, text, metadata, cancel,
                )
            finally:
                if self._cancel.get(key) is cancel:
                    del self._cancel[key]

    async def _process_event(
        self,
        source: str,
        session_id: str,
        text: str,
        metadata: dict,
        cancel: asyncio.Event,
    ) -> str | None:
        """Inner event processing with cancellation support."""
        # Load session history
        raw_history = self.session.load_truncated(source, session_id)

        # Build user message with metadata context + current timestamp
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%A)")
        meta = f" | {json.dumps(metadata)}" if metadata else ""
        enriched_text = f"[{source} | time={now}{meta}] {text}"

        # Build input: history + new message (filter reasoning — output-only type)
        history = [m for m in raw_history if m.get("type") != "reasoning"]
        input_items = history + [{"role": "user", "content": enriched_text}]

        # Build API call kwargs
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": self._build_instructions(),
            "input": input_items,
            "max_output_tokens": 16384,
        }
        if self.reasoning_effort:
            call_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.openai_tools:
            call_kwargs["tools"] = self.openai_tools

        logger.info(f"Calling LLM: {len(input_items)} input items, {len(self.openai_tools)} tools")

        # Collect all new items to save to history
        stored_msg = {"role": "user", "content": text}
        all_new_messages = [stored_msg]

        for iteration in range(self.max_tool_iterations):
            try:
                response, completed_items = await self._stream_response(call_kwargs, cancel)
            except Exception as e:
                logger.error(f"LLM call failed: {type(e).__name__}: {e!r}", exc_info=True)
                return None

            # --- Interrupted during streaming ---
            if response is None:
                desc = _describe_interrupted(completed_items)
                if desc:
                    all_new_messages.append({"role": "assistant", "content": f"[interrupted] {desc}"})
                    logger.info(f"Interrupted, partial: {desc[:200]}")
                else:
                    logger.info("Interrupted before any output completed")
                self.session.append(source, session_id, all_new_messages)
                return None

            logger.info(f"LLM response (iter {iteration}): status={response.status}")

            # Process output items
            tool_calls = []
            assistant_text = ""
            reasoning_text = ""

            for item in response.output:
                if item.type == "function_call":
                    tool_calls.append(item)
                elif item.type == "reasoning":
                    for content in item.content:
                        if hasattr(content, "text"):
                            reasoning_text += content.text
                elif item.type == "message":
                    for content in item.content:
                        if hasattr(content, "text"):
                            assistant_text += content.text

            if reasoning_text:
                all_new_messages.append({"type": "reasoning", "content": reasoning_text})

            if tool_calls:
                # Save function calls to history
                for tc in tool_calls:
                    # Re-encode arguments to get proper unicode instead of ascii escapes
                    args_unicode = json.dumps(json.loads(tc.arguments), ensure_ascii=False)
                    all_new_messages.append({
                        "type": "function_call",
                        "call_id": tc.call_id,
                        "name": tc.name,
                        "arguments": args_unicode,
                    })

                # Execute tool calls, checking for interruption between each
                tool_results = []
                for tc in tool_calls:
                    if cancel.is_set():
                        pending = tool_calls[tool_calls.index(tc):]
                        pending_desc = "; ".join(
                            f"tool call: {t.name}({json.dumps(json.loads(t.arguments), ensure_ascii=False)})"
                            for t in pending
                        )
                        logger.info(f"Interrupted between tool calls, pending: {pending_desc[:200]}")
                        all_new_messages.append({
                            "role": "assistant",
                            "content": f"[interrupted — not yet executed] {pending_desc}",
                        })
                        self.session.append(source, session_id, all_new_messages)
                        return None

                    logger.info(f"  Tool call: {tc.name}({tc.arguments[:100]})")
                    result, images = await self._execute_tool_call(tc.call_id, tc.name, tc.arguments)
                    logger.info(f"  Result: {result['output'][:200]}")
                    tool_results.append(result)
                    all_new_messages.append(_prepare_for_history(result))
                    for img in images:
                        tool_results.append(img)
                        all_new_messages.append(_prepare_for_history(img))

                # Feed results back — use previous response + results as new input
                call_kwargs["input"] = input_items + response.output + tool_results
                input_items = call_kwargs["input"]
                continue

            # No tool calls — final response
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
