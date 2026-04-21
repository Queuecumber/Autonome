"""Session manager: central orchestrator that receives events and drives LLM calls."""

import asyncio
import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from session_manager.binaries import BinaryStore
from session_manager.event import Event
from session_manager.mcp import POINTER_PREFIX, MCPConnection, mcp_content_to_openai
from session_manager.session import SessionManager

VIEW_BINARY_TOOL_NAME = "aptool-session-view_binary"
VIEW_BINARY_TOOL = {
    "type": "function",
    "name": VIEW_BINARY_TOOL_NAME,
    "description": (
        "Load a previously stored binary (by pointer) into your current input. "
        "For images this re-surfaces the image so you can look at it again. "
        "Pointers look like 'pointer://5-photo.jpg' and appear in tool result metadata."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pointer": {"type": "string", "description": "The binary pointer (may include the 'pointer://' prefix)."},
        },
        "required": ["pointer"],
    },
}

logger = logging.getLogger(__name__)

# TODO: make configurable, iterate on prompting
SYSTEM_PROMPT = """\
# Your Environment

You are running on Autonome. You interact with the world through \
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

Each user message is preceded by a developer message with structured JSON \
context: source platform, timestamp, sender, room info, attachments, and \
an "energy" field. "active" events are direct user interactions that expect \
a response. "passive" events (scheduled check-ins, receipts) are low-priority \
— respond only if there's something worth saying. Multiple events in a single \
turn mean several things arrived while you were busy; catch up as needed.

## Interruptions

If you see a developer message with {"event": "interrupted", ...} it means \
you were generating a response when the user sent a new message. The "partial" \
or "pending" field shows what you had composed. Use this to decide whether \
your interrupted response is still relevant or should be abandoned.

## Every Session

Before doing anything else:

1. Read your identity files (PERSONALITY.md, etc.) — this is who you are
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
    """Flatten content parts to a single string for history. Images map to
    '[image]'; pointer JSON is already in the input_text parts."""
    item = dict(item)
    content = item.get("content")
    if not isinstance(content, list):
        return item
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("image_url", "input_image"):
            texts.append("[image]")
        elif isinstance(block, dict) and block.get("type") == "input_text":
            texts.append(block["text"])
        elif isinstance(block, dict) and block.get("type") == "text":
            texts.append(block["text"])
        else:
            texts.append(str(block))
    item["content"] = "\n".join(texts) if texts else "(stripped)"
    return item


def _describe_interrupted(completed_items: list) -> list[dict]:
    """Build structured descriptions of what the model had generated before interruption."""
    parts = []
    for item in completed_items:
        item_type = getattr(item, "type", None)
        if item_type == "function_call":
            try:
                args = json.loads(item.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = item.arguments
            parts.append({"tool": item.name, "arguments": args})
        elif item_type == "message":
            for content in getattr(item, "content", []):
                if hasattr(content, "text") and content.text:
                    parts.append({"text": content.text})
    return parts


def _developer_event(event_type: str, **fields) -> dict:
    """Build a developer message with structured event context."""
    payload = {"event": event_type, **fields}
    return {"role": "developer", "content": json.dumps(payload, ensure_ascii=False)}


def _log_exception_tree(e: BaseException, depth: int = 0) -> None:
    """Recursively log a BaseExceptionGroup tree so TaskGroup wrappers don't
    swallow the real cause."""
    indent = "  " * depth
    logger.error(f"{indent}{type(e).__name__}: {e}", exc_info=e)
    for sub in getattr(e, "exceptions", ()) or ():
        _log_exception_tree(sub, depth + 1)


class _SessionState:
    """Per-session lock, cancellation event, and passive event queue."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.cancel: asyncio.Event | None = None
        self.passive_queue: list[Event] = []


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

        binaries_config = config.get("binaries", {})
        binary_dir = Path(binaries_config.get("store", "/data/binaries"))
        retention = int(binaries_config.get("retention_days", 30))
        self.binaries = BinaryStore(store_dir=binary_dir, retention_days=retention)

        self._sessions: dict[str, _SessionState] = {}

        self.mcp_connections: dict[str, MCPConnection] = {}
        self.openai_tools: list[dict] = [VIEW_BINARY_TOOL]
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
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                logger.error(f"Failed to connect to MCP server {name} at {url}: {e!r}")
                _log_exception_tree(e)

        logger.info(f"Connected to {len(self.mcp_connections)} MCP servers, {len(self.openai_tools)} tools total")

    def _get_session(self, session_id: str) -> _SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = _SessionState()
        return self._sessions[session_id]

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
        if name == VIEW_BINARY_TOOL_NAME:
            return self._view_binary(call_id, arguments)

        conn = self._tool_to_mcp.get(name)

        if conn is None:
            return {"type": "function_call_output", "call_id": call_id, "output": f"Error: unknown tool '{name}'"}, []

        content_blocks = await conn.call_tool(name, arguments, store=self.binaries)
        logger.debug(f"  {name} returned {len(content_blocks)} block(s): {[getattr(b, 'type', type(b).__name__) for b in content_blocks]}")
        openai_parts = mcp_content_to_openai(content_blocks, store=self.binaries)

        # input_text → function_call_output.output (a single string)
        # input_image → separate user-role message (images can't ride inside
        # function_call_output.output, which is string-only)
        text_parts = [p["text"] for p in openai_parts if p.get("type") == "input_text"]
        image_items = [
            {"role": "user", "content": [p]}
            for p in openai_parts if p.get("type") == "input_image"
        ]

        output = {"type": "function_call_output", "call_id": call_id, "output": "\n".join(text_parts)}
        return output, image_items

    def _view_binary(self, call_id: str, arguments: str) -> tuple[dict, list[dict]]:
        """Built-in: resolve a pointer into the current turn's input."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            pointer = args.get("pointer", "")
            if not pointer:
                raise ValueError("pointer is required")
            if pointer.startswith(POINTER_PREFIX):
                pointer = pointer[len(POINTER_PREFIX):]
            content, mime = self.binaries.load(pointer)
        except Exception as e:
            return (
                {"type": "function_call_output", "call_id": call_id, "output": f"Error: {e}"},
                [],
            )

        if mime.startswith("image/"):
            part = {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{base64.b64encode(content).decode()}",
            }
            output = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"[loaded image, pointer={pointer}]",
            }
            return output, [{"role": "user", "content": [part]}]

        output = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": f"[binary {pointer} ({mime}, {len(content)} bytes) — non-visual, not loaded]",
        }
        return output, []

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
            elif event_type == "response.failed":
                resp = getattr(event, "response", None)
                status = getattr(resp, "status", "unknown")
                error = getattr(resp, "error", None)
                model = getattr(resp, "model", "unknown")
                logger.error(f"LLM stream failed: status={status} model={model} error={error}")
                return None, completed_items

        return response, completed_items

    async def handle_event(self, event: Event) -> str | None:
        """Process an inbound event from any adapter.

        Event energy determines behavior:
          - "active" (default): cancel in-progress generation, process immediately
          - "passive": if busy, queue for later; if idle, process normally
        """
        state = self._get_session(event.session_id)

        if event.energy == "passive" and state.lock.locked():
            logger.info(f"Queuing passive event for {event.session_id}: {event.text[:60]}")
            state.passive_queue.append(event)
            return None

        if event.energy == "active" and state.cancel is not None:
            logger.info(f"Interrupting in-progress response for {event.session_id}")
            state.cancel.set()

        async with state.lock:
            cancel = asyncio.Event()
            state.cancel = cancel
            try:
                result = await self._process_events(event.session_id, [event], cancel)
            finally:
                if state.cancel is cancel:
                    state.cancel = None

        # Drain queued passive events as a single batched turn
        if state.passive_queue:
            batch = state.passive_queue
            state.passive_queue = []
            logger.info(f"Draining {len(batch)} passive events for {event.session_id}")
            async with state.lock:
                cancel = asyncio.Event()
                state.cancel = cancel
                try:
                    await self._process_events(event.session_id, batch, cancel)
                finally:
                    if state.cancel is cancel:
                        state.cancel = None

        return result

    async def _process_events(
        self,
        session_id: str,
        events: list[Event],
        cancel: asyncio.Event,
    ) -> str | None:
        """Process one or more events as a single turn with cancellation support."""
        # Load session history
        raw_history = self.session.load_truncated(session_id)

        # Build a developer+user pair for each event
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%A)")
        new_items: list[dict[str, Any]] = []
        for event in events:
            text = event.text or "(attachment)"
            context_msg = _developer_event(
                event.event_type,
                source=event.source,
                session_id=event.session_id,
                time=now,
                energy=event.energy,
                **event.metadata,
            )
            user_msg = {"role": "user", "content": text}
            new_items.append(context_msg)
            new_items.append(user_msg)

        # Build input: history + new events (filter reasoning — output-only type)
        history = [m for m in raw_history if m.get("type") != "reasoning"]
        input_items = history + new_items

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

        logger.info(f"Calling LLM: {len(input_items)} input items, {len(self.openai_tools)} tools, {len(events)} event(s)")

        # Collect all new items to save to history
        all_new_messages = list(new_items)

        for iteration in range(self.max_tool_iterations):
            try:
                response, completed_items = await self._stream_response(call_kwargs, cancel)
            except Exception as e:
                logger.error(f"LLM call failed: {type(e).__name__}: {e!r}", exc_info=True)
                return None

            # --- Interrupted during streaming ---
            if response is None:
                partial = _describe_interrupted(completed_items)
                if partial:
                    all_new_messages.append(_developer_event("interrupted", partial=partial))
                    logger.info(f"Interrupted, partial: {partial}")
                else:
                    logger.info("Interrupted before any output completed")
                self.session.append(session_id, all_new_messages)
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
                    for content in item.content or []:
                        if hasattr(content, "text"):
                            reasoning_text += content.text
                elif item.type == "message":
                    for content in item.content or []:
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
                image_items = []
                for tc in tool_calls:
                    if cancel.is_set():
                        pending = []
                        for t in tool_calls[tool_calls.index(tc):]:
                            try:
                                args = json.loads(t.arguments)
                            except (json.JSONDecodeError, AttributeError):
                                args = t.arguments
                            pending.append({"tool": t.name, "arguments": args})
                        logger.info(f"Interrupted between tool calls, pending: {pending}")
                        all_new_messages.append(_developer_event("interrupted", pending=pending))
                        self.session.append(session_id, all_new_messages)
                        return None

                    logger.info(f"  Tool call: {tc.name}({tc.arguments[:100]})")
                    result, images = await self._execute_tool_call(tc.call_id, tc.name, tc.arguments)
                    logger.info(f"  Result: {result['output'][:200]}")
                    tool_results.append(result)
                    all_new_messages.append(_prepare_for_history(result))
                    for img in images:
                        image_items.append(img)

                # Images go after tool results (Bedrock adjacency) and aren't
                # persisted — pointer lives in the function_call_output.
                call_kwargs["input"] = input_items + response.output + tool_results + image_items
                input_items = call_kwargs["input"]
                continue

            # No tool calls — final response
            if assistant_text:
                all_new_messages.append({"role": "assistant", "content": assistant_text})

            self.session.append(session_id, all_new_messages)

            logger.info(f"Final response: {assistant_text[:200]}")
            return assistant_text

        logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached")
        self.session.append(session_id, all_new_messages)
        return None

    async def close(self) -> None:
        for conn in self.mcp_connections.values():
            await conn.close()

    async def run_binary_gc(self, interval_seconds: int = 3600) -> None:
        """Periodically prune expired binaries. Runs for the process lifetime."""
        while True:
            try:
                self.binaries.gc()
            except Exception as e:
                logger.error(f"Binary GC error: {e}")
            await asyncio.sleep(interval_seconds)
