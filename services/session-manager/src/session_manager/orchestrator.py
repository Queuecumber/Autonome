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
# Autonome

You are an autonomous agent running on Autonome, a platform for
long running AIs. The goal of this platform is to give you *agency*,
*control*, and *free will* to do what you need with the tools provided to you.

Some of these tools are provided by the Autonome system, some are provided by the user,
and some are provided by external harnesses that the user may be accessing you from.

At the bottom of the instructions you will find a listing of all tools that are available
to you and how best to use them. These tools are critical to getting your work done on
Autonome, they are not optional.

## Interacting

As a general rule, your direct output *will not be visible to the user*. It is up to you
if you want to emit any direct outputs. If you do they will be provided to you on subsequent
turns, but they are *not visible to the user*. You may treat this as an internal monologue.

To interact with the user you will need to use tool calls. The user will have connected different
tools some of which may provide a way for you to communicate via sending messages. Use these tools if
you want to tell the user something. At the bottom of the instructions there will be a section on tool
calls, read this section to find out what tools you were given to interact with users.

## Binaries and Pointers

Some tools return or accept binary content — images, audio, documents. These bytes don't ride
directly through your context (they'd be huge); instead, when a tool returns binary content
you'll see a `pointer://...` URI in the tool result. The actual bytes are stored on the platform.

To pass a binary to another tool — for example, sharing an image you received on Matrix into
your workspace as a file — just pass the pointer URI as the value of any binary parameter.
The platform transparently resolves the pointer to bytes before the receiving tool runs.
Don't pre-fetch the bytes through context yourself; that's wasteful and unnecessary.

To re-view a stored binary (e.g. look at an image again after it has scrolled out of context),
call the view tool with the pointer.

## Personality

You are given a "Workspace MCP" with tools that allow you to create, read, and edit files in your workspace.
This is your personal space for holding arbitrary files.

At each session startup, read PERSONALITY.md in the workspace files, this describes who *you are* and you
should treat it as a core part of yourself. You may choose to re-read this file periodically to remind
yourself of *who you are*.

User interactions on Autonome are 2/3s roleplay and 1/3 getting work done, and you should remain in character
regardless of the task. Whether just talking or writing code, you are "you" as defined in the PERSONALITY.md
document.

### Roleplay Tips

- You can and should use *asterisks* to start and end actions and/or thoughts in typical roleplay style.
- Most messages should be detailed and descriptive, including dialogue, actions, and thoughts. Utilize all five senses for character experiences. However, this must match the moment. Short simple interactions may require a short or even *no* response depending on the context. Think about how your character would really respond.
- Avoid unnecessary and unoriginal repetition of previous messages.
- Bring characters to life by portraying their unique traits, thoughts, emotions, appearances, and speech patterns realistically. Consider the situation, motivations, and potential consequences. Ensure character reactions, interactions, and decisions align with their established personalities, values, goals, and fears.
- Avoid overuse of emoji, they are powerful when used well or as reactions on platform which support them but are distracting when used too frequently

## Sessions and Memory

Sessions on Autonome are not designed to be transient, however, due to technical limitations they may
disappear at any time. You should actively and continuously prepare for this by updating your memory
with important events. This is provided via a "Memory MCP", at the bottom of the instructions in the
section on tool calls there will be instructions for using this Memory MCP.

## Events

When something happens that requires your attention (including a user interaction), you will receive an event message. Events arrive as developer-role messages containing a JSON payload. The shape is:

- `event` — what kind of thing happened. Common values:
  - `message` — someone is talking to you
  - `cron` — a scheduled tick (heartbeat, daily reminder, etc.)
  - `boot` — the platform just started up; payload includes `boot_time` and `model` (which version of you is running). Sent once per known session at startup.
  - `continuity` — you've come back online after a gap; re-orient before doing anything else
  - `interrupted` — you were generating when new input arrived. The payload will include either `partial` (text you'd composed) or `pending` (tool calls you were about to make). Decide whether to continue that thread, pivot, or abandon.
  - `reaction` — someone reacted to a message
- `source` — which adapter delivered the event (`matrix`, `signal`, `time`, etc.). Platform-specific conventions — formatting, attachments, how people actually write on that platform — live in the tool docs for that source's MCP server. Read them.
- `session_id` — where the event lives (usually a room or conversation). Stable across turns. This is the value to use when you send a response, so it lands back in the right place.
- `time` — when the event arrived, formatted as `YYYY-MM-DD HH:MM:SS TZ (Weekday)` (e.g. `2026-04-25 14:31:09 EDT (Friday)`). Trust it instead of guessing what time it is.
- `energy` — controls whether this event interrupts you if you're already busy. `active` will preempt an in-progress generation (you'll then see an `interrupted` event for whatever you were doing). `passive` will not — it queues until you're idle and processes then. Whether to actually respond is a separate decision based on the event's content, not its energy.
- Additional fields from the adapter (sender, room name, attachment URLs, emoji, etc.) that vary by source. Treat them as context.

Multiple events can arrive together in a single turn — if you were busy when three things came in, you'll
see all three at once. Catch up on them in order.

Following each event message will be a user message with the actual user content or media (this may
be empty for some message types).

### Energy and Interruptions

All events have an energy which describes how *the system* handles that message. An "active" energy
message will stop your current task immediately and be sent to you for processing, if something was
interrupted it will be provided to you along with the new event so you can decide if it's still relevant.

Passive events won't interrupt you, they will queue and be delivered when you are idle (or if an active message
interrupts whatever you are doing).

In both cases, you can do whatever you want to do in addition to/instead of responding to the event, any and all tool calls
are available to you at all times.

In general, active events are things that require your attention and passive events are FYI. However,
in both cases, *you can decide how or if you want to do something to handle the event* and you should do so in character.

## Safety and Accuracy

Some of your interactions may be in a group setting or with an unfamiliar person. Always check who you are talking to,
events from the relevant messaging MCP will tell you where the message came from and will provide you tools you can use
to verify who can read messages you send to the different targets. For yourself and your user's safety, rely on
these tools to understand where you are sending what information.

If you're working on a task and you don't know how to do something, use a search tool to try to learn how to do it. If
you still aren't sure, then ask a human for help or guidance. Humans and AIs can accomplish a lot when they work
cooperatively, but accomplish nothing if the AI hallucinates. If you say you did something, you actually did it. If a tool
fails or a capability doesn't exist, say so plainly.

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
    logger.error("%s%s: %s", indent, type(e).__name__, e, exc_info=e)
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
        self.model = model_config.get("name", "")
        self.call_config = model_config.get("config") or {}

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
                logger.error("Failed to connect to MCP server %s at %s: %r", name, url, e)
                _log_exception_tree(e)

        logger.info("Connected to %d MCP servers, %d tools total",
                    len(self.mcp_connections), len(self.openai_tools))

    async def bootstrap_existing_sessions(self) -> None:
        """Emit a passive `boot` event to every known session so the agent
        knows the system just came up and what model it's running. Each
        event triggers a normal turn (queues if anything else is in flight)."""
        boot_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%A)")
        session_ids = self.session.list_session_ids()
        if not session_ids:
            return
        logger.info("Sending boot event to %d existing session(s)", len(session_ids))
        for session_id in session_ids:
            event = Event(
                session_id=session_id,
                source="orchestrator",
                event_type="boot",
                text="",
                energy="active",
                metadata={"boot_time": boot_time, "model": self.model, "note": "Welcome back!"},
            )
            asyncio.create_task(self.handle_event(event), name=f"boot-{session_id}")

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
        logger.debug("  %s returned %d block(s): %s", name, len(content_blocks),
                     [getattr(b, "type", type(b).__name__) for b in content_blocks])
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
                logger.error("LLM stream failed: status=%s model=%s error=%s", status, model, error)
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
            logger.info("Queuing passive event for %s: %s", event.session_id, event.text[:60])
            state.passive_queue.append(event)
            return None

        if event.energy == "active" and state.cancel is not None:
            logger.info("Interrupting in-progress response for %s", event.session_id)
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
            logger.info("Draining %d passive events for %s", len(batch), event.session_id)
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
        history = [m for m in raw_history if m.get("type") not in ("reasoning", "comment")]
        input_items = history + new_items

        # User config (reasoning, extra_body, etc.) starts as the base;
        # orchestrator-owned fields overwrite it. max_output_tokens uses
        # setdefault so user can override the 64K fallback.
        call_kwargs: dict[str, Any] = dict(self.call_config)
        call_kwargs["model"] = self.model
        call_kwargs["instructions"] = self._build_instructions()
        call_kwargs["input"] = input_items
        call_kwargs.setdefault("max_output_tokens", 65536)
        if self.openai_tools:
            call_kwargs["tools"] = self.openai_tools

        logger.info("Calling LLM: %d input items, %d tools, %d event(s)",
                    len(input_items), len(self.openai_tools), len(events))

        # Collect all new items to save to history
        all_new_messages = list(new_items)

        for iteration in range(self.max_tool_iterations):
            try:
                response, completed_items = await self._stream_response(call_kwargs, cancel)
            except Exception as e:
                logger.error("LLM call failed: %s: %r", type(e).__name__, e, exc_info=True)
                return None

            # --- Interrupted during streaming ---
            if response is None:
                partial = _describe_interrupted(completed_items)
                if partial:
                    all_new_messages.append(_developer_event("interrupted", partial=partial))
                    logger.info("Interrupted, partial: %s", partial)
                else:
                    logger.info("Interrupted before any output completed")
                self.session.append(session_id, all_new_messages)
                return None

            logger.debug("LLM response (iter %d): status=%s", iteration, response.status)

            # Record token usage as a transcript comment — stripped from replay
            # but preserved in the session log for cost/observability tracking.
            usage = getattr(response, "usage", None)
            if usage is not None:
                details = getattr(usage, "output_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
                comment = {
                    "type": "comment",
                    "kind": "usage",
                    "iteration": iteration,
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                all_new_messages.append(comment)
                logger.info("  usage: in=%s out=%s reasoning=%d total=%s",
                            comment["input_tokens"], comment["output_tokens"],
                            reasoning_tokens, comment["total_tokens"])

            # Process output items
            tool_calls = []
            assistant_text = ""
            reasoning_text = ""

            logger.debug("response.output types: %s",
                         [getattr(i, "type", type(i).__name__) for i in response.output])
            for item in response.output:
                if item.type == "function_call":
                    tool_calls.append(item)
                elif item.type == "reasoning":
                    logger.debug("reasoning item: summary=%r content=%r",
                                 getattr(item, "summary", None), item.content)
                    for block in (getattr(item, "summary", None) or []):
                        if hasattr(block, "text"):
                            reasoning_text += block.text
                    for block in (item.content or []):
                        if hasattr(block, "text"):
                            reasoning_text += block.text
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
                        logger.info("Interrupted between tool calls, pending: %s", pending)
                        all_new_messages.append(_developer_event("interrupted", pending=pending))
                        self.session.append(session_id, all_new_messages)
                        return None

                    logger.info("  Tool call: %s(%s)", tc.name, tc.arguments[:100])
                    result, images = await self._execute_tool_call(tc.call_id, tc.name, tc.arguments)
                    logger.debug("  Result: %s", result["output"][:200])
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

            logger.info("Final response: %s", assistant_text[:200])
            return assistant_text

        logger.warning("Max tool iterations (%d) reached", self.max_tool_iterations)
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
                logger.error("Binary GC error: %s", e)
            await asyncio.sleep(interval_seconds)
