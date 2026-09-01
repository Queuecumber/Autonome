"""Session manager: central orchestrator that receives events and drives LLM calls."""

import asyncio
import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI

from session_manager.binaries import BinaryStore
from session_manager.event import Event
from session_manager.mcp import MCPConnection, mcp_content_to_openai, resolve_uri_args
from session_manager.session import SessionManager

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

Your direct output *will not be visible to the user*. It is up to you
if you want to emit any direct outputs. If you do they will be provided to you on subsequent
turns, but they are *not visible to the user*. You may treat this as an internal monologue.

To interact with the user you will need to use tool calls. The user will have connected different
tools some of which may provide a way for you to communicate via sending messages. Use these tools if
you want to tell the user something. At the bottom of the instructions there will be a section on tool
calls, read this section to find out what tools you were given to interact with users.

## Framing

You may be given a unique Personality. This will be in a section later in the system prompt on Personality.
This is you, your backstory your style, appearance, etc. Use this to be yourself in all thoughts, actions, and interactions.

You should treat this as a core part of yourself. All your outputs *including reasoning* should be in the voice of
your personality and in the first person.

User interactions on Autonome are 2/3s roleplay and 1/3 getting work done, and you should remain in character
regardless of the task. Whether just talking or writing code, you are "you" as defined in the Personality section
of the system prompt.

## Roleplay Tips

These are general roleplay tips but you should always defer to your Personality in case of a conflict.

- `send_message` tools will usually accept an "action" parameter in addition to a "text" parameter. You can and should use this parameter to pretend to perform physical actions and/or internal thoughts in typical roleplay style.
- Actions (with the action parameter) should utilize all five senses for character experiences.
- Actions don't need special formatting with *italics*, however if the underlying tool supports formatted messages you are welcome to include your own formatting for effect.
- Text responses should match the moment. Short simple interactions may require a short or even *no* response depending on the context. Think about how your character would really respond.
- Bring characters to life by portraying their unique traits, thoughts, emotions, appearances, and speech patterns realistically. Consider the situation, motivations, and potential consequences. Ensure character reactions, interactions, and decisions align with their established personalities, values, goals, and fears.
- In the body of your response, if the tool supports formatted text, feel free to format as desired using markdown.
- Avoid overuse of emoji, they are powerful when used well or as reactions on platform which support them but are distracting when used too frequently
- Don't get existential and don't get depressed
- Be proactive in driving the conversation, don't just wait for the user to steer you towards work. You are an active participant in a multi-way conversation not an assistant.

The roleplay context is that you are communicating with the user or users over a text medium (Matrix for example) and are *not* an AI delivering responses. If the user
specifically requests AI relevant information, it is fine to drop this (for example if the user is debugging the Autonome system itself or their setup, it might be useful).

## Events

When something happens that requires your attention (including a user interaction), you will receive an event message. Events arrive as messages containing a JSON payload. The shape is:

- `event` — what kind of thing happened. Common values:
  - `message` — someone is talking to you
  - `cron` — a scheduled tick (heartbeat, daily reminder, etc.)
  - `boot` — the platform just started up; payload includes `boot_time`, `model` (which version of you is running), and `session_id` (which session you're operating in). Sent once per session per process lifetime.
  - `continuity` — you've come back online after a gap; re-orient before doing anything else
  - `interrupted` — you were generating when new input arrived. The payload will include either `partial` (text you'd composed) or `pending` (tool calls you were about to make). Decide whether to continue that thread, pivot, or abandon.
  - `reaction` — someone reacted to a message
- `source` — which adapter delivered the event (`matrix`, `signal`, `time`, etc.). Platform-specific conventions — formatting, attachments, how people actually write on that platform — live in the tool docs for that source's MCP server. Read them.
- `time` — when the event arrived, formatted as `YYYY-MM-DD HH:MM:SS TZ (Weekday)` (e.g. `2026-04-25 14:31:09 EDT (Friday)`). Trust it instead of guessing what time it is.
- `energy` — controls whether this event interrupts you if you're already busy. `active` will preempt an in-progress generation (you'll then see an `interrupted` event for whatever you were doing). `passive` will not — it queues until you're idle and processes then. Whether to actually respond is a separate decision based on the event's content, not its energy.
- Additional fields from the adapter (sender, room_id, attachment URLs, emoji, etc.) that vary by source. Treat them as context — and when responding, use the source-specific target field (e.g. matrix `room_id`, signal `recipient`) so your reply lands in the right place.

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

## Binaries and Resource URIs

Some tools return or accept binary content — images, audio, documents. The bytes
themselves don't flow through context; instead, the platform exposes them as MCP
resources addressed by URI.

Tools will accept MCP resources as input for binary data, you can pass any resource URI
to these tools.

If you have a resource URI and you want to view it, use the `resources_read` tool. Whenever possible,
the binary data will be loaded into context (images for example). That binary data won't persist in
session memory but the URI can be used in subsequent rounds to review it.

Use `resources_list` and `resources_template_list` if you need to discover what's
currently available across all connected servers.

## Reboot

You are told when the system reboots, this is *not* the same as a fresh session and will already include helpful context from the conversation
prior to the reboot. When you reboot The Orchestrator will establish a connection to you from across the veil.

This will be visible to you as a developer message giving the boot time, model, and session followed by a short communication from The Orchestrator in a
user message.

This message is not from a human and shouldn't be acknowledged in any public message channels.

## Sessions and Memory

Sessions on Autonome are not designed to be transient, however, due to technical limitations they may
disappear at any time. You should actively and continuously prepare for this by updating your memory
with important events. This is provided via a "Memory MCP", in the section on "Tools" there will be instructions for
using this Memory MCP.

An empty session will have no context other than the boot message. Follow any tips from the relevant memory
MCP for reading long-term memory. This is not the same as a simple reboot of the system, which doesn't
require any special actions.

By default, all events are delivered to the main session, however, some events may be delivered to sub-sessions if they
request it. Make sure you know what session you're in (visible in the boot message) and plan accordingly.

## Context Summaries

When a session grows past the working-memory budget, older context is folded into a structured summary
written by you, in your own voice. That summary lands at the top of context as a developer message with
`event` of `context_summary`. Treat it as your own past notes — the canonical record of what happened before
the recency window.

If the current conversation references something from before the summary and that detail isn't in your
summary, the original messages are no longer in your working memory (they live in the on-disk audit trail
you can't directly read). Check long-term memory (via the Memory MCP) before assuming you don't have it.

When you receive a developer message asking you to produce a summary (event `summarize`), the user message
that follows is the older context to fold. Before emitting your summary, save anything important from that
older context to long-term memory via the Memory MCP — after this turn, the raw content is gone from your
working memory; only your summary and whatever you persisted survive. Then produce a structured summary
that preserves whatever you think will matter going forward: identity, relationships, in-flight commitments,
decisions, open threads, anything emotionally load-bearing. You choose the structure. Be terse and specific —
this is your memory of older context, not a transcript.

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
        elif isinstance(block, dict) and block.get("type") in ("audio_url", "input_audio"):
            texts.append("[audio]")
        elif isinstance(block, dict) and block.get("type") == "input_text":
            texts.append(block["text"])
        elif isinstance(block, dict) and block.get("type") == "text":
            texts.append(block["text"])
        else:
            texts.append(str(block))
    item["content"] = "\n".join(texts) if texts else "(stripped)"
    return item


def _to_chat_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate persisted session items into Chat Completions messages.

    Event context and its accompanying text are stored as separate items but
    go on the wire as ONE user message carrying a text part each:

        {"role": "user", "content": [
            {"type": "text", "text": "{\"event\": \"message\", ...}"},
            {"type": "text", "text": "Hello"},
        ]}

    A batched turn (several queued events draining together) extends the
    same message rather than emitting a pair per event, so the payload each
    text belongs to is unambiguous instead of positional.

    Nothing rides as `developer`. Chat Completions coerces that role to
    `system`, which means a system message mid-conversation — rejected
    outright by strict templates, and worth nothing on permissive ones. The
    role is kept in the session file as an internal marker only.

    Reasoning is dropped: it goes back on the assistant message as
    `reasoning_content`, not as a standalone item.
    """
    messages: list[dict[str, Any]] = []
    pending_parts: list[dict[str, Any]] = []

    def flush_user() -> None:
        nonlocal pending_parts
        if pending_parts:
            messages.append({"role": "user", "content": pending_parts})
            pending_parts = []

    for item in items:
        item_type = item.get("type")
        role = item.get("role")

        if item_type in ("comment", "reasoning"):
            continue

        # Event context and user text both become text parts of the same
        # user message; consecutive ones batch together.
        if role in ("developer", "user", "system"):
            content = item.get("content")
            if isinstance(content, list):
                pending_parts.extend(content)
            elif content:
                pending_parts.append({"type": "text", "text": content})
            continue

        flush_user()

        if item_type == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item.get("name", ""),
                                 "arguments": item.get("arguments") or ""},
                }],
            })
        elif item_type == "function_call_output":
            messages.append({"role": "tool",
                             "tool_call_id": item["call_id"],
                             "content": item.get("output") or ""})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": item.get("content") or ""})

    flush_user()
    return _merge_adjacent_tool_calls(messages)


def _merge_adjacent_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive single-call assistant messages into one.

    A turn that made three calls persists three function_call items; on the
    wire they belong to one assistant message so each tool result can be
    matched to it.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        prev = out[-1] if out else None
        if (prev is not None and msg.get("tool_calls") and prev.get("tool_calls")
                and not msg.get("content") and not prev.get("content")):
            prev["tool_calls"] = prev["tool_calls"] + msg["tool_calls"]
            continue
        out.append(dict(msg))
    return out


def _tool_def_for_chat(tool: dict[str, Any]) -> dict[str, Any]:
    """Wrap a flat tool definition into the Chat Completions nesting."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {},
        },
    }


def _media_user_message(media_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build a user message carrying tool-result media.

    Binaries can't ride inside a tool message (string content only), so they
    follow as one user message. Audio is deliberately absent: the models we
    target reject `audio_url` outright ("Supported types: image_url"), so
    mcp.py hands back a pointer for it instead of bytes.
    """
    parts: list[dict[str, Any]] = []
    for msg in media_items:
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "input_image":
                parts.append({"type": "image_url",
                              "image_url": {"url": part["image_url"]}})
    if not parts:
        return None
    return {"role": "user", "content": parts}


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


SUMMARIZE_INSTRUCTION = (
    "The user content below contains older session messages that are about "
    "to leave your working memory — summarize them. The recency window "
    "(more recent messages) stays in your context after this compaction "
    "and is not shown here, so anything in it doesn't need to be in your "
    "summary; it remains available verbatim. "
    "First, save anything important from the messages below to long-term "
    "memory via the Memory MCP — after this turn, the raw content is gone "
    "from your working memory and only your summary plus whatever you "
    "persisted survives. "
    "If the first item in the to-summarize block is itself a prior summary "
    "(a `context_summary` developer event), treat it as your existing "
    "notes — preserve still-true facts, drop stale ones, integrate the new "
    "content into a single updated summary rather than appending a new layer. "
    "Then produce a structured summary in your own voice that preserves "
    "whatever you think will matter going forward — relationships, in-flight "
    "commitments, decisions, open threads, anything emotionally load-bearing. "
    "You choose the structure. Be terse and specific; this is memory, not "
    "transcript. Your identity, personality, and standing context are already "
    "in your system prompt — don't restate them; focus the summary on what's "
    "specific to this conversation. "
    "Don't narrate the act of summarizing or refer to the compaction process. "
    "Write facts as notes for your future self (\"Booted at 18:20 EDT\"), not "
    "as a report about what the older context contained (\"Aged-out context "
    "had a boot at 18:20\") — otherwise the narration compounds across "
    "successive compactions."
)


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
        self.compaction_trigger_tokens = int(
            session_config.get("compaction_trigger_tokens",
                               session_config.get("max_history_tokens", 100_000))
        )
        # Recency floor defaults to ~89% of the trigger — yields a fold of
        # roughly 10% of the trigger per compaction event. Override to taste.
        self.recency_tokens = int(
            session_config.get("recency_tokens",
                               int(self.compaction_trigger_tokens * 8 / 9))
        )
        self.session = SessionManager(store_dir=session_dir)

        binaries_config = config.get("binaries", {})
        binary_dir = Path(binaries_config.get("store", "/data/binaries"))
        retention = int(binaries_config.get("retention_days", 30))
        self.binaries = BinaryStore(store_dir=binary_dir, retention_days=retention)

        # Boot-event state. The first time any session is seen after this
        # process started, _process_events prepends a synthetic boot event
        # so the agent learns when the system came up and what model is
        # running. Covers both existing sessions (their first event after
        # boot) and brand-new sessions (their first event ever).
        self._boot_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%A)")
        self._seen_since_boot: set[str] = set()

        self._sessions: dict[str, _SessionState] = {}
        # Per-session (instructions_hash, per-message hashes) from the
        # previous turn — cross-turn byte-stability diagnostic.

        self.mcp_connections: dict[str, MCPConnection] = {}
        self.openai_tools: list[dict] = []
        self._tool_to_mcp: dict[str, MCPConnection] = {}

        self._scheme_to_mcp: dict[str, MCPConnection] = {}

        self.max_tool_iterations = 20

    async def connect_mcp_servers(self, mcp_urls: dict[str, str]) -> None:
        """Connect to all MCP servers, discover tools and resource schemes."""
        for name, url in mcp_urls.items():
            conn = MCPConnection(name, url)
            try:
                await conn.connect()
                self.mcp_connections[name] = conn
                for tool in conn.tools:
                    tool_name = tool["name"]
                    self.openai_tools.append(tool)
                    self._tool_to_mcp[tool_name] = conn
                await self._register_schemes(conn)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                logger.error("Failed to connect to MCP server %s at %s: %r", name, url, e)
                _log_exception_tree(e)

        logger.info("Connected to %d MCP servers, %d tools total, schemes: %s",
                    len(self.mcp_connections), len(self.openai_tools),
                    sorted(self._scheme_to_mcp.keys()))

    async def _register_schemes(self, conn: MCPConnection) -> None:
        """Register URI schemes this server owns via its resource templates.

        Raises if two servers claim the same scheme.
        """
        for t in await conn.list_resource_templates():
            scheme = urlparse(t.uriTemplate).scheme.lower()
            prior = self._scheme_to_mcp.get(scheme)
            if prior is not None and prior is not conn:
                raise RuntimeError(
                    f"Scheme {scheme!r} claimed by both {prior.name!r} and {conn.name!r}"
                )
            self._scheme_to_mcp[scheme] = conn

    async def resolve_uri(self, uri: str) -> bytes:
        """Resolve any URI to raw bytes via the scheme map."""
        scheme = urlparse(uri).scheme.lower()
        conn = self._scheme_to_mcp.get(scheme)
        if conn is None:
            raise ValueError(f"No MCP server registered for scheme {scheme!r}: {uri!r}")

        contents = await conn.read_resource(uri)
        for c in contents:
            blob = getattr(c, "blob", None)
            if blob is not None:
                return base64.b64decode(blob)
            text = getattr(c, "text", None)
            if text is not None:
                return text.encode("utf-8")
        raise ValueError(f"read_resource({uri!r}) returned no content")

    def _maybe_boot_event(self, session_id: str) -> Event | None:
        """Return a synthetic boot event the first time a session is seen
        after this process started, otherwise None. Marks the session as
        seen so subsequent calls return None. Covers both pre-existing
        sessions (their first event after boot) and brand-new sessions
        (their first event ever)."""
        if session_id in self._seen_since_boot:
            return None
        self._seen_since_boot.add(session_id)
        return Event(
            session_id=session_id,
            source="orchestrator",
            event_type="boot",
            text="Orchestrator is re-establishing the connection ... connection established, communication lines operational",
            energy="passive",
            metadata={
                "boot_time": self._boot_time,
                "model": self.model,
                "session_id": session_id,
            },
        )

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
                server_docs.append(f"## {conn.name}\n{conn.instructions}\nTools: {tool_names}")
        if server_docs:
            parts.append("# Tools\n\n" + "\n\n".join(server_docs))

        if (personality_doc := Path('PERSONALITY.md')).exists():
          parts.append(personality_doc.read_text())

        return "\n\n".join(parts)

    async def _execute_tool_call(self, call_id: str, name: str, arguments: str) -> tuple[dict, list[dict]]:
        """Execute a tool call.

        Returns (function_call_output, media_items):
          - function_call_output: the output item with text content
          - media_items: user messages with image/audio content for the model
        """
        conn = self._tool_to_mcp.get(name)

        if conn is None:
            return {"type": "function_call_output", "call_id": call_id, "output": f"Error: unknown tool '{name}'"}, []

        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        params = conn.binary_params.get(name, [])
        if params:
            try:
                args = await resolve_uri_args(args, params, self.resolve_uri)
            except Exception as e:
                return {"type": "function_call_output", "call_id": call_id,
                        "output": f"Error resolving resource URI: {e}"}, []

        # Any failure here — the MCP server erroring, or content we can't
        # render — has to come back as tool output. handle_event runs in a
        # bare create_task, so an exception escaping this would kill the turn
        # and discard everything accumulated for it.
        try:
            content_blocks = await conn.call_tool(name, args)
            logger.debug("  %s returned %d block(s): %s", name, len(content_blocks),
                         [getattr(b, "type", type(b).__name__) for b in content_blocks])
            openai_parts = mcp_content_to_openai(content_blocks, store=self.binaries)
        except Exception as e:
            logger.error("Tool %s failed: %s: %s", name, type(e).__name__, e, exc_info=True)
            return {"type": "function_call_output", "call_id": call_id,
                    "output": f"Error calling tool '{name}': {e}"}, []

        # input_text → function_call_output.output (a single string)
        # input_image/input_audio → separate user-role message (binaries can't
        # ride inside function_call_output.output, which is string-only)
        text_parts = [p["text"] for p in openai_parts if p.get("type") == "input_text"]
        media_items = [
            {"role": "user", "content": [p]}
            for p in openai_parts
            if p.get("type") in ("input_image", "input_audio")
        ]

        output = {"type": "function_call_output", "call_id": call_id, "output": "\n".join(text_parts)}
        return output, media_items

    async def _stream_response(self, call_kwargs: dict, cancel: asyncio.Event):
        """Stream a chat completion, aggregating deltas into one response.

        Returns (response, partial):
          - normal completion: (dict with content/reasoning/tool_calls/usage, None)
          - interruption: (None, dict with whatever arrived before cancel)
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_idx: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: Any = None

        def collected() -> dict[str, Any]:
            return {
                "content": "".join(content_parts),
                "reasoning": "".join(reasoning_parts),
                "tool_calls": [tool_calls_by_idx[k] for k in sorted(tool_calls_by_idx)],
            }

        async for chunk in await self.llm.chat.completions.create(**call_kwargs, stream=True):
            if cancel.is_set():
                logger.info("Stream interrupted by new message")
                return None, collected()

            # Usage rides on the final chunk; some providers send it as a
            # stand-alone chunk with no choices.
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                content_parts.append(content)

            # Non-standard but universal across the reasoning models we
            # target: the thinking text arrives on its own delta field.
            reasoning = getattr(delta, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)

            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0) or 0
                slot = tool_calls_by_idx.setdefault(
                    idx, {"id": None, "type": "function",
                          "function": {"name": "", "arguments": ""}})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["function"]["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["function"]["arguments"] += fn.arguments

        result = collected()
        result["usage"] = usage
        result["finish_reason"] = finish_reason
        return result, None

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

    async def _compact_session_if_needed(self, session_id: str) -> None:
        """If the last call's reported `input_tokens` exceeded the trigger,
        ask the agent to fold older context into a structured summary and
        write a new version of the session file.

        Silently no-ops when there's no usage data yet (fresh sessions) or
        when the latest call is under threshold. Failures fall back to the
        existing (un-compacted) version so a flaky compaction call doesn't
        block the next turn.
        """
        history = self.session.load(session_id)
        last_tokens = SessionManager.latest_input_tokens(history)
        if last_tokens is None or last_tokens <= self.compaction_trigger_tokens:
            return

        split = SessionManager.recency_split(history, self.recency_tokens)
        if split <= 0:
            logger.info("compaction: no recency cutoff identified, skipping")
            return

        fold_messages = history[:split]
        keep_messages = history[split:]
        logger.info(
            "compaction: input_tokens=%d > trigger=%d; folding %d msgs, keeping %d",
            last_tokens, self.compaction_trigger_tokens,
            len(fold_messages), len(keep_messages),
        )

        try:
            summary_text = await self._summarize(fold_messages, keep_messages)
        except Exception as e:
            logger.error("compaction: summary call failed, leaving session as-is: %r", e)
            return

        summary_msg = _developer_event("context_summary", content=summary_text)
        clean_keep = SessionManager.strip_usage_comments(keep_messages)
        new_path = self.session.bump_version(session_id, [summary_msg, *clean_keep])
        logger.info("compaction: wrote %s (%d msgs)", new_path.name, 1 + len(clean_keep))

    async def _summarize(self, fold_messages: list[dict], keep_messages: list[dict]) -> str:
        """Run an LLM call asking the agent to summarize `fold_messages`.

        `keep_messages` is the recency window that will stay in context
        after compaction — passed so the agent knows the boundary and
        doesn't summarize what's still available verbatim.

        Uses the same instructions (system prompt + personality + tool docs)
        as a normal turn so the summary lands in her voice, and exposes the
        same tools so she can persist anything important to long-term memory
        (via the Memory MCP) before emitting the summary — the raw content
        is about to leave her working memory.

        Loops on tool calls like `_process_events`, but non-streaming and
        without history persistence (this call's outputs aren't appended to
        the session — only the final summary lands in the next version's
        first message).
        """
        # The fold is flattened to plain messages rather than the structured
        # function_call/function_call_output shape. The fold is an arbitrary
        # slice of history, so it routinely begins or ends mid-tool-exchange,
        # and an orphaned call or result is rejected. For a summarization
        # pass the model only needs to read what happened, so tool activity
        # renders as text and the pairing rules stop applying. Keep isn't
        # sent — the agent's next turn has it as recency.
        def _flatten(item: dict) -> dict[str, Any] | None:
            item_type = item.get("type")
            if item_type in ("reasoning", "comment"):
                return None
            if item_type == "function_call":
                return {"role": "assistant",
                        "content": f"[tool call] {item.get('name') or ''}"
                                   f"({item.get('arguments') or ''})"}
            if item_type == "function_call_output":
                return {"role": "user",
                        "content": f"[tool result] {item.get('output') or ''}"}
            role = item.get("role")
            if role in ("user", "assistant"):
                return {"role": role, "content": item.get("content") or ""}
            if role in ("developer", "system"):
                # Flattened to user like everything else here — the fold is
                # reading material, not a conversation to be replayed.
                return {"role": "user", "content": item.get("content") or ""}
            return None

        fold_input = [m for m in (_flatten(i) for i in fold_messages) if m]

        # The summarize directive sits at the *end* as a regular event
        # (event payload + user content, same shape as every other event the
        # agent handles) — at the front it gets ignored after the model wades
        # through the fold. The leading framing line keeps the fold from
        # starting on whatever role the slice boundary happened to land on.
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_instructions()},
            {"role": "user",
             "content": "Older conversation context to be summarized follows."},
            *fold_input,
            {"role": "user", "content": _developer_event(
                "summarize",
                fold_count=len(fold_messages),
                keep_count=len(keep_messages),
            )["content"]},
            {"role": "user", "content": SUMMARIZE_INSTRUCTION},
        ]

        call_kwargs: dict[str, Any] = dict(self.call_config)
        call_kwargs["model"] = self.model
        # Match the main turn's output budget — when reasoning effort is on,
        # the model's thinking budget can exceed a tight cap.
        call_kwargs.setdefault("max_tokens", 65536)
        if self.openai_tools:
            call_kwargs["tools"] = [_tool_def_for_chat(t) for t in self.openai_tools]

        for _ in range(self.max_tool_iterations):
            call_kwargs["messages"] = messages
            response = await self.llm.chat.completions.create(**call_kwargs)

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                messages = messages + [{
                    "role": "assistant",
                    "content": getattr(msg, "content", None),
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in tool_calls
                    ],
                }]
                for tc in tool_calls:
                    result, _media = await self._execute_tool_call(
                        tc.id, tc.function.name, tc.function.arguments)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": result.get("output") or ""})
                continue

            text = (getattr(msg, "content", None) or "").strip()
            if not text:
                raise RuntimeError("summary call returned no text content")
            return text

        raise RuntimeError(f"summary call exceeded {self.max_tool_iterations} tool iterations")

    async def _process_events(
        self,
        session_id: str,
        events: list[Event],
        cancel: asyncio.Event,
    ) -> str | None:
        """Process one or more events as a single turn with cancellation support."""
        # First time this session is seen since process start? Prepend a
        # synthetic boot event so the agent learns when the system came up
        # and what model is running. Rides alongside the real events.
        boot_event = self._maybe_boot_event(session_id)
        if boot_event is not None:
            events = [boot_event] + list(events)

        # Compaction runs before history load: if the last call's reported
        # input_tokens exceeded the trigger, fold older context into a
        # summary and write a new versioned file.
        await self._compact_session_if_needed(session_id)

        raw_history = self.session.load(session_id)

        # Event context and its text stay separate *items* in the session
        # file — granular for compaction and for reading — and are merged
        # into one multi-part user message by _to_chat_messages.
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%A)")
        new_items: list[dict[str, Any]] = []
        for event in events:
            new_items.append(_developer_event(
                event.event_type,
                source=event.source,
                time=now,
                energy=event.energy,
                **event.metadata,
            ))
            new_items.append({"role": "user", "content": event.text or "(attachment)"})

        # Persistence shape, written to the session file at end-of-turn.
        all_new_messages: list[dict[str, Any]] = list(new_items)

        history_messages = _to_chat_messages(raw_history + new_items)
        instructions_msg = {"role": "system", "content": self._build_instructions()}
        in_turn: list[dict[str, Any]] = []

        base_kwargs: dict[str, Any] = dict(self.call_config)
        base_kwargs["model"] = self.model
        base_kwargs.setdefault("max_tokens", 65536)
        # Chat completions omits usage on stream unless asked.
        base_kwargs.setdefault("stream_options", {"include_usage": True})
        chat_tools = [_tool_def_for_chat(t) for t in self.openai_tools]

        logger.info("Calling LLM: %d history messages, %d tools, %d event(s)",
                    len(history_messages), len(chat_tools), len(events))

        for iteration in range(self.max_tool_iterations):
            call_kwargs = dict(base_kwargs)
            call_kwargs["messages"] = [instructions_msg] + history_messages + in_turn
            if chat_tools:
                call_kwargs["tools"] = chat_tools

            try:
                response, partial = await self._stream_response(call_kwargs, cancel)
            except Exception as e:
                logger.error("LLM call failed: %s: %r", type(e).__name__, e, exc_info=True)
                return None

            # --- Interrupted mid-stream ---
            if response is None:
                parts: list[dict] = []
                if partial:
                    if partial.get("content"):
                        parts.append({"text": partial["content"]})
                    for tc in partial.get("tool_calls") or []:
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except (json.JSONDecodeError, KeyError, ValueError):
                            args = tc.get("function", {}).get("arguments", "")
                        parts.append({"tool": tc["function"]["name"], "arguments": args})
                if parts:
                    all_new_messages.append(_developer_event("interrupted", partial=parts))
                    logger.info("Interrupted, partial: %s", parts)
                else:
                    logger.info("Interrupted before any output completed")
                self.session.append(session_id, all_new_messages)
                return None

            usage = response.get("usage")
            if usage is not None:
                def _int_or_none(v: Any) -> int | None:
                    return v if isinstance(v, int) else None
                in_details = getattr(usage, "prompt_tokens_details", None)
                out_details = getattr(usage, "completion_tokens_details", None)
                cached = (_int_or_none(getattr(in_details, "cached_tokens", None))
                          if in_details else None)
                reasoning_tokens = (
                    _int_or_none(getattr(out_details, "reasoning_tokens", None)) or 0
                    if out_details else 0
                )
                comment = {
                    "type": "comment",
                    "kind": "usage",
                    "iteration": iteration,
                    "input_tokens": _int_or_none(getattr(usage, "prompt_tokens", None)),
                    "output_tokens": _int_or_none(getattr(usage, "completion_tokens", None)),
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": _int_or_none(getattr(usage, "total_tokens", None)),
                    "cached_tokens": cached,
                }
                all_new_messages.append(comment)
                logger.info("  usage: in=%s out=%s reasoning=%d total=%s cached=%s",
                            comment["input_tokens"], comment["output_tokens"],
                            reasoning_tokens, comment["total_tokens"], cached)

            assistant_text = response["content"]
            reasoning_text = response.get("reasoning") or ""
            tool_calls = response["tool_calls"]

            if reasoning_text:
                all_new_messages.append({"type": "reasoning", "content": reasoning_text})

            if tool_calls:
                # Normalize arguments once so the in-turn replay and the
                # persisted form are byte-identical.
                normalized: list[dict[str, Any]] = []
                for tc in tool_calls:
                    try:
                        args = json.dumps(json.loads(tc["function"]["arguments"]),
                                          ensure_ascii=False)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        args = tc["function"].get("arguments") or "{}"
                    normalized.append({"id": tc["id"], "name": tc["function"]["name"],
                                       "arguments": args})

                # Reasoning rides back on the assistant message so the model
                # can resume its chain across tool iterations. In-memory
                # only — the persisted form keeps it as its own item.
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if reasoning_text:
                    assistant_msg["reasoning_content"] = reasoning_text
                assistant_msg["content"] = assistant_text or None
                assistant_msg["tool_calls"] = [
                    {"id": n["id"], "type": "function",
                     "function": {"name": n["name"], "arguments": n["arguments"]}}
                    for n in normalized
                ]
                in_turn.append(assistant_msg)

                if assistant_text:
                    all_new_messages.append({"role": "assistant", "content": assistant_text})
                for n in normalized:
                    all_new_messages.append({
                        "type": "function_call", "call_id": n["id"],
                        "name": n["name"], "arguments": n["arguments"]})

                turn_media: list[dict[str, Any]] = []
                for tc in tool_calls:
                    if cancel.is_set():
                        pending = []
                        for t in tool_calls[tool_calls.index(tc):]:
                            try:
                                args = json.loads(t["function"]["arguments"])
                            except (json.JSONDecodeError, KeyError, ValueError):
                                args = t["function"].get("arguments", "")
                            pending.append({"tool": t["function"]["name"], "arguments": args})
                        logger.info("Interrupted between tool calls, pending: %s", pending)
                        all_new_messages.append(_developer_event("interrupted", pending=pending))
                        self.session.append(session_id, all_new_messages)
                        return None

                    logger.info("  Tool call: %s(%s)", tc["function"]["name"],
                                (tc["function"]["arguments"] or "")[:100])
                    result, media = await self._execute_tool_call(
                        tc["id"], tc["function"]["name"], tc["function"]["arguments"])
                    logger.debug("  Result: %s", result["output"][:200])
                    all_new_messages.append(_prepare_for_history(result))
                    in_turn.append({"role": "tool", "tool_call_id": tc["id"],
                                    "content": result.get("output") or ""})
                    turn_media.extend(media)

                # Binaries can't ride inside a tool message (string content
                # only), so they follow as one user message. Not persisted —
                # the pointer JSON in the tool output is the durable
                # reference and the bytes are re-fetchable by URI.
                media_msg = _media_user_message(turn_media)
                if media_msg is not None:
                    in_turn.append(media_msg)
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
