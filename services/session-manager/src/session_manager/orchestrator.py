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


REASONING_REPLAY_NOTE = """\
# Your Prior Reasoning

This backend can't carry your thinking from one turn to the next, so the platform
replays it for you. Immediately before each of your earlier assistant messages you
will see a developer message shaped:

    {"event": "reasoning", "content": "..."}

That content is *your own thinking* from that moment — not input from anyone else,
and not something you need to reply to. Read it as the continuation of your own
thought process and pick up where it left off. It is the only record you have of
why you did what you did on earlier turns, so use it when deciding what to do now.

Your current turn's thinking goes wherever it normally does; never write into this
channel yourself.
"""


def _reasoning_dev_message(text: str, developer_role: str) -> dict[str, Any]:
    """Carry reasoning text on the developer channel.

    Backends whose gateway drops the native `reasoning_content` field can
    still see their own prior thinking this way — ugly, but it's the only
    shape that survives. Paired with REASONING_REPLAY_NOTE, which tells the
    model what these messages are.
    """
    return {
        "role": developer_role,
        "content": json.dumps({"event": "reasoning", "content": text},
                              ensure_ascii=False),
    }


def _is_anthropic_model(model: str) -> bool:
    """Is this an Anthropic model reached through a translating gateway?

    Two behaviors key off this. The Bedrock translation hoists every
    system-role message into the top-level system array, so developer
    events have to ride as user messages to keep their position in the
    conversation (and to keep the front of the wire prefix stable).
    Anthropic also replays thinking in-turn via signed `thinking_blocks`
    rather than by echoing reasoning text back, so persisted reasoning —
    which has no signature — stays dropped there.
    """
    m = model.lower()
    return "anthropic" in m or "claude" in m


def _to_chat_messages(
    items: list[dict[str, Any]],
    *,
    developer_role: str = "user",
    preserve_reasoning: bool = False,
) -> list[dict[str, Any]]:
    """Translate persisted session items (Responses-API shape) to Chat
    Completions messages.

    Persisted items come in these flavors:
      - role-based: {role: user/assistant/developer, content: str}
      - function_call: {type, call_id, name, arguments}
      - function_call_output: {type, call_id, output}
      - reasoning: {type, content} — replayed as `reasoning_content` on the
        assistant message it preceded when `preserve_reasoning`, else dropped
      - comment: {type: comment, ...} — dropped (telemetry)

    Output is the chat-completions form Anthropic expects on the wire:
      - assistant messages may carry `tool_calls`; adjacent function_call
        items are merged into one assistant message so all tool_use blocks
        appear together (matches Anthropic's requirement)
      - function_call_output → {role: tool, tool_call_id, content}
      - developer-role events → `developer_role` messages

    `preserve_reasoning` is for models trained with preserved thinking
    history (Kimi K3), which expect the whole assistant message — reasoning
    and tool_calls, not just content — handed back verbatim on every
    subsequent turn. Without it they stop emitting thinking altogether, and
    since nothing then gets persisted the next turn is equally bare.
    """
    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []
    pending_text = ""
    pending_reasoning = ""

    def flush_assistant() -> None:
        nonlocal pending_text, pending_calls, pending_reasoning
        if not pending_calls and not pending_text:
            # Reasoning arrives before the message it belongs to, so hold it
            # rather than emitting a content-less assistant message.
            return
        msg: dict[str, Any] = {"role": "assistant"}
        if pending_reasoning:
            # Native field first (correct wherever it survives), then the
            # developer-channel copy for gateways that strip it.
            msg["reasoning_content"] = pending_reasoning
            messages.append(_reasoning_dev_message(pending_reasoning, developer_role))
        if pending_calls:
            msg["tool_calls"] = pending_calls
            msg["content"] = pending_text or None
        else:
            msg["content"] = pending_text
        messages.append(msg)
        pending_calls = []
        pending_text = ""
        pending_reasoning = ""

    for item in items:
        item_type = item.get("type")
        role = item.get("role")

        if item_type == "function_call":
            pending_calls.append({
                "id": item["call_id"],
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or "",
                },
            })
        elif item_type == "function_call_output":
            flush_assistant()
            messages.append({
                "role": "tool",
                "tool_call_id": item["call_id"],
                "content": item.get("output") or "",
            })
        elif item_type == "reasoning":
            if preserve_reasoning:
                pending_reasoning = item.get("content") or ""
        elif item_type == "comment":
            continue
        elif role == "assistant":
            flush_assistant()
            pending_text = item.get("content") or ""
        elif role == "developer":
            # On the Anthropic paths this is "user", not "developer": the
            # Bedrock translation hoists every system-role message into the
            # top-level system array, so system-role events (a) lose their
            # position in the conversation and (b) mutate the front of the
            # wire prefix each turn, which invalidates every cache entry
            # behind it (verified: an added tail event collapses the cache
            # read to the system-section size; the identical structure with
            # user-role events extends the cache and pays only for the new
            # tokens). Backends that take `developer` natively get it.
            flush_assistant()
            pending_reasoning = ""
            messages.append(
                {"role": developer_role, "content": item.get("content") or ""})
        elif role in ("user", "system"):
            flush_assistant()
            pending_reasoning = ""
            messages.append({"role": role, "content": item.get("content") or ""})

    flush_assistant()
    return messages


def _image_user_message(image_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build a chat-completions user message carrying tool-result images.

    `image_items` are Responses-shape user messages from _execute_tool_call
    ({"role": "user", "content": [{"type": "input_image", "image_url": "data:..."}]}).
    Images can't ride inside a tool-role message (string content only), so
    the convention is a follow-up user message after the tool results.
    Translates input_image -> chat-completions image_url. Returns None if
    there are no images.
    """
    parts: list[dict[str, Any]] = []
    for msg in image_items:
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "input_image":
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": part["image_url"]},
                })
    if not parts:
        return None
    return {"role": "user", "content": parts}


def _tool_def_for_chat(tool: dict[str, Any]) -> dict[str, Any]:
    """Wrap a Responses-shape function tool into Chat Completions shape."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {},
        },
    }


_CACHE_DIRECTIVE: dict[str, Any] = {"type": "ephemeral", "ttl": "1h"}


def _with_cache_breakpoint(msg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of msg with cache_control attached to its last text block.

    String content gets promoted to a single-block list with the directive.
    List content gets the directive added to its last block. Messages that
    don't carry cacheable text (e.g. an assistant message with only
    tool_calls) are returned unchanged.
    """
    content = msg.get("content")
    if isinstance(content, str) and content:
        return {**msg, "content": [{"type": "text", "text": content, "cache_control": _CACHE_DIRECTIVE}]}
    if isinstance(content, list) and content:
        new_content = [dict(b) if isinstance(b, dict) else b for b in content]
        if isinstance(new_content[-1], dict):
            new_content[-1] = {**new_content[-1], "cache_control": _CACHE_DIRECTIVE}
            return {**msg, "content": new_content}
    return msg


def _cache_last_msg(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a cache breakpoint to the last cacheable message in the list.

    Walks back from the tail to find a message with text content; assistant
    messages that only carry tool_calls (no text) get skipped. Returns a new
    list; the original is untouched.
    """
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        content = out[i].get("content")
        has_text = (isinstance(content, str) and content) or (
            isinstance(content, list) and content
        )
        if has_text:
            out[i] = _with_cache_breakpoint(out[i])
            break
    return out


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

        # Anthropic-on-Bedrock needs developer events rendered as user (the
        # translation hoists system-role messages out of position) and can't
        # replay unsigned reasoning. Other backends take `developer`
        # natively, and Kimi-style models need their own prior thinking
        # handed back to keep reasoning at all.
        anthropic = _is_anthropic_model(self.model)
        self.developer_role = "user" if anthropic else "developer"
        self.replay_reasoning = not anthropic

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
        self._prefix_hashes: dict[str, tuple[str, list[str]]] = {}

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

        if self.replay_reasoning:
            parts.append(REASONING_REPLAY_NOTE)

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

        Returns (function_call_output, image_items):
          - function_call_output: the output item with text content
          - image_items: user messages with image_url content for the model to see
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

        content_blocks = await conn.call_tool(name, args)
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

    async def _stream_response(self, call_kwargs: dict, cancel: asyncio.Event):
        """Stream a chat completion and aggregate deltas into a single
        completed response.

        Returns (response, partial):
          - On normal completion: (dict with content/tool_calls/usage/finish_reason, None)
          - On interruption: (None, dict with whatever content/tool_calls
            were collected before cancel)
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        thinking_blocks: list[dict[str, Any]] = []  # by position, accumulated across chunks
        tool_calls_by_idx: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: Any = None

        async for chunk in await self.llm.chat.completions.create(**call_kwargs, stream=True):
            if cancel.is_set():
                logger.info("Stream interrupted by new message")
                return None, {
                    "content": "".join(content_parts),
                    "reasoning": "".join(reasoning_parts),
                    "thinking_blocks": [b for b in thinking_blocks if b.get("thinking") or b.get("signature")],
                    "tool_calls": [tool_calls_by_idx[k] for k in sorted(tool_calls_by_idx)],
                }

            # Usage typically rides on the final chunk (some providers send
            # it as a stand-alone chunk with empty choices).
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                content_parts.append(content)

            # Extended-thinking text — non-standard chat completions field
            # surfaced by some providers (mapped from Anthropic thinking blocks).
            reasoning = getattr(delta, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)

            # Structured thinking blocks with signatures — needed for in-turn
            # replay so the model can resume its reasoning chain across tool
            # calls. Accumulate by position; each delta may carry partial
            # thinking text and a final signature.
            tbs_delta = getattr(delta, "thinking_blocks", None)
            if isinstance(tbs_delta, list):
                for i, tb in enumerate(tbs_delta):
                    while len(thinking_blocks) <= i:
                        thinking_blocks.append({"type": "thinking", "thinking": ""})
                    block = thinking_blocks[i]
                    tb_type = tb.get("type") if isinstance(tb, dict) else getattr(tb, "type", None)
                    if isinstance(tb_type, str):
                        block["type"] = tb_type
                    tb_thinking = tb.get("thinking") if isinstance(tb, dict) else getattr(tb, "thinking", None)
                    if isinstance(tb_thinking, str) and tb_thinking:
                        block["thinking"] += tb_thinking
                    tb_signature = tb.get("signature") if isinstance(tb, dict) else getattr(tb, "signature", None)
                    if isinstance(tb_signature, str) and tb_signature:
                        block["signature"] = tb_signature

            for tc_delta in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc_delta, "index", 0) or 0
                tc = tool_calls_by_idx.setdefault(idx, {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if getattr(tc_delta, "id", None):
                    tc["id"] = tc_delta.id
                func_delta = getattr(tc_delta, "function", None)
                if func_delta is not None:
                    if getattr(func_delta, "name", None):
                        tc["function"]["name"] += func_delta.name
                    if getattr(func_delta, "arguments", None):
                        tc["function"]["arguments"] += func_delta.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        return {
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts),
            "thinking_blocks": [b for b in thinking_blocks if b.get("thinking") or b.get("signature")],
            "tool_calls": [tool_calls_by_idx[k] for k in sorted(tool_calls_by_idx)],
            "finish_reason": finish_reason,
            "usage": usage,
        }, None

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
        # First time this session is seen since process start? Prepend a
        # synthetic boot event so the agent learns when the system came
        # up and what model is running. Rides alongside the real events
        # in the same turn.
        boot_event = self._maybe_boot_event(session_id)
        if boot_event is not None:
            events = [boot_event] + list(events)

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
                time=now,
                energy=event.energy,
                **event.metadata,
            )
            user_msg = {"role": "user", "content": text}
            new_items.append(context_msg)
            new_items.append(user_msg)

        # Two parallel records of this turn's content:
        #   - all_new_messages: persistence shape (function_call,
        #     function_call_output, role messages, reasoning, comments).
        #     Saved to the session file at end-of-turn. Leads with a
        #     turn_start comment so the next turn can locate this turn's
        #     boundary for the rolling cache markers.
        #   - in_turn_chat: chat-completions shape (assistant w/ tool_calls,
        #     tool messages). Held in memory only so we can attach
        #     thinking_blocks for in-turn reasoning continuity without
        #     polluting persisted history (Anthropic doesn't replay thinking
        #     across user turns anyway).
        all_new_messages: list[dict[str, Any]] = [
            {"type": "comment", "kind": "turn_start"}, *new_items]
        developer_role = self.developer_role
        preserve_reasoning = self.replay_reasoning

        in_turn_chat: list[dict[str, Any]] = []
        for item in new_items:
            role = item.get("role")
            content = item.get("content") or ""
            in_turn_chat.append({
                # Same developer-role rule as _to_chat_messages, so the
                # in-turn shape matches what the next turn rebuilds.
                "role": developer_role if role == "developer" else role,
                "content": content,
            })

        # Rolling cache markers. The cache lookup only scans a bounded
        # number of content-block boundaries behind each breakpoint, and a
        # turn appends more blocks than that — so a single tail marker
        # never finds the previous turn's entry and re-creates the whole
        # prefix every turn. Instead, keep the marker at the PREVIOUS
        # turn's boundary (byte-identical to where the last turn's tail
        # marker sat -> cache read hits) and add one at the new tail
        # (writes this turn's delta; becomes next turn's read point).
        # The boundary is the last turn_start comment in the raw history.
        # Sessions without one (pre-rolling-markers) fall back to a single
        # tail marker: one full re-create, then self-heals.
        boundary = next(
            (i for i in range(len(raw_history) - 1, -1, -1)
             if raw_history[i].get("type") == "comment"
             and raw_history[i].get("kind") == "turn_start"),
            None,
        )

        # Reasoning items survive the filter only when we're going to replay
        # them; comments are telemetry and never go on the wire.
        dropped = ("comment",) if preserve_reasoning else ("reasoning", "comment")

        def _filt(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [m for m in items if m.get("type") not in dropped]

        def _to_chat(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return _to_chat_messages(
                items,
                developer_role=developer_role,
                preserve_reasoning=preserve_reasoning,
            )

        if boundary is None:
            history = _filt(raw_history)
            history_chat_raw = _to_chat(history)
            history_chat = _cache_last_msg(history_chat_raw)
        else:
            # Rendering the two segments separately is safe: a turn's block
            # always starts with a developer event, so no assistant/tool_call
            # merge in _to_chat_messages can span the boundary.
            prev_items = _filt(raw_history[:boundary])
            recent_items = _filt(raw_history[boundary:])
            history = prev_items + recent_items
            prev_chat = _to_chat(prev_items)
            recent_chat = _to_chat(recent_items)
            history_chat_raw = prev_chat + recent_chat
            history_chat = _cache_last_msg(prev_chat) + _cache_last_msg(recent_chat)
        instructions_msg = _with_cache_breakpoint({
            "role": "system",
            "content": self._build_instructions(),
        })
        marked = [i for i, m in enumerate(history_chat)
                  if isinstance(m.get("content"), list)
                  and any(isinstance(b, dict) and "cache_control" in b
                          for b in m["content"])]
        logger.info("cache markers: boundary_raw_idx=%s history_marks=%s of %d chat msgs",
                    boundary, marked, len(history_chat))

        # Diagnostic: per-message hashes compared against the previous turn
        # (in-memory). If any position in the shared prefix differs, log
        # exactly where and what — that's byte drift, our bug. If every
        # shared position matches and the cache still misses, the problem
        # is on the gateway side.
        import hashlib

        def _h(obj: Any) -> str:
            return hashlib.sha256(
                json.dumps(obj, sort_keys=False, ensure_ascii=False).encode()
            ).hexdigest()[:12]

        per_msg = [_h(m) for m in history_chat_raw]
        instructions_hash = _h(instructions_msg)
        prev = self._prefix_hashes.get(session_id)
        if prev is not None:
            prev_instructions, prev_msgs = prev
            if prev_instructions != instructions_hash:
                logger.warning("prefix drift: instructions hash changed %s -> %s",
                               prev_instructions, instructions_hash)
            common = min(len(prev_msgs), len(per_msg))
            diffs = [i for i in range(common) if prev_msgs[i] != per_msg[i]]
            if diffs:
                logger.warning(
                    "prefix drift: %d/%d shared positions differ vs last turn, "
                    "first at %d: %.400r",
                    len(diffs), common, diffs[0], history_chat_raw[diffs[0]])
            else:
                logger.info("prefix stable vs last turn: %d shared msgs identical "
                            "(prev %d, now %d)", common, len(prev_msgs), len(per_msg))
        self._prefix_hashes[session_id] = (instructions_hash, per_msg)

        # Base config + tools.
        base_kwargs: dict[str, Any] = dict(self.call_config)
        base_kwargs["model"] = self.model
        base_kwargs.setdefault("max_tokens", 65536)
        # Chat completions doesn't include usage on stream by default; opt in.
        base_kwargs.setdefault("stream_options", {"include_usage": True})
        chat_tools = [_tool_def_for_chat(t) for t in self.openai_tools]

        logger.info("Calling LLM: %d history items, %d new items, %d tools, %d event(s)",
                    len(history), len(new_items), len(chat_tools), len(events))

        for iteration in range(self.max_tool_iterations):
            messages = [instructions_msg] + history_chat + in_turn_chat
            call_kwargs = dict(base_kwargs)
            call_kwargs["messages"] = messages
            if chat_tools:
                call_kwargs["tools"] = chat_tools

            logger.debug("iter %d: %d messages", iteration, len(messages))

            try:
                response, partial = await self._stream_response(call_kwargs, cancel)
            except Exception as e:
                logger.error("LLM call failed: %s: %r", type(e).__name__, e, exc_info=True)
                return None

            # --- Interrupted during streaming ---
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

            # Record token usage as a transcript comment — stripped from replay
            # but preserved in the session log for cost/observability tracking.
            usage = response.get("usage")
            if usage is not None:
                def _int_or_none(v: Any) -> int | None:
                    return v if isinstance(v, int) else None
                input_details = getattr(usage, "prompt_tokens_details", None)
                cached = (
                    _int_or_none(getattr(input_details, "cached_tokens", None))
                    if input_details else None
                )
                output_details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = (
                    _int_or_none(getattr(output_details, "reasoning_tokens", None)) or 0
                    if output_details else 0
                )
                cache_read = _int_or_none(getattr(usage, "cache_read_input_tokens", None))
                cache_creation = _int_or_none(getattr(usage, "cache_creation_input_tokens", None))
                comment = {
                    "type": "comment",
                    "kind": "usage",
                    "iteration": iteration,
                    "input_tokens": _int_or_none(getattr(usage, "prompt_tokens", None)),
                    "output_tokens": _int_or_none(getattr(usage, "completion_tokens", None)),
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": _int_or_none(getattr(usage, "total_tokens", None)),
                    "cached_tokens": cached,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": cache_creation,
                }
                all_new_messages.append(comment)
                logger.info(
                    "  usage: in=%s out=%s reasoning=%d total=%s cached=%s cache_read=%s cache_create=%s",
                    comment["input_tokens"], comment["output_tokens"], reasoning_tokens,
                    comment["total_tokens"], cached, cache_read, cache_creation,
                )

            assistant_text = response["content"]
            tool_calls = response["tool_calls"]
            reasoning_text = response.get("reasoning") or ""
            thinking_blocks = response.get("thinking_blocks") or []

            if reasoning_text:
                all_new_messages.append({"type": "reasoning", "content": reasoning_text})

            if tool_calls:
                # Normalize tool_call arguments once so the in-turn replay and
                # the persisted form are byte-identical — if they diverge, the
                # cache prefix in turn N (raw args) won't match turn N+1's
                # rebuilt prefix (normalized args).
                normalized_calls: list[dict[str, Any]] = []
                for tc in tool_calls:
                    try:
                        args_unicode = json.dumps(json.loads(tc["function"]["arguments"]), ensure_ascii=False)
                    except (json.JSONDecodeError, ValueError):
                        args_unicode = tc["function"].get("arguments") or "{}"
                    normalized_calls.append({
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": args_unicode,
                    })

                # Build the assistant message we'll send back on the next
                # iteration. Reasoning rides along in both normalized forms
                # so the model can resume its chain across tool calls —
                # `thinking_blocks` is what the Anthropic paths consume
                # (signature round-trip), `reasoning_content` is the
                # cross-provider field others require on tool-call assistant
                # messages (Kimi-style models reject or degrade without it).
                # Each attaches only when non-empty, so providers that emit
                # neither (GPT) see a plain assistant message. All of it is
                # post-marker and in-memory only — dropped at end-of-turn.
                assistant_chat: dict[str, Any] = {"role": "assistant"}
                if reasoning_text:
                    assistant_chat["reasoning_content"] = reasoning_text
                    # Gateways that strip the native field would otherwise
                    # break the chain between tool iterations too, not just
                    # across turns.
                    if preserve_reasoning:
                        in_turn_chat.append(
                            _reasoning_dev_message(reasoning_text, developer_role))
                if thinking_blocks:
                    assistant_chat["thinking_blocks"] = thinking_blocks
                assistant_chat["content"] = assistant_text or None
                assistant_chat["tool_calls"] = [
                    {
                        "id": nc["id"],
                        "type": "function",
                        "function": {"name": nc["name"], "arguments": nc["arguments"]},
                    }
                    for nc in normalized_calls
                ]
                in_turn_chat.append(assistant_chat)

                # Persist in our session shape (no thinking_blocks).
                if assistant_text:
                    all_new_messages.append({"role": "assistant", "content": assistant_text})
                for nc in normalized_calls:
                    all_new_messages.append({
                        "type": "function_call",
                        "call_id": nc["id"],
                        "name": nc["name"],
                        "arguments": nc["arguments"],
                    })

                # Execute tool calls, checking for interruption between each
                turn_images: list[dict[str, Any]] = []
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

                    logger.info("  Tool call: %s(%s)", tc["function"]["name"], tc["function"]["arguments"][:100])
                    result, images = await self._execute_tool_call(
                        tc["id"], tc["function"]["name"], tc["function"]["arguments"]
                    )
                    logger.debug("  Result: %s", result["output"][:200])
                    all_new_messages.append(_prepare_for_history(result))
                    in_turn_chat.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result.get("output") or "",
                    })
                    turn_images.extend(images)

                # Images can't ride inside a tool-role message (string content
                # only), so after all tool results are in, surface them as one
                # follow-up user message. Not persisted — the pointer JSON in
                # the tool output text is the durable reference; the bytes are
                # re-fetchable via URI on a later turn.
                image_msg = _image_user_message(turn_images)
                if image_msg is not None:
                    in_turn_chat.append(image_msg)

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
