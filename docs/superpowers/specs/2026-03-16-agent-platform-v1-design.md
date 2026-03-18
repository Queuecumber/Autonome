# Agent Platform v1 — Design Spec

A minimal, composable agent runtime that replaces OpenClaw by gluing together existing tools. No monolithic framework — just thin Python code between battle-tested components.

## Problem

OpenClaw tries to be everything: chat routing, tool execution, session management, web UI, plugin system. The result is a massive attack surface (bash access as security model), half-baked features (no MCP support, broken ACP), and a monolithic Node.js process where any issue affects everything. See `openclaw-analysis.md` for the full critique.

## v1 Scope

Replace OpenClaw for a single user talking to a single agent over Signal. Same UX — chat, heartbeat, MCP tools, personality files, memory — with better security and minimal new code.

### In Scope

- Session manager (central orchestrator — receives events, drives LLM calls)
- Signal adapter (inbound listener + outbound MCP tools)
- LiteLLM proxy (model routing + MCP tool execution)
- Filesystem MCP server (workspace access — personality, memory, heartbeat files)
- Auto-discovery of workspace markdown files for system prompt injection
- Config loader (single `agent.yaml` drives everything)
- Heartbeat via cron
- User-provided MCP servers declared in config

### Out of Scope (Future)

- Other adapters (web UI, Discord, coding agent bridge, OpenAI-compatible responses API)
- Multi-user / multi-agent
- Context architecture (cross-session awareness, hot/warm context tiers)
- Real-time voice/video calls
- Session compaction/summarization (truncate oldest messages as safety net)

## Architecture

The session manager is the central orchestrator. Adapters are split into two halves: an **inbound listener** that pushes events to the session manager, and an **outbound MCP server** that the agent calls to act on the channel. The agent decides where and how to respond — the session manager does not route responses automatically.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Signal        │  │ Coding Tool  │  │ Heartbeat    │
│ Inbound       │  │ Inbound      │  │ Cron         │
│ (polls        │  │ (HTTP server │  │              │
│  signal-cli)  │  │  /v1/chat/*) │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │ push event      │ push event      │ push event
       ▼                 ▼                 ▼
┌──────────────────────────────────────────────────┐
│              Session Manager                      │
│  - receives events from all adapters              │
│  - maintains per-(channel, session) JSONL history │
│  - builds LLM requests with MCP tool declarations │
│  - calls LiteLLM                                  │
│  - agent replies via MCP tools (not auto-routed)  │
└──────────────────┬───────────────────────────────┘
                   │ POST /v1/chat/completions
             ┌─────▼─────┐
             │  LiteLLM   │
             └─────┬─────┘
                   │ MCP tool calls
     ┌─────────────┼──────────────┬────────────┐
     │             │              │            │
  ┌──▼──┐   ┌─────▼──────┐  ┌───▼────────┐ ┌─▼──────────┐
  │ FS  │   │ Signal     │  │ Coding     │ │ User MCP   │
  │ MCP │   │ Outbound   │  │ Outbound   │ │ (HA, etc)  │
  │     │   │ MCP        │  │ MCP        │ │            │
  └─────┘   └────────────┘  └────────────┘ └────────────┘
 implicit    send_message    reply           from config
             react           (request_id)
             stage_attachment
```

### Key Design Principle: Agent Controls Output

The agent decides where and how to respond. When the session manager receives an event (e.g., "Signal message from +1234: hey"), it wakes the agent with that context. The agent then calls MCP tools to act:

- `signal.send_message("+1234", "on it")` — reply on Signal
- `signal.react(msg_id, "👍")` — react instead of replying
- Could also act on other channels if available

The session manager never automatically routes responses back to the originating channel. The response is whatever tool calls the agent makes.

### Channel Types

| Type | Inbound | Outbound | Push? | Example |
|------|---------|----------|-------|---------|
| **Push** | Adapter pushes events | `send_message` works anytime | Yes | Signal, Discord, WebSocket UI |
| **Request/Response** | HTTP request arrives | `reply(request_id)` only during active request | No | OpenAI API, coding tools |

For request/response channels, the `reply` tool only works while the originating request is pending. If the agent calls it outside a request context, the tool returns an error. The agent can then reach the user via a push channel instead.

### Data Flow

1. Signal adapter polls signal-cli, receives a message
2. Adapter pushes an event to the session manager: `{source: "signal", sender: "+1234", text: "hey", message_id: "ts_123"}`
3. Session manager loads session history for `(signal, +1234)` from JSONL
4. Session manager builds chat completion request: system prompt + history + event context + MCP tool declarations
5. Session manager POSTs to LiteLLM `/v1/chat/completions`
6. LiteLLM injects system prompt (workspace markdown files)
7. LiteLLM discovers tools from declared MCP servers
8. LiteLLM forwards to model provider
9. Model may request tool calls → LiteLLM executes via MCP → feeds results back → model continues
10. During tool loop, agent calls `signal.send_message("+1234", "yo!")` — this executes in the Signal adapter's outbound MCP server, which calls signal-cli
11. Agent may also call `signal.react("ts_123", "👍")`, `workspace_fs.write_file(...)`, etc.
12. Final response returns to session manager (may be empty if agent handled everything via tools)
13. Session manager appends the full exchange to JSONL session file

### Sessions and Concurrency

Each `(channel, session_id)` pair gets its own conversation history. For Signal, `session_id` is the contact phone number. For a coding tool, it would be a project or connection ID.

Sessions are independent — they share MCP tools but not conversation history. This matches how coding sessions work (each has its own project context) and avoids the context window explosion problem.

**v1 concurrency model:** The session manager serializes events per session. Two messages to the same session are processed sequentially. Events to different sessions can be processed concurrently (different contacts, different channels).

## Components

### 1. Session Manager (~200 lines, new)

The central orchestrator. A Python service that:

- **Receives events** via internal HTTP API (`POST /event`)
- **Maintains session histories** in JSONL files, keyed by `(channel, session_id)`
- **Builds LLM requests** with conversation history, event context, system prompt references, and MCP tool declarations
- **Calls LiteLLM** and handles the response
- **Saves exchanges** to session history

The session manager does NOT:
- Know anything about Signal, Discord, or any specific channel
- Route responses to adapters (the agent does this via MCP tools)
- Own the MCP servers (adapters and config do)

**Event format (inbound):**

```json
{
  "source": "signal",
  "session_id": "+16092409191",
  "text": "hey what's up",
  "metadata": {
    "message_id": "ts_1234567890",
    "sender": "+16092409191"
  }
}
```

**Internal HTTP API:**

- `POST /event` — Push an event from any adapter. Session manager processes it asynchronously.
- `POST /heartbeat` — Trigger a heartbeat (sends configured prompt as an event).

### 2. LiteLLM Proxy (off-the-shelf)

The LLM execution engine. Provides:

- **OpenAI-compatible API** at `/v1/chat/completions`
- **Model routing** to any provider (Anthropic, OpenAI, Google, NVIDIA, etc.)
- **System prompt injection** via callback/middleware — reads workspace markdown files and prepends to every request
- **MCP gateway** — when the client includes `"type": "mcp"` tool declarations, LiteLLM fetches tool schemas from the MCP servers, injects them into the request, executes tool calls server-side, and runs the agentic loop until the model produces a final response
- **Per-key ACLs** for tool access control (future multi-user)

Config is generated by our config loader from `agent.yaml`.

### 3. Filesystem MCP Server

A workspace filesystem MCP server (`src/mcp_servers/workspace_fs/server.py`) that exposes `read_file`, `write_file`, `list_directory`, `search_files`. Runs as an HTTP service. Path traversal outside the workspace root is rejected.

Scoped to the workspace root. This gives the agent access to:

- **Personality files** (`SOUL.md`, `USER.md`, `AGENTS.md`, etc.) — read and write. The agent can edit its own personality. Changes take effect next request when the callback re-reads files.
- **Memory files** (`memory/`, `MEMORY.md`) — read and write.
- **Heartbeat config** (`HEARTBEAT.md`) — read.
- Any other files the operator places in the workspace.

Implicit MCP server — always present, configured automatically from `workspace:` in `agent.yaml`.

### 4. Signal Adapter

Split into two parts running in the same process:

**Inbound listener:**
- Polls signal-cli JSON-RPC for new messages
- Filters by allowed contacts
- Normalizes messages into the event format
- Pushes events to the session manager via `POST /event`

**Outbound MCP server (FastMCP HTTP):**
- `send_message(recipient: str, text: str)` — Send a message via signal-cli
- `send_attachment(recipient: str, file_path: str, mime_type: str, caption: str | None)` — Send a file
- `react(message_id: str, emoji: str)` — React to a message
- `set_typing(recipient: str, enabled: bool)` — Typing indicator

Note: tools no longer take `conversation_id` — they take `recipient` (phone number) or `message_id` directly. The agent knows who to message from the event context.

Future tools (not v1): `pin`, `quote_reply`, `set_disappearing`, `initiate_call`.

**Attachment lifecycle:** The adapter deletes staged files after successful delivery to signal-cli.

### 5. Config Loader (~100 lines)

Reads `agent.yaml` and wires everything together.

**Generates:**
- LiteLLM `config.yaml` with model routing and MCP server definitions (user-defined + implicit workspace FS + adapter outbound MCPs)
- MCP tool declaration list for the session manager
- Adapter configs

**Environment variable substitution:** `${VAR_NAME}` in config values are resolved from environment variables.

**Workspace markdown auto-discovery:** All `*.md` files in the workspace root are collected for system prompt injection.

### 6. System Prompt Injection

LiteLLM callback that:

1. Reads all `*.md` files from the workspace root (auto-discovered)
2. Concatenates them in alphabetical order
3. Prepends the result as the system message on every chat completion request

Re-reads from disk on every request (files are small, ensures edits take effect immediately).

### 7. Heartbeat

A cron job that pushes a heartbeat event to the session manager:

```bash
*/20 * * * * curl -s -X POST http://localhost:5000/heartbeat
```

The session manager processes it like any other event: loads session history, sends the heartbeat prompt to LiteLLM, and the agent decides what to do (check calendar, message the user, do nothing). The heartbeat prompt comes from `agent.yaml`.

## Configuration

```yaml
# agent.yaml

model:
  provider: anthropic
  model: claude-opus-4-6
  api_key: ${ANTHROPIC_API_KEY}

workspace: ./workspace

# Optional: explicit ordering for system prompt files.
# system_prompt:
#   - ./workspace/SOUL.md
#   - ./workspace/USER.md

mcp_servers:
  # User-provided MCP servers.
  home_assistant:
    url: http://ha.home.arpa/mcp
    headers:
      Authorization: "Bearer ${HA_TOKEN}"

channels:
  signal:
    account: "+12025240651"
    signal_cli: http://localhost:8080
    allow_from:
      - "+16092409191"

session:
  store: ./sessions/
  max_history_tokens: 100000

heartbeat:
  interval: 20m
  prompt: "Check HEARTBEAT.md and decide if anything needs attention."
```

## Generated LiteLLM Config

```yaml
# generated/litellm-config.yaml (DO NOT EDIT)

model_list:
  - model_name: main
    litellm_params:
      model: anthropic/claude-opus-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

mcp_servers:
  workspace_fs:
    url: "http://workspace-fs-mcp:8000/mcp"
    transport: "http"
  signal:
    url: "http://signal-adapter:8100/mcp"
    transport: "http"
  home_assistant:
    url: "http://ha.home.arpa/mcp"
    transport: "http"
    headers:
      Authorization: "Bearer os.environ/HA_TOKEN"
```

## Design Decisions

**Agent controls output:** The agent decides where and how to respond via MCP tool calls. The session manager does not auto-route responses. This enables cross-channel responses and gives the agent true agency over its interactions.

**Session per (channel, session_id):** Each channel connection gets its own conversation history. Histories are independent — shared tools, separate context. This avoids context window explosion from mixing unrelated conversations and matches how tools like Claude Code already work.

**Push vs request/response channels:** Push channels (Signal) expose `send_message` that works anytime. Request/response channels (future: OpenAI API) expose `reply(request_id)` that only works during an active request — otherwise returns an error. The agent learns the constraints of each channel naturally.

**Session manager as standalone service:** The session manager is a real service, not embedded in an adapter. This is the natural architecture for multi-channel support and avoids duplicating orchestration logic across adapters.

**`require_approval: "never"` on all MCP servers:** The agent can call any tool without human confirmation. Security is at the MCP server level (each server only exposes safe operations), not tool-call approval.

**Token counting:** Uses LiteLLM's `token_counter()` utility. When truncating, entire exchanges are dropped as a unit to preserve conversation structure.

**System prompt file reads:** Re-read from disk every request. Files are small, ensures edits take effect immediately.

## Security Model

| Component | Trust Boundary | Attack Surface |
|-----------|---------------|----------------|
| Session manager | Internal HTTP API (not exposed externally) | Receives events from adapters |
| LiteLLM proxy | API keys, per-key tool ACLs | HTTP endpoint (well-audited) |
| Filesystem MCP | Scoped to workspace directory | Agent can read/write workspace only |
| Signal outbound MCP | Explicit recipient on every call | Agent must specify who to message |
| User MCP servers | Each server exposes only its tools | Per-server process isolation |
| Signal inbound | Allowed contacts list | signal-cli auth + contact filter |
| Session store | Filesystem permissions | Append-only JSONL files |
| System prompt | Auto-discovered from workspace | Agent can edit (by design) |

**Key improvement over OpenClaw:** No bash access. The agent can only interact with the world through MCP servers the operator explicitly configures. No shell = no RCE.

## Deployment

### docker-compose (primary for v1)

```yaml
services:
  session-manager:
    build: ./src/session_manager
    volumes:
      - ./sessions:/sessions
      - ./agent.yaml:/app/agent.yaml:ro
    ports:
      - "5000:5000"
    environment:
      - LITELLM_URL=http://litellm:4000
      - AGENT_CONFIG=/app/agent.yaml
    depends_on: [litellm]
    restart: unless-stopped

  litellm:
    image: ghcr.io/berriai/litellm:latest
    volumes:
      - ./generated/litellm-config.yaml:/config/config.yaml
      - ./workspace:/workspace:ro
      - ./src/agent_platform/callbacks:/callbacks:ro
    ports:
      - "4000:4000"
    command: ["--config", "/config/config.yaml"]
    environment:
      - PYTHONPATH=/callbacks
      - WORKSPACE_DIR=/workspace
    restart: unless-stopped

  signal-adapter:
    build: ./src/adapters/signal
    ports:
      - "8100:8100"
    environment:
      - SESSION_MANAGER_URL=http://session-manager:5000
      - SIGNAL_CLI_URL=http://signal-cli:8080
      - SIGNAL_ACCOUNT=+12025240651
      - ALLOW_FROM=+16092409191
      - CHANNEL_MCP_PORT=8100
    depends_on: [session-manager, signal-cli]
    restart: unless-stopped

  signal-cli:
    image: bbernhard/signal-cli-rest-api:latest
    volumes:
      - signal-data:/home/.local/share/signal-cli
    ports:
      - "8080:8080"
    environment:
      - MODE=json-rpc
    restart: unless-stopped

  workspace-fs-mcp:
    build: ./src/mcp_servers/workspace_fs
    volumes:
      - ./workspace:/workspace
    environment:
      - WORKSPACE_DIR=/workspace
    restart: unless-stopped

volumes:
  signal-data:
```

## File Structure

```
agent-platform/
├── agent.yaml                        # User config
├── src/
│   ├── agent_platform/
│   │   ├── config.py                 # Config loader
│   │   ├── session.py                # Session manager (JSONL, truncation)
│   │   └── callbacks/
│   │       └── system_prompt.py      # LiteLLM callback
│   ├── session_manager/
│   │   ├── server.py                 # HTTP server: /event, /heartbeat
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── adapters/
│   │   └── signal/
│   │       ├── inbound.py            # Polls signal-cli, pushes events
│   │       ├── outbound_mcp.py       # FastMCP: send_message, react, etc.
│   │       ├── main.py               # Runs both in same process
│   │       ├── Dockerfile
│   │       └── requirements.txt
│   └── mcp_servers/
│       └── workspace_fs/
│           ├── server.py
│           └── Dockerfile
├── workspace/
├── sessions/
├── generated/
├── docker-compose.yaml
└── tests/
```

## Open Questions / Future Work

1. **Session compaction:** v1 truncates oldest messages when over token budget. Future versions should summarize using a cheaper model.

2. **Context architecture:** The main agent session should eventually be aware of other sessions. This requires research into hot/warm context tiers, structured digests, and on-demand context promotion.

3. **Real-time voice/video:** Architecturally separate. Would be a streaming adapter connecting to real-time model APIs. The outbound MCP could expose `initiate_call(contact)` to bridge chat and call modes.

4. **Request/response adapters:** OpenAI-compatible API adapter for coding tools. Inbound: HTTP request arrives. Outbound: `reply(request_id)` tool that only works during active requests, errors otherwise.

5. **Config validation:** Validate `agent.yaml` before generating configs — fail fast with clear errors.

6. **Observability:** LiteLLM has Langfuse/Prometheus support. v1 uses container stdout. Future versions may enable structured logging.

7. **Cross-session awareness:** Agent in the main Signal session knowing about coding sessions. The deferred research problem — independent sessions are the v1 answer.
