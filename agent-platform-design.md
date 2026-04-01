# Agent Platform Design Doc

A minimal, composable agent runtime built from proven existing tools. No monolithic framework — just thin glue between battle-tested components.

## Problem

Existing agent platforms (e.g. OpenClaw) try to be everything: chat routing, tool execution, session management, web UI, plugin systems. The result is a massive attack surface, half-baked features, and no native support for standards like MCP.

## Core Principles

1. **Compose, don't build** — use existing tools for each capability
2. **MCP is the only tool interface** — no shell access, no curl skills, no bash
3. **Security by default** — the agent can only do what its MCP servers expose
4. **Each component is replaceable** — swap out any piece without rewriting the others

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Signal Bot  │     │  Discord Bot │     │    Web UI     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────┬───────┘────────────────────┘
                    │
            ┌───────▼────────┐
            │   LiteLLM      │
            │   Proxy         │
            │                │
            │ - Model routing │
            │ - System prompt │
            │ - MCP gateway  │
            │ - Per-key ACL  │
            └───────┬────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
 ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
 │ IMAP MCP  │ │ iCal  │ │  HA MCP   │
 │ Server    │ │ MCP   │ │  Server   │
 └───────────┘ └───────┘ └───────────┘
```

### Components

**1. LiteLLM Proxy (the brain)**
- OpenAI-compatible `/v1/chat/completions` endpoint
- Routes to any model provider (Anthropic, OpenAI, Google, etc.)
- Injects system prompt at proxy level (personality, behavior rules, memory instructions)
- Acts as MCP gateway — exposes MCP server tools to the model with per-API-key access control
- Handles tool call execution loop (model requests tool → proxy calls MCP server → returns result → model continues)
- Already exists: https://docs.litellm.ai

**2. MCP Servers (capabilities)**
- Each capability is a separate process with its own permissions
- Examples: IMAP (read email), SMTP (send email), iCal (calendar), Home Assistant, ElevenLabs TTS
- Standard protocol — use any existing MCP server or write new ones with FastMCP (~200 lines each)
- The operator decides which MCP servers to connect — this IS the security model
- No shell access = no RCE. The agent literally cannot do anything its MCP servers don't expose.

**3. Chat Adapters (thin clients)**
- Each is ~500 lines: receive message → POST to proxy → return response
- Signal: signal-cli JSON-RPC → adapter → proxy
- Discord: discord.py bot → adapter → proxy
- Web UI: simple websocket chat page → adapter → proxy
- No agent logic in the adapter. It's a pipe.

**4. Session Store**
- Append-only conversation log (JSONL or SQLite)
- Each chat channel maps to a session
- Adapter includes session history in the API call to the proxy
- Compaction: when history exceeds token budget, summarize older messages (can use a cheaper model)

**5. System Prompt**
- Markdown files, same pattern as OpenClaw's AGENTS.md/SOUL.md
- Injected by the proxy into every request
- Agent can update its own memory files (via a filesystem MCP server scoped to the memory directory)
- The personality system is 100% portable — it's just text

### Configuration

Single declarative config (YAML or TOML):

```yaml
proxy:
  provider: litellm
  model: anthropic/claude-opus-4-6
  fallback_models:
    - google/gemini-3-pro
    - openai/gpt-5.4

system_prompt:
  files:
    - ./personality/SOUL.md
    - ./personality/USER.md
  memory_dir: ./memory/

mcp_servers:
  imap:
    url: http://localhost:13087
    description: "Read emails via IMAP"
  calendar:
    url: http://localhost:13086
    description: "Check calendar events"
  home_assistant:
    url: http://ha.home.arpa/mcp_server
    headers:
      Authorization: "Bearer ${HA_TOKEN}"
  tts:
    url: http://localhost:13089
    description: "Text-to-speech via ElevenLabs"
  memory:
    command: ["python3", "./mcp-servers/memory-fs.py"]
    args: ["--root", "./memory/"]
    description: "Read and write agent memory files"

channels:
  signal:
    adapter: ./adapters/signal.py
    account: "+12025240651"
    allow_from:
      - "+16092409191"
  # discord:
  #   adapter: ./adapters/discord.py
  #   token: ${DISCORD_TOKEN}

session:
  store: ./sessions/
  max_history_tokens: 100000
  compaction_model: anthropic/claude-haiku-4-5

heartbeat:
  interval: 20m
  prompt: "Check memory/HEARTBEAT.md and decide if anything needs attention."
```

### Heartbeat

Not a framework feature — just a cron job that sends a message:

```bash
# crontab
*/20 * * * * curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"main","messages":[{"role":"user","content":"[HEARTBEAT] Check HEARTBEAT.md"}]}'
```

The agent's system prompt tells it how to handle heartbeats (same as OpenClaw's AGENTS.md pattern). Quiet hours? Add a time check to the cron script. Different schedules per day? Cron handles that natively.

### Security Model

| Component | Trust Boundary | Attack Surface |
|-----------|---------------|----------------|
| LiteLLM proxy | API keys, per-key tool ACLs | HTTP endpoint (standard, well-audited) |
| MCP servers | Each server exposes only its tools | Per-server process isolation |
| Chat adapters | Message in, message out | Channel-specific auth (Signal verification, Discord tokens) |
| Session store | Append-only files | Filesystem permissions |
| System prompt | Read-only to the model | Can't be modified by the model unless memory MCP allows it |

Compare to OpenClaw: the agent has bash, can run arbitrary commands, all tools go through shell, everything runs in one process, and the skill system is prompt injection.

### What Exists vs What Needs Building

**Already exists:**
- LiteLLM with MCP gateway support
- FastMCP for writing MCP servers (IMAP, SMTP, iCal servers already built in ~/projects/nvidia/aibs)
- signal-cli for Signal integration
- AGENTS.md/SOUL.md personality pattern (portable from OpenClaw)

**Needs building (~1000 lines total):**
- Signal chat adapter (~200 lines) — signal-cli JSON-RPC → session store → proxy
- Session management (~300 lines) — history loading, token counting, compaction trigger
- Config loader (~200 lines) — parse YAML, wire components together
- Web UI adapter (~300 lines) — simple websocket chat page (optional, low priority)

**Nice to have (later):**
- Bidirectional context sync with Claude Code (hooks already prototyped)
- Multi-agent routing (different system prompts per channel/user)
- Voice input/output pipeline (Whisper → agent → TTS)

### Deployment

Kubernetes (Helm chart) or docker-compose. Each component is a container:

```yaml
# docker-compose.yml sketch
services:
  proxy:
    image: ghcr.io/berriai/litellm:latest
    volumes:
      - ./config:/config
      - ./personality:/personality
      - ./memory:/memory

  signal-adapter:
    build: ./adapters/signal
    depends_on: [proxy, signal-cli]

  signal-cli:
    image: signal-cli-rest-api
    volumes:
      - signal-data:/signal

  imap-mcp:
    build: ./mcp-servers/imap
    environment:
      - IMAP_SERVER=tls://127.0.0.1:1143
      - IMAP_USER=${PROTONMAIL_USER}

  calendar-mcp:
    build: ./mcp-servers/ical
    environment:
      - CALENDAR_URL=${PROTONMAIL_ICAL_URL}

  ha-mcp:
    # Use HA's built-in MCP server
    # Just configure the URL in the proxy
```

### Migration from OpenClaw

1. Copy SOUL.md, USER.md, NANHI.md, memory files → system prompt directory
2. Configure LiteLLM with same model provider (NVIDIA inference API)
3. Point signal-cli at the new adapter instead of OpenClaw
4. MCP servers replace OpenClaw skills (already built for IMAP/iCal/SMTP)
5. Session history starts fresh (or import via transcript conversion)

The personality, memory, and behavior are 100% portable. The infrastructure is swappable.
