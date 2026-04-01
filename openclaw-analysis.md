# OpenClaw: What It Gets Right and Where It Falls Short

Based on extensive hands-on experience deploying and operating OpenClaw as a persistent AI assistant on a home Kubernetes cluster with Signal integration, ElevenLabs TTS, Home Assistant control, and ACP/Zed IDE integration.

## What OpenClaw Gets Right

### Multi-channel chat routing
The killer feature. One agent session reachable from Signal, Discord, WebUI, etc. Messages from any channel hit the same conversation context. This is genuinely hard to build and OpenClaw does it well.

### AGENTS.md / SOUL.md personality system
The workspace file approach (AGENTS.md, SOUL.md, IDENTITY.md, HEARTBEAT.md) for defining agent behavior is elegant. It's just prompt engineering in markdown files that the agent reads — no proprietary personality format, no DSL. The agent can even edit its own files to evolve. This pattern is portable and doesn't depend on OpenClaw at all.

### Heartbeat system
Simple but effective: fire a prompt on a schedule, let the agent decide what to do. Combined with the personality files, this gives the agent autonomous behavior (checking in, continuing conversations, managing its own sleep schedule). The implementation has issues (see below) but the concept is sound.

### ClawHub skill ecosystem
Community-contributed skills with a package manager (`npx clawhub search/install`). Good for discovery even if the skill format itself has problems.

### Sub-agent routing
Ability to define multiple agents (main, deep-research, web-search) with different models and let the main agent delegate. Works well for Perplexity/Sonar integration.

## Where It Falls Short
TL;DR: Two major issues
1. Ignores established standards 
2. Security

### Skills are prompt injection, not tools
Skills are markdown files that teach the agent to shell out with curl/jq. There's no typed tool interface, no structured input/output, no server-side response processing. Compare this to MCP servers which give you proper tool schemas, context management, and clean responses. The agent has to parse raw JSON from curl output and figure out what's relevant. This is fragile, token-wasteful, and error-prone.

### No native MCP support
Despite MCP being the obvious standard for tool integration, OpenClaw has no native MCP server connection support. The best option is `mcporter`, a CLI wrapper that the agent calls through bash — still going through the shell for every tool invocation. This means you can't bring your own MCP servers without a shell layer in between.

### Security attack surface is massive
The agent has full bash access. Skills execute shell commands. The entire gateway (chat channels, heartbeat, agent runtime, web UI, ACP bridge) runs in one process. There's no capability-based security model — the agent can do anything the container user can do. Compare this to an architecture where the agent's only capabilities are the MCP servers you explicitly configure.

### ACP is half-baked
ACP (Agent Control Protocol) is advertised as IDE integration but it's just a messaging bridge:
- It can't proxy tools — Zed can chat with the agent but the agent can't edit files in your project
- For actual code editing, you need the `acpx` harness which is a separate, poorly documented system
- `--server-args` on `acp client` is broken for flags containing dashes
- `session/new` creates bare sessions with no model config; you have to use `session/load` with a specific session key
- Device pairing is required for ACP connections with no way to disable it globally
bottom line: it's hard to use the agent as a coding assistant 

### Auth model is all-or-nothing
Can't separate WebUI auth from API auth. The WebSocket control UI auth happens at the application protocol level (hello message), not HTTP headers, so standard reverse proxy auth (oauth2-proxy) can't protect the WebUI without also blocking ACP. You either trust the gateway's own token auth for everything or add complexity that doesn't fully work.

### Monolithic architecture
Everything runs in one Node.js process: chat channel providers, heartbeat scheduler, agent runtime, web UI, ACP bridge, browser control, canvas hosting. Any issue in one subsystem affects everything. Config changes require full restarts. There's no way to scale individual components independently.

### Env var security theater
The env override system blocks variables matching patterns like `*_TOKEN` or `*_API_KEY` for "security," but then provides an allowlist mechanism through skill frontmatter. The blocking logic is in the wrong place — it should be at the deployment/secret level, not in application code that can be worked around. We spent significant time debugging why `HA_TOKEN` was being blocked despite the skill declaring it as required.

### Massive, rapidly-changing codebase
The codebase is enormous and entirely new/custom code. Every feature feels 80% complete. The documentation covers setup but not the actual architecture or edge cases. When things break, debugging requires reading minified JavaScript in the container. 

## What the Alternative Looks Like

### Core principle: compose existing tools, don't build a platform

The ideal architecture is thin glue between battle-tested components (hypothetical arch):

1. **LiteLLM proxy** — OpenAI-compatible endpoint that handles model routing, can inject system prompts at the proxy level, and acts as an MCP gateway with per-key tool access control
2. **MCP servers** — each capability (email, calendar, home automation, TTS) is a separate MCP server process with its own permissions. The operator chooses what to expose. No shell access needed.
3. **Thin chat adapters** — Signal bot, Discord bot, WebUI. Each is ~500 lines. Receives text, sends to proxy, returns response. No agent logic.
4. **Session store** — append-only conversation log in a file or database. Simple state management.
5. **Declarative config** — a single config file (or Helm values.yaml) that wires together: which proxy, which MCP servers, which chat channels, which system prompt.

### Why this is better

- **Security**: the agent can only do what its MCP servers expose. No shell = no RCE. Each MCP server is its own trust boundary.
- **Reliability**: each component is independent. A broken MCP server doesn't take down chat. A chat adapter restart doesn't reset sessions.
- **Portability**: the personality (system prompt / AGENTS.md) is just text. MCP servers are a standard protocol. Nothing is locked to a specific platform.
- **Simplicity**: the total amount of custom code is maybe 1000 lines of glue. Everything else is off-the-shelf.

### What exists today

- LiteLLM already supports MCP gateway mode and system prompt injection via middleware
- FastMCP makes writing MCP servers trivial (the IMAP/iCal/SMTP servers in our `aibs` project are ~200 lines each)
- signal-cli provides Signal integration via JSON-RPC
- The AGENTS.md prompt engineering pattern from OpenClaw is fully portable — it's just markdown that goes in the system prompt
- OpenCode is another potential backend that already supports remote execution; the question is whether its client supports tool call passthrough for local filesystem access

### What's missing

- A clean "assembly" layer that wires these pieces together declaratively
- Session management with proper compaction/continuity (the hard problem OpenClaw also hasn't solved)
- Making it easy for non-technical users (the product problem, not the technical one)
