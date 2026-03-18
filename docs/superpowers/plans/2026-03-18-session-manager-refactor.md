# Session Manager Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the agent platform so the session manager is the central orchestrator and adapters are split into inbound (push events) and outbound (MCP tools), giving the agent control over where and how it responds.

**Architecture:** Extract LLM orchestration from the Signal adapter into a standalone session manager service with `POST /event` and `POST /heartbeat`. The Signal adapter splits into an inbound listener (polls signal-cli, pushes events) and an outbound MCP server (send_message, react, etc.). The agent decides where to respond via MCP tool calls — the session manager never auto-routes responses.

**Tech Stack:** Python 3.12+, FastMCP, LiteLLM, httpx, Starlette, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-agent-platform-v1-design.md`

**NixOS dev:** Run all commands inside `nix-shell` (auto-activates venv + sets LD_LIBRARY_PATH)

---

## File Structure (after refactor)

```
src/
├── agent_platform/
│   ├── __init__.py                   # (no change)
│   ├── config.py                     # (minor: update generate_litellm_config MCP labels)
│   ├── session.py                    # (update: key by (channel, session_id) tuples)
│   └── callbacks/
│       ├── __init__.py               # (no change)
│       └── system_prompt.py          # (no change)
├── session_manager/
│   ├── __init__.py                   # NEW
│   ├── server.py                     # NEW: HTTP server — POST /event, POST /heartbeat
│   ├── Dockerfile                    # NEW
│   └── requirements.txt              # NEW
├── adapters/
│   ├── __init__.py                   # (no change)
│   └── signal/
│       ├── __init__.py               # (no change)
│       ├── inbound.py                # NEW: polls signal-cli, pushes events to session manager
│       ├── outbound_mcp.py           # RENAME+REFACTOR from channel_mcp.py: send_message, react, etc.
│       ├── main.py                   # REFACTOR: runs inbound + outbound concurrently
│       ├── Dockerfile                # (update)
│       └── requirements.txt          # (update)
└── mcp_servers/
    └── workspace_fs/                 # (no change)

tests/
├── conftest.py                       # (add session_manager fixture)
├── test_config.py                    # (minor update for new MCP labels)
├── test_session.py                   # (update: channel-aware session keys)
├── test_session_manager.py           # NEW: event handling, LLM orchestration
├── test_signal_inbound.py            # NEW: polling, event pushing
├── test_signal_outbound.py           # RENAME+REFACTOR from test_channel_mcp.py
├── test_system_prompt.py             # (no change)
├── test_workspace_fs.py              # (no change)
└── test_integration.py              # (rewrite for new architecture)
```

**Files to delete:**
- `src/adapters/signal/adapter.py` — replaced by session_manager/server.py + inbound.py
- `src/adapters/signal/channel_mcp.py` — replaced by outbound_mcp.py
- `tests/test_adapter.py` — replaced by test_session_manager.py + test_signal_inbound.py
- `tests/test_channel_mcp.py` — replaced by test_signal_outbound.py

---

## Task 1: Update session.py for channel-aware keys

**Files:**
- Modify: `src/agent_platform/session.py`
- Modify: `tests/test_session.py`

The session manager needs to store history per `(channel, session_id)` instead of just contact string. The change is small: the file path becomes `{channel}_{session_id}.jsonl`.

- [ ] **Step 1: Update test for channel-aware session**

```python
# tests/test_session.py — REPLACE entire file
import json
from pathlib import Path

from agent_platform.session import SessionManager


def test_load_empty_session(tmp_sessions):
    """Loading a session for a new channel+contact returns empty history."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    history = mgr.load("signal", "+11111111111")
    assert history == []


def test_session_file_naming(tmp_sessions):
    """Session files are named by channel and session_id."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    path = mgr._session_path("signal", "+11111111111")
    assert path.name == "signal_+11111111111.jsonl"


def test_separate_channel_sessions(tmp_sessions):
    """Different channels have separate session files."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    mgr.append("signal", "+11111111111", [{"role": "user", "content": "signal msg"}])
    mgr.append("coding", "project-x", [{"role": "user", "content": "coding msg"}])

    signal_history = mgr.load("signal", "+11111111111")
    coding_history = mgr.load("coding", "project-x")

    assert len(signal_history) == 1
    assert signal_history[0]["content"] == "signal msg"
    assert len(coding_history) == 1
    assert coding_history[0]["content"] == "coding msg"


def test_append_and_reload(tmp_sessions):
    """Messages appended to session can be reloaded."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    mgr.append("signal", "+11111111111", [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    history = mgr.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"


def test_load_truncated_drops_oldest(tmp_sessions):
    """When over token budget, oldest exchanges are dropped."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    messages = []
    for i in range(10):
        messages.extend([
            {"role": "user", "content": f"Message {i} " + "x" * 100},
            {"role": "assistant", "content": f"Response {i} " + "y" * 100},
        ])
    mgr.append("signal", "+11111111111", messages)

    truncated = mgr.load_truncated("signal", "+11111111111")
    assert len(truncated) < len(messages)
    assert truncated[-1]["content"].startswith("Response 9")


def test_load_truncated_preserves_exchange_integrity(tmp_sessions):
    """Truncation drops complete exchanges, not partial ones."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    mgr.append("signal", "+11111111111", [
        {"role": "user", "content": "old " + "x" * 200},
        {"role": "assistant", "content": "old reply " + "y" * 200},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new reply"},
    ])

    truncated = mgr.load_truncated("signal", "+11111111111")
    if truncated:
        assert truncated[0]["role"] == "user"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix-shell --run "pytest tests/test_session.py -v"`
Expected: FAIL — `load()` takes wrong number of args

- [ ] **Step 3: Update session.py**

```python
# src/agent_platform/session.py — REPLACE entire file
"""Session manager: JSONL history per (channel, session_id) with token-based truncation."""

import json
from pathlib import Path
from typing import Any

import litellm


class SessionManager:
    def __init__(self, store_dir: Path, max_history_tokens: int = 100000):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_tokens = max_history_tokens

    def _session_path(self, channel: str, session_id: str) -> Path:
        """Session file path keyed by (channel, session_id)."""
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return self.store_dir / f"{channel}_{safe_id}.jsonl"

    def load(self, channel: str, session_id: str) -> list[dict[str, Any]]:
        """Load conversation history for a (channel, session_id)."""
        path = self._session_path(channel, session_id)
        if not path.exists():
            return []
        messages = []
        for line in path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        return messages

    def append(self, channel: str, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append messages to a session file."""
        path = self._session_path(channel, session_id)
        with path.open("a") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    def load_truncated(
        self, channel: str, session_id: str, model: str = "claude-opus-4-6"
    ) -> list[dict[str, Any]]:
        """Load history, truncating oldest complete exchanges if over token budget."""
        messages = self.load(channel, session_id)
        if not messages:
            return messages

        token_count = self._count_tokens(messages, model)
        if token_count <= self.max_history_tokens:
            return messages

        exchanges = self._group_exchanges(messages)

        while exchanges and self._count_tokens(
            [m for ex in exchanges for m in ex], model
        ) > self.max_history_tokens:
            exchanges.pop(0)

        return [m for ex in exchanges for m in ex]

    def _group_exchanges(self, messages: list[dict]) -> list[list[dict]]:
        """Group messages into exchanges, each starting with a user message."""
        exchanges: list[list[dict]] = []
        current: list[dict] = []
        for msg in messages:
            if msg.get("role") == "user" and current:
                exchanges.append(current)
                current = []
            current.append(msg)
        if current:
            exchanges.append(current)
        return exchanges

    def _count_tokens(self, messages: list[dict], model: str) -> int:
        """Count tokens using LiteLLM's token counter with fallback."""
        try:
            return litellm.token_counter(model=model, messages=messages)
        except Exception:
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            return total_chars // 4
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `nix-shell --run "pytest tests/test_session.py -v"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_platform/session.py tests/test_session.py
git commit -m "refactor: session manager keyed by (channel, session_id) tuples"
```

---

## Task 2: Create session manager service

**Files:**
- Create: `src/session_manager/__init__.py`
- Create: `src/session_manager/server.py`
- Create: `tests/test_session_manager.py`

The session manager is the central orchestrator: receives events, loads history, calls LiteLLM, saves exchanges.

- [ ] **Step 1: Write tests**

```python
# tests/test_session_manager.py
"""Tests for session manager orchestrator service."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from starlette.testclient import TestClient

from session_manager.server import create_app, SessionOrchestrator


@pytest.fixture
def orchestrator_config(tmp_path, tmp_workspace):
    """Config for the session orchestrator."""
    config = {
        "model": {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "api_key": "test-key",
        },
        "workspace": str(tmp_workspace),
        "mcp_servers": {},
        "channels": {
            "signal": {
                "account": "+10000000000",
                "signal_cli": "http://localhost:8080",
                "allow_from": ["+11111111111"],
            }
        },
        "session": {
            "store": str(tmp_path / "sessions"),
            "max_history_tokens": 100000,
        },
        "heartbeat": {
            "interval": "20m",
            "prompt": "Check HEARTBEAT.md",
        },
    }
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.dump(config))
    return config


@pytest.fixture
def mock_litellm_response():
    return {
        "choices": [{"message": {"role": "assistant", "content": "I'll handle it!"}}]
    }


@pytest.fixture
def orchestrator(orchestrator_config, tmp_path):
    """Create orchestrator with mocked HTTP client."""
    return SessionOrchestrator(
        config=orchestrator_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_path / "sessions",
    )


def test_orchestrator_init(orchestrator):
    """Orchestrator initializes with config."""
    assert orchestrator.litellm_url == "http://localhost:4000"
    assert len(orchestrator.mcp_tools) >= 1  # at least workspace_fs


@pytest.mark.asyncio
async def test_handle_event(orchestrator, mock_litellm_response):
    """handle_event calls LiteLLM and saves session."""
    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hello!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'll handle it!"

    # Verify session was saved
    history = orchestrator.session.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_handle_event_includes_metadata_in_context(orchestrator, mock_litellm_response):
    """Event metadata is included in the user message for agent context."""
    captured_payload = {}

    async def mock_post(url, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hey",
        "metadata": {"message_id": "ts_456", "sender": "+11111111111"},
    }

    await orchestrator.handle_event(event)

    # The user message sent to LiteLLM should include source context
    last_msg = captured_payload["messages"][-1]
    assert "signal" in last_msg["content"]
    assert "Hey" in last_msg["content"]


@pytest.mark.asyncio
async def test_handle_heartbeat(orchestrator, mock_litellm_response):
    """Heartbeat generates an event with the configured prompt."""
    captured_payload = {}

    async def mock_post(url, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    result = await orchestrator.handle_heartbeat()

    last_msg = captured_payload["messages"][-1]
    assert "HEARTBEAT" in last_msg["content"]
    assert "Check HEARTBEAT.md" in last_msg["content"]


def test_http_event_endpoint(orchestrator, mock_litellm_response):
    """POST /event accepts events via HTTP."""
    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    app = create_app(orchestrator)
    client = TestClient(app)

    resp = client.post("/event", json={
        "source": "signal",
        "session_id": "+11111111111",
        "text": "test via HTTP",
        "metadata": {},
    })
    assert resp.status_code == 200
    assert "response" in resp.json()


def test_http_heartbeat_endpoint(orchestrator, mock_litellm_response):
    """POST /heartbeat triggers heartbeat processing."""
    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_litellm_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    app = create_app(orchestrator)
    client = TestClient(app)

    resp = client.post("/heartbeat")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix-shell --run "pytest tests/test_session_manager.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'session_manager'`

- [ ] **Step 3: Create session_manager package**

```python
# src/session_manager/__init__.py
```

```python
# src/session_manager/server.py
"""Session manager: central orchestrator that receives events and drives LLM calls."""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_platform.config import build_mcp_tool_declarations
from agent_platform.session import SessionManager

logger = logging.getLogger(__name__)


class SessionOrchestrator:
    """Receives events from adapters, maintains session history, drives LLM calls."""

    def __init__(self, config: dict, litellm_url: str, session_dir: Path):
        self.config = config
        self.litellm_url = litellm_url.rstrip("/")
        self.model = "main"

        session_config = config.get("session", {})
        max_tokens = session_config.get("max_history_tokens", 100000)
        self.session = SessionManager(store_dir=session_dir, max_history_tokens=max_tokens)

        self.mcp_tools = build_mcp_tool_declarations(config)

        self.heartbeat_prompt = config.get("heartbeat", {}).get(
            "prompt", "Check HEARTBEAT.md"
        )
        self.heartbeat_source = "heartbeat"
        self.heartbeat_session_id = "system"

        self._http = httpx.AsyncClient(timeout=300)

    async def handle_event(self, event: dict[str, Any]) -> str | None:
        """Process an inbound event from any adapter.

        Event format:
            {
                "source": "signal",
                "session_id": "+16092409191",
                "text": "hey what's up",
                "metadata": {"message_id": "ts_123", "sender": "+16092409191"}
            }
        """
        source = event["source"]
        session_id = event["session_id"]
        text = event.get("text", "")
        metadata = event.get("metadata", {})

        # Load session history
        history = self.session.load_truncated(
            channel=source,
            session_id=session_id,
            model=self.config["model"]["model"],
        )

        # Build user message with source context
        user_content = f"[{source}] {text}"
        if metadata:
            user_content = f"[{source} | {json.dumps(metadata)}] {text}"
        user_msg = {"role": "user", "content": user_content}

        # Build chat completion request
        messages = history + [user_msg]
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": self.mcp_tools,
        }

        # Call LiteLLM
        try:
            resp = await self._http.post(
                f"{self.litellm_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            logger.error(f"LiteLLM request failed: {e}")
            return None

        assistant_msg = result["choices"][0]["message"]
        assistant_text = assistant_msg.get("content", "")

        # Save the exchange to session
        self.session.append(source, session_id, [
            user_msg,
            {"role": "assistant", "content": assistant_text},
        ])

        return assistant_text

    async def handle_heartbeat(self) -> str | None:
        """Process a heartbeat by generating an event with the configured prompt."""
        event = {
            "source": self.heartbeat_source,
            "session_id": self.heartbeat_session_id,
            "text": f"[HEARTBEAT] {self.heartbeat_prompt}",
            "metadata": {},
        }
        return await self.handle_event(event)

    async def close(self) -> None:
        await self._http.aclose()


def create_app(orchestrator: SessionOrchestrator) -> Starlette:
    """Create the HTTP application for the session manager."""

    async def event_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        result = await orchestrator.handle_event(body)
        return JSONResponse({"status": "ok", "response": result})

    async def heartbeat_endpoint(request: Request) -> JSONResponse:
        result = await orchestrator.handle_heartbeat()
        return JSONResponse({"status": "ok", "response": result})

    return Starlette(routes=[
        Route("/event", event_endpoint, methods=["POST"]),
        Route("/heartbeat", heartbeat_endpoint, methods=["POST"]),
    ])
```

- [ ] **Step 4: Run tests**

Run: `nix-shell --run "pytest tests/test_session_manager.py -v"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/session_manager/ tests/test_session_manager.py
git commit -m "feat: session manager orchestrator with HTTP event and heartbeat endpoints"
```

---

## Task 3: Create Signal outbound MCP

**Files:**
- Create: `src/adapters/signal/outbound_mcp.py`
- Create: `tests/test_signal_outbound.py`
- Delete: `src/adapters/signal/channel_mcp.py` (after outbound_mcp is working)
- Delete: `tests/test_channel_mcp.py` (after test_signal_outbound is working)

Refactor the channel MCP into outbound-only tools. Tool signatures change: `recipient`/`message_id` instead of `conversation_id`. Add `send_message` as the primary text sending tool.

- [ ] **Step 1: Write tests**

```python
# tests/test_signal_outbound.py
"""Tests for Signal outbound MCP server."""

import pytest
from unittest.mock import AsyncMock

from adapters.signal.outbound_mcp import create_signal_mcp, SignalSender


@pytest.fixture
def mock_sender():
    """A mock SignalSender for testing tools without signal-cli."""
    sender = SignalSender(
        signal_cli_url="http://localhost:8080",
        account="+10000000000",
    )
    sender._http = AsyncMock()
    return sender


def test_signal_sender_init(mock_sender):
    """SignalSender initializes with signal-cli URL and account."""
    assert mock_sender.account == "+10000000000"


@pytest.mark.asyncio
async def test_mcp_has_expected_tools():
    """The MCP server exposes the expected Signal tools."""
    sender = SignalSender(
        signal_cli_url="http://localhost:8080",
        account="+10000000000",
    )
    mcp = create_signal_mcp(sender)
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "send_message" in tool_names
    assert "send_attachment" in tool_names
    assert "react" in tool_names
    assert "set_typing" in tool_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix-shell --run "pytest tests/test_signal_outbound.py -v"`
Expected: FAIL

- [ ] **Step 3: Implement outbound_mcp.py**

```python
# src/adapters/signal/outbound_mcp.py
"""Signal outbound MCP server: tools for sending messages, attachments, reactions."""

import base64
import logging
from pathlib import Path

import httpx
from fastmcp import FastMCP

logger = logging.getLogger(__name__)


class SignalSender:
    """Handles sending messages via signal-cli REST API."""

    def __init__(self, signal_cli_url: str, account: str):
        self.signal_cli_url = signal_cli_url.rstrip("/")
        self.account = account
        self._http = httpx.AsyncClient(timeout=60)

    async def send_text(self, recipient: str, text: str) -> None:
        """Send a text message via signal-cli."""
        await self._http.post(
            f"{self.signal_cli_url}/v2/send",
            json={
                "message": text,
                "number": self.account,
                "recipients": [recipient],
            },
        )

    async def send_file(
        self, recipient: str, file_path: str, mime_type: str, caption: str | None = None
    ) -> None:
        """Send a file attachment via signal-cli."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        data = base64.b64encode(path.read_bytes()).decode()
        payload = {
            "message": caption or "",
            "number": self.account,
            "recipients": [recipient],
            "base64_attachments": [data],
        }
        await self._http.post(f"{self.signal_cli_url}/v2/send", json=payload)

        # Clean up attachment file after successful send
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")

    async def close(self) -> None:
        await self._http.aclose()


def create_signal_mcp(sender: SignalSender) -> FastMCP:
    """Create the Signal outbound MCP server."""
    mcp = FastMCP("signal")

    @mcp.tool
    async def send_message(recipient: str, text: str) -> str:
        """Send a text message to a recipient on Signal."""
        try:
            await sender.send_text(recipient, text)
            return f"Sent message to {recipient}"
        except Exception as e:
            return f"Error sending message: {e}"

    @mcp.tool
    async def send_attachment(
        recipient: str, file_path: str, mime_type: str, caption: str | None = None
    ) -> str:
        """Send a file attachment to a recipient on Signal."""
        try:
            await sender.send_file(recipient, file_path, mime_type, caption)
            return f"Sent {file_path} to {recipient}"
        except Exception as e:
            return f"Error sending attachment: {e}"

    @mcp.tool
    async def react(message_id: str, emoji: str) -> str:
        """React to a message with an emoji."""
        # TODO: wire to signal-cli reaction API
        return f"Reacted with {emoji} to {message_id} (stub)"

    @mcp.tool
    async def set_typing(recipient: str, enabled: bool) -> str:
        """Show or hide the typing indicator for a recipient."""
        # TODO: wire to signal-cli typing API
        status = "started" if enabled else "stopped"
        return f"Typing {status} for {recipient} (stub)"

    return mcp
```

- [ ] **Step 4: Run tests**

Run: `nix-shell --run "pytest tests/test_signal_outbound.py -v"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/adapters/signal/outbound_mcp.py tests/test_signal_outbound.py
git commit -m "feat: Signal outbound MCP server with send_message, react, typing tools"
```

---

## Task 4: Create Signal inbound listener

**Files:**
- Create: `src/adapters/signal/inbound.py`
- Create: `tests/test_signal_inbound.py`

The inbound listener polls signal-cli and pushes events to the session manager via `POST /event`.

- [ ] **Step 1: Write tests**

```python
# tests/test_signal_inbound.py
"""Tests for Signal inbound listener."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from adapters.signal.inbound import SignalInbound


@pytest.fixture
def inbound():
    return SignalInbound(
        signal_cli_url="http://localhost:8080",
        session_manager_url="http://localhost:5000",
        account="+10000000000",
        allow_from=["+11111111111"],
    )


def test_inbound_init(inbound):
    """Inbound listener initializes with config."""
    assert inbound.account == "+10000000000"
    assert inbound.allow_from == ["+11111111111"]


@pytest.mark.asyncio
async def test_poll_pushes_event_to_session_manager(inbound):
    """Polling signal-cli pushes normalized events to session manager."""
    signal_response = [
        {
            "envelope": {
                "source": "+11111111111",
                "dataMessage": {
                    "message": "Hello from Signal",
                    "timestamp": 1234567890,
                },
            }
        }
    ]

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=signal_response)
        return resp

    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"status": "ok"})
        return resp

    inbound._http = AsyncMock()
    inbound._http.get = AsyncMock(side_effect=mock_get)
    inbound._http.post = AsyncMock(side_effect=mock_post)

    await inbound.poll_messages()

    # Verify event was pushed to session manager
    inbound._http.post.assert_called_once()
    call_args = inbound._http.post.call_args
    assert "/event" in call_args.args[0]
    event = call_args.kwargs["json"]
    assert event["source"] == "signal"
    assert event["session_id"] == "+11111111111"
    assert event["text"] == "Hello from Signal"


@pytest.mark.asyncio
async def test_poll_filters_unauthorized(inbound):
    """Messages from unauthorized senders are not pushed."""
    signal_response = [
        {
            "envelope": {
                "source": "+19999999999",
                "dataMessage": {
                    "message": "I'm not allowed",
                    "timestamp": 1234567890,
                },
            }
        }
    ]

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=signal_response)
        return resp

    inbound._http = AsyncMock()
    inbound._http.get = AsyncMock(side_effect=mock_get)
    inbound._http.post = AsyncMock()

    await inbound.poll_messages()

    # Should not push any event
    inbound._http.post.assert_not_called()


@pytest.mark.asyncio
async def test_poll_skips_empty_messages(inbound):
    """Envelopes without dataMessage or text are skipped."""
    signal_response = [
        {"envelope": {"source": "+11111111111"}},  # no dataMessage
        {"envelope": {"source": "+11111111111", "dataMessage": {}}},  # no message text
    ]

    async def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=signal_response)
        return resp

    inbound._http = AsyncMock()
    inbound._http.get = AsyncMock(side_effect=mock_get)
    inbound._http.post = AsyncMock()

    await inbound.poll_messages()

    inbound._http.post.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix-shell --run "pytest tests/test_signal_inbound.py -v"`
Expected: FAIL

- [ ] **Step 3: Implement inbound.py**

```python
# src/adapters/signal/inbound.py
"""Signal inbound listener: polls signal-cli and pushes events to session manager."""

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


class SignalInbound:
    """Polls signal-cli for messages and pushes events to the session manager."""

    def __init__(
        self,
        signal_cli_url: str,
        session_manager_url: str,
        account: str,
        allow_from: list[str] | None = None,
    ):
        self.signal_cli_url = signal_cli_url.rstrip("/")
        self.session_manager_url = session_manager_url.rstrip("/")
        self.account = account
        self.allow_from = allow_from or []
        self._http = httpx.AsyncClient(timeout=60)

    async def poll_messages(self) -> None:
        """Poll signal-cli for new messages and push events to session manager."""
        try:
            resp = await self._http.get(
                f"{self.signal_cli_url}/v1/receive/{self.account}",
            )
            resp.raise_for_status()
            messages = resp.json()
        except Exception as e:
            logger.error(f"Failed to poll signal-cli: {e}")
            return

        for envelope in messages:
            data_msg = envelope.get("envelope", {}).get("dataMessage")
            if not data_msg or not data_msg.get("message"):
                continue

            sender = envelope["envelope"].get("source")
            if self.allow_from and sender not in self.allow_from:
                logger.debug(f"Ignoring message from unauthorized sender: {sender}")
                continue

            text = data_msg["message"]
            timestamp = str(data_msg.get("timestamp", ""))

            event = {
                "source": "signal",
                "session_id": sender,
                "text": text,
                "metadata": {
                    "message_id": timestamp,
                    "sender": sender,
                },
            }

            try:
                await self._http.post(
                    f"{self.session_manager_url}/event",
                    json=event,
                )
            except Exception as e:
                logger.error(f"Failed to push event to session manager: {e}")

    async def run(self, poll_interval: float = 1.0) -> None:
        """Main loop: poll signal-cli for messages."""
        logger.info(f"Signal inbound listener started. Account: {self.account}")
        while True:
            await self.poll_messages()
            await asyncio.sleep(poll_interval)

    async def close(self) -> None:
        await self._http.aclose()
```

- [ ] **Step 4: Run tests**

Run: `nix-shell --run "pytest tests/test_signal_inbound.py -v"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/adapters/signal/inbound.py tests/test_signal_inbound.py
git commit -m "feat: Signal inbound listener pushes events to session manager"
```

---

## Task 5: Update Signal adapter main.py and Dockerfile

**Files:**
- Rewrite: `src/adapters/signal/main.py`
- Modify: `src/adapters/signal/Dockerfile`
- Modify: `src/adapters/signal/requirements.txt`

The Signal adapter process now runs the inbound listener and outbound MCP server. It no longer calls LiteLLM directly.

- [ ] **Step 1: Rewrite main.py**

```python
# src/adapters/signal/main.py
"""Signal adapter: runs inbound listener + outbound MCP server."""

import asyncio
import logging
import os

from adapters.signal.inbound import SignalInbound
from adapters.signal.outbound_mcp import SignalSender, create_signal_mcp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    signal_cli_url = os.environ.get("SIGNAL_CLI_URL", "http://localhost:8080")
    session_manager_url = os.environ.get("SESSION_MANAGER_URL", "http://localhost:5000")
    account = os.environ.get("SIGNAL_ACCOUNT", "")
    allow_from = os.environ.get("ALLOW_FROM", "").split(",") if os.environ.get("ALLOW_FROM") else []
    mcp_port = int(os.environ.get("CHANNEL_MCP_PORT", "8100"))

    # Inbound: polls signal-cli, pushes events to session manager
    inbound = SignalInbound(
        signal_cli_url=signal_cli_url,
        session_manager_url=session_manager_url,
        account=account,
        allow_from=allow_from,
    )

    # Outbound: MCP server with send_message, react, etc.
    sender = SignalSender(signal_cli_url=signal_cli_url, account=account)
    mcp = create_signal_mcp(sender)

    async def run_mcp():
        await mcp.run_async(transport="http", host="0.0.0.0", port=mcp_port)

    try:
        await asyncio.gather(
            inbound.run(),
            run_mcp(),
        )
    finally:
        await inbound.close()
        await sender.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Update Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../../adapters/ ./src/adapters/
COPY ../../agent_platform/ ./src/agent_platform/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "adapters.signal.main"]
```

Note: the Dockerfile copies the full src tree. Review the existing Dockerfile pattern and match it.

- [ ] **Step 3: Update requirements.txt**

```
fastmcp>=2.0
httpx>=0.27
uvicorn>=0.30
```

Note: signal adapter no longer needs litellm (session manager handles that).

- [ ] **Step 4: Commit**

```bash
git add src/adapters/signal/main.py src/adapters/signal/Dockerfile src/adapters/signal/requirements.txt
git commit -m "refactor: Signal adapter runs inbound + outbound, no longer calls LiteLLM"
```

---

## Task 6: Create session manager Dockerfile and entrypoint

**Files:**
- Create: `src/session_manager/Dockerfile`
- Create: `src/session_manager/requirements.txt`
- Create: `src/session_manager/main.py`

- [ ] **Step 1: Create main.py entrypoint**

```python
# src/session_manager/main.py
"""Entrypoint for the session manager service."""

import logging
import os
from pathlib import Path

import uvicorn

from agent_platform.config import load_config
from session_manager.server import SessionOrchestrator, create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def main():
    config_path = Path(os.environ.get("AGENT_CONFIG", "agent.yaml"))
    config = load_config(config_path)

    litellm_url = os.environ.get("LITELLM_URL", "http://localhost:4000")
    session_dir = Path(config.get("session", {}).get("store", "./sessions"))
    port = int(os.environ.get("SESSION_MANAGER_PORT", "5000"))

    orchestrator = SessionOrchestrator(
        config=config,
        litellm_url=litellm_url,
        session_dir=session_dir,
    )

    app = create_app(orchestrator)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create requirements.txt**

```
fastmcp>=2.0
litellm>=1.80
httpx>=0.27
pyyaml>=6
uvicorn>=0.30
starlette>=0.40
```

- [ ] **Step 3: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../agent_platform/ ./src/agent_platform/
COPY . ./src/session_manager/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "session_manager.main"]
```

Note: Review the existing signal adapter Dockerfile for the actual COPY pattern used in this repo and match it.

- [ ] **Step 4: Commit**

```bash
git add src/session_manager/main.py src/session_manager/Dockerfile src/session_manager/requirements.txt
git commit -m "feat: session manager service Dockerfile and entrypoint"
```

---

## Task 7: Update config.py and docker-compose

**Files:**
- Modify: `src/agent_platform/config.py`
- Modify: `docker-compose.yaml`
- Modify: `tests/test_config.py`

Update the config loader to generate the correct MCP server labels for the new architecture, and update docker-compose for the new service topology.

- [ ] **Step 1: Update config.py — rename signal_channel to signal**

In `generate_litellm_config`, change the implicit adapter MCP server label from `"signal_channel"` to `"signal"`:

In `src/agent_platform/config.py`, find:
```python
            "signal_channel": {
```
Replace with:
```python
            "signal": {
```

And in `build_mcp_tool_declarations`, find:
```python
        {"type": "mcp", "server_label": "signal_channel", "require_approval": "never"},
```
Replace with:
```python
        {"type": "mcp", "server_label": "signal", "require_approval": "never"},
```

- [ ] **Step 2: Update test_config.py**

In `tests/test_config.py`, replace all occurrences of `"signal_channel"` with `"signal"`.

- [ ] **Step 3: Run config tests**

Run: `nix-shell --run "pytest tests/test_config.py -v"`
Expected: All PASS

- [ ] **Step 4: Update docker-compose.yaml**

```yaml
services:
  session-manager:
    build:
      context: .
      dockerfile: src/session_manager/Dockerfile
    volumes:
      - ./sessions:/sessions
      - ./agent.yaml:/app/agent.yaml:ro
    ports:
      - "5000:5000"
    environment:
      - LITELLM_URL=http://litellm:4000
      - AGENT_CONFIG=/app/agent.yaml
      - SESSION_MANAGER_PORT=5000
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
    build:
      context: .
      dockerfile: src/adapters/signal/Dockerfile
    ports:
      - "8100:8100"
    environment:
      - SESSION_MANAGER_URL=http://session-manager:5000
      - SIGNAL_CLI_URL=http://signal-cli:8080
      - SIGNAL_ACCOUNT=${SIGNAL_ACCOUNT}
      - ALLOW_FROM=${SIGNAL_ALLOW_FROM}
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

- [ ] **Step 5: Commit**

```bash
git add src/agent_platform/config.py tests/test_config.py docker-compose.yaml
git commit -m "refactor: update config labels and docker-compose for session manager architecture"
```

---

## Task 8: Clean up old files and update integration test

**Files:**
- Delete: `src/adapters/signal/adapter.py`
- Delete: `src/adapters/signal/channel_mcp.py`
- Delete: `tests/test_adapter.py`
- Delete: `tests/test_channel_mcp.py`
- Rewrite: `tests/test_integration.py`

- [ ] **Step 1: Delete old files**

```bash
git rm src/adapters/signal/adapter.py src/adapters/signal/channel_mcp.py
git rm tests/test_adapter.py tests/test_channel_mcp.py
```

- [ ] **Step 2: Rewrite integration test**

```python
# tests/test_integration.py
"""Integration smoke test: verify the full event flow with mocked externals."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from agent_platform.config import load_config, generate_litellm_config, build_mcp_tool_declarations
from agent_platform.session import SessionManager
from agent_platform.callbacks.system_prompt import build_system_message
from session_manager.server import SessionOrchestrator, create_app
from adapters.signal.inbound import SignalInbound
from adapters.signal.outbound_mcp import SignalSender, create_signal_mcp


@pytest.mark.asyncio
async def test_full_event_flow(tmp_path):
    """End-to-end: config → session manager receives event → LLM called → session saved."""
    # 1. Set up workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "SOUL.md").write_text("# Soul\nYou are a helpful agent.\n")

    # 2. Create config
    config_data = {
        "model": {"provider": "anthropic", "model": "claude-opus-4-6", "api_key": "test-key"},
        "workspace": str(workspace),
        "mcp_servers": {},
        "channels": {
            "signal": {
                "account": "+10000000000",
                "signal_cli": "http://localhost:8080",
                "allow_from": ["+11111111111"],
            },
        },
        "session": {"store": str(tmp_path / "sessions"), "max_history_tokens": 100000},
        "heartbeat": {"interval": "20m", "prompt": "Check HEARTBEAT.md"},
    }
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.dump(config_data))
    config = load_config(config_path)

    # 3. Verify system prompt injection
    system_msg = build_system_message(workspace)
    assert "helpful agent" in system_msg

    # 4. Verify LiteLLM config generation
    litellm_config = generate_litellm_config(config)
    assert "workspace_fs" in litellm_config["mcp_servers"]
    assert "signal" in litellm_config["mcp_servers"]

    # 5. Create orchestrator and handle an event
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    orchestrator = SessionOrchestrator(
        config=config,
        litellm_url="http://localhost:4000",
        session_dir=sessions_dir,
    )

    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "I'm here to help!"}}]
    }

    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=mock_response)
        return resp

    orchestrator._http = AsyncMock()
    orchestrator._http.post = AsyncMock(side_effect=mock_post)

    event = {
        "source": "signal",
        "session_id": "+11111111111",
        "text": "Hi there!",
        "metadata": {"message_id": "ts_123", "sender": "+11111111111"},
    }

    result = await orchestrator.handle_event(event)
    assert result == "I'm here to help!"

    # 6. Verify session persistence with channel key
    session = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    history = session.load("signal", "+11111111111")
    assert len(history) == 2
    assert "Hi there!" in history[0]["content"]
    assert history[1]["content"] == "I'm here to help!"

    # 7. Verify outbound MCP tools exist
    sender = SignalSender(signal_cli_url="http://localhost:8080", account="+10000000000")
    mcp = create_signal_mcp(sender)
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "send_message" in tool_names
    assert "send_attachment" in tool_names

    # 8. Verify inbound listener can be created
    inbound = SignalInbound(
        signal_cli_url="http://localhost:8080",
        session_manager_url="http://localhost:5000",
        account="+10000000000",
        allow_from=["+11111111111"],
    )
    assert inbound.account == "+10000000000"
```

- [ ] **Step 3: Run full test suite**

Run: `nix-shell --run "pytest -v"`
Expected: All tests PASS (no references to deleted modules)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: clean up old adapter code, rewrite integration test for event-driven architecture"
```

---

## Task 9: Update pyproject.toml and conftest

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

Add `session_manager` to the build packages and add `starlette` as a dependency.

- [ ] **Step 1: Update pyproject.toml**

In `[tool.hatch.build.targets.wheel]`, change:
```toml
packages = ["src/agent_platform", "src/adapters"]
```
to:
```toml
packages = ["src/agent_platform", "src/adapters", "src/session_manager", "src/mcp_servers"]
```

Add `starlette>=0.40` to dependencies list.

- [ ] **Step 2: Reinstall**

```bash
nix-shell --run "pip install -e '.[dev]'"
```

- [ ] **Step 3: Run full test suite**

Run: `nix-shell --run "pytest -v --tb=short"`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add session_manager and mcp_servers to build packages, add starlette dep"
```

---

## Summary

| Task | What | Files Changed |
|------|------|--------------|
| 1 | Session keys: `(channel, session_id)` | session.py, test_session.py |
| 2 | Session manager orchestrator | NEW: session_manager/server.py, test_session_manager.py |
| 3 | Signal outbound MCP | NEW: outbound_mcp.py, test_signal_outbound.py |
| 4 | Signal inbound listener | NEW: inbound.py, test_signal_inbound.py |
| 5 | Signal adapter main.py + Docker | main.py, Dockerfile, requirements.txt |
| 6 | Session manager Docker + entrypoint | NEW: main.py, Dockerfile, requirements.txt |
| 7 | Config labels + docker-compose | config.py, test_config.py, docker-compose.yaml |
| 8 | Delete old files + integration test | DELETE: adapter.py, channel_mcp.py, old tests |
| 9 | Build config + deps | pyproject.toml |

**Net result:** Session manager is the orchestrator. Signal adapter is inbound + outbound. Agent decides where to respond via MCP tools. Architecture supports adding future adapters (Discord, coding tools) without changing the session manager.
