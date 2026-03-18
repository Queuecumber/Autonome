# Agent Platform v1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, composable agent runtime that replaces OpenClaw — Signal chat with an AI agent backed by LiteLLM and MCP servers.

**Architecture:** Signal adapter (Python) receives messages, manages session history, and runs a channel MCP server for Signal-specific tools. It sends chat completion requests to a LiteLLM proxy, which handles model routing, system prompt injection (via custom callback), and MCP tool execution. A filesystem MCP server provides workspace access. Everything is wired together by a config loader that reads a single `agent.yaml`.

**Tech Stack:** Python 3.12+, FastMCP, LiteLLM, signal-cli REST API, Docker/docker-compose, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-agent-platform-v1-design.md`

---

## File Structure

```
agent-platform/
├── pyproject.toml                          # Project metadata, dependencies, pytest config
├── agent.yaml.example                      # Example config for users to copy and edit
├── docker-compose.yaml                     # Orchestrates all services
├── .gitignore
├── src/
│   ├── agent_platform/
│   │   ├── __init__.py
│   │   ├── config.py                       # Config loader: reads agent.yaml, generates LiteLLM config
│   │   ├── session.py                      # Session manager: JSONL history, token counting, truncation
│   │   └── callbacks/
│   │       ├── __init__.py
│   │       └── system_prompt.py            # LiteLLM callback: auto-discovers workspace *.md, injects system prompt
│   └── adapters/
│       └── signal/
│           ├── __init__.py
│           ├── adapter.py                  # Message bridge: signal-cli ↔ LiteLLM, heartbeat endpoint
│           ├── channel_mcp.py              # Signal channel MCP: stage/send attachment, react, typing
│           ├── main.py                     # Entrypoint: starts adapter + channel MCP in same process
│           ├── Dockerfile
│           └── requirements.txt
├── tests/
│   ├── conftest.py                         # Shared fixtures (tmp workspace, sample config)
│   ├── test_config.py                      # Config loader tests
│   ├── test_session.py                     # Session manager tests
│   ├── test_system_prompt.py               # System prompt callback tests
│   ├── test_channel_mcp.py                 # Channel MCP tool tests
│   └── test_adapter.py                     # Signal adapter tests (mocked signal-cli + LiteLLM)
├── workspace/                              # Agent workspace (not in git, user creates)
├── sessions/                               # JSONL session files (not in git)
└── generated/                              # Generated configs (not in git)
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/agent_platform/__init__.py`
- Create: `src/agent_platform/callbacks/__init__.py`
- Create: `src/adapters/signal/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /home/max/projects/personal/AgentPlatform
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "agent-platform"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastmcp>=2.0",
    "litellm>=1.80",
    "httpx>=0.27",
    "pyyaml>=6",
    "uvicorn>=0.30",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "pytest-httpx>=0.30",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agent_platform", "src/adapters"]
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
generated/
sessions/
workspace/
.env
*.egg-info/
dist/
```

- [ ] **Step 4: Create empty __init__.py files**

Create empty files at:
- `src/agent_platform/__init__.py`
- `src/agent_platform/callbacks/__init__.py`
- `src/adapters/__init__.py`
- `src/adapters/signal/__init__.py`

- [ ] **Step 5: Create tests/conftest.py with shared fixtures**

```python
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with sample personality files."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "SOUL.md").write_text("# Soul\nI am a test agent.\n")
    (ws / "USER.md").write_text("# User\nTest user.\n")
    (ws / "memory").mkdir()
    return ws


@pytest.fixture
def sample_config(tmp_path, tmp_workspace):
    """Create a minimal agent.yaml for testing."""
    config = {
        "model": {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "api_key": "${ANTHROPIC_API_KEY}",
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
    return config_path


@pytest.fixture
def tmp_sessions(tmp_path):
    """Create a temporary sessions directory."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions
```

- [ ] **Step 6: Install project in dev mode**

```bash
cd /home/max/projects/personal/AgentPlatform
python -m venv .venv
source .venv/bin/activate.fish
pip install -e ".[dev]"
```

- [ ] **Step 7: Run pytest to verify setup**

Run: `cd /home/max/projects/personal/AgentPlatform && .venv/bin/pytest --co -q`
Expected: `no tests ran` (no errors)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/conftest.py
git commit -m "feat: project scaffolding with dependencies and test fixtures"
```

---

## Task 2: Config Loader

**Files:**
- Create: `src/agent_platform/config.py`
- Test: `tests/test_config.py`

The config loader reads `agent.yaml`, substitutes environment variables, and generates LiteLLM's native config format. It also produces the list of MCP tool declarations that the adapter injects into every request.

- [ ] **Step 1: Write test for loading and parsing agent.yaml**

```python
# tests/test_config.py
import os
from pathlib import Path

import yaml

from agent_platform.config import load_config


def test_load_config_basic(sample_config):
    """Config loads and has expected top-level keys."""
    config = load_config(sample_config)
    assert config["model"]["provider"] == "anthropic"
    assert config["model"]["model"] == "claude-opus-4-6"
    assert "channels" in config
    assert "session" in config
    assert "heartbeat" in config
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_load_config_basic -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_platform.config'`

- [ ] **Step 3: Implement load_config**

```python
# src/agent_platform/config.py
"""Config loader: reads agent.yaml, generates LiteLLM config, produces MCP tool declarations."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""
    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return re.sub(r"\$\{(\w+)\}", _replace, value)


def _walk_substitute(obj: Any) -> Any:
    """Recursively substitute env vars in all string values."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_substitute(item) for item in obj]
    return obj


def load_config(config_path: Path) -> dict:
    """Load and parse agent.yaml with env var substitution."""
    raw = Path(config_path).read_text()
    config = yaml.safe_load(raw)
    return _walk_substitute(config)


def generate_litellm_config(
    config: dict,
    adapter_host: str = "signal-adapter",
    adapter_mcp_port: int = 8100,
    fs_mcp_host: str = "workspace-fs-mcp",
    fs_mcp_port: int = 8000,
) -> dict:
    """Generate LiteLLM config.yaml content from agent config."""
    model = config["model"]
    provider = model["provider"]
    model_name = model["model"]
    api_key = model.get("api_key", "")

    litellm_config: dict[str, Any] = {
        "model_list": [
            {
                "model_name": "main",
                "litellm_params": {
                    "model": f"{provider}/{model_name}",
                    "api_key": api_key,
                },
            }
        ],
        "mcp_servers": {
            "workspace_fs": {
                "url": f"http://{fs_mcp_host}:{fs_mcp_port}/mcp",
                "transport": "http",
            },
            "signal_channel": {
                "url": f"http://{adapter_host}:{adapter_mcp_port}/mcp",
                "transport": "http",
            },
        },
    }

    # Add user-defined MCP servers
    for name, server_config in config.get("mcp_servers", {}).items():
        entry: dict[str, Any] = {}
        if "url" in server_config:
            entry["url"] = server_config["url"]
            entry["transport"] = "http"
            if "headers" in server_config:
                entry["headers"] = server_config["headers"]
        elif "command" in server_config:
            entry["transport"] = "stdio"
            entry["command"] = server_config["command"]
            if "args" in server_config:
                entry["args"] = server_config["args"]
            if "env" in server_config:
                entry["env"] = server_config["env"]
        litellm_config["mcp_servers"][name] = entry

    return litellm_config


def build_mcp_tool_declarations(config: dict) -> list[dict]:
    """Build the MCP tool declarations list for chat completion requests."""
    tools = [
        {"type": "mcp", "server_label": "workspace_fs", "require_approval": "never"},
        {"type": "mcp", "server_label": "signal_channel", "require_approval": "never"},
    ]
    for name in config.get("mcp_servers", {}):
        tools.append({"type": "mcp", "server_label": name, "require_approval": "never"})
    return tools


def write_litellm_config(litellm_config: dict, output_path: Path) -> None:
    """Write generated LiteLLM config to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"# generated/litellm-config.yaml (DO NOT EDIT — generated from agent.yaml)\n\n"
        + yaml.dump(litellm_config, default_flow_style=False, sort_keys=False)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py::test_load_config_basic -v`
Expected: PASS

- [ ] **Step 5: Write test for env var substitution**

```python
# Append to tests/test_config.py

def test_env_var_substitution(tmp_path, monkeypatch):
    """Environment variables in config values get substituted."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test-123")

    config_data = {
        "model": {
            "provider": "anthropic",
            "model": "test",
            "api_key": "${TEST_API_KEY}",
        },
        "workspace": str(tmp_path),
        "channels": {},
        "session": {"store": str(tmp_path), "max_history_tokens": 1000},
        "heartbeat": {"interval": "20m", "prompt": "test"},
    }
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    assert config["model"]["api_key"] == "sk-test-123"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py::test_env_var_substitution -v`
Expected: PASS

- [ ] **Step 7: Write test for LiteLLM config generation**

```python
# Append to tests/test_config.py

from agent_platform.config import generate_litellm_config, build_mcp_tool_declarations


def test_generate_litellm_config_basic(sample_config):
    """Generated LiteLLM config has model list and implicit MCP servers."""
    config = load_config(sample_config)
    litellm = generate_litellm_config(config)

    assert len(litellm["model_list"]) == 1
    assert litellm["model_list"][0]["model_name"] == "main"
    assert "workspace_fs" in litellm["mcp_servers"]
    assert "signal_channel" in litellm["mcp_servers"]


def test_generate_litellm_config_with_user_mcp(tmp_path, tmp_workspace):
    """User-defined MCP servers appear in generated config."""
    config_data = {
        "model": {"provider": "anthropic", "model": "test", "api_key": "key"},
        "workspace": str(tmp_workspace),
        "mcp_servers": {
            "home_assistant": {
                "url": "http://ha.local/mcp",
                "headers": {"Authorization": "Bearer token"},
            },
            "custom_tool": {
                "command": "python3",
                "args": ["./custom.py"],
                "env": {"KEY": "val"},
            },
        },
        "channels": {},
        "session": {"store": str(tmp_path), "max_history_tokens": 1000},
        "heartbeat": {"interval": "20m", "prompt": "test"},
    }
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_config(config_path)
    litellm = generate_litellm_config(config)

    assert litellm["mcp_servers"]["home_assistant"]["transport"] == "http"
    assert litellm["mcp_servers"]["home_assistant"]["headers"]["Authorization"] == "Bearer token"
    assert litellm["mcp_servers"]["custom_tool"]["transport"] == "stdio"
    assert litellm["mcp_servers"]["custom_tool"]["command"] == "python3"


def test_build_mcp_tool_declarations(sample_config):
    """Tool declarations include implicit + user-defined servers."""
    config = load_config(sample_config)
    tools = build_mcp_tool_declarations(config)

    labels = [t["server_label"] for t in tools]
    assert "workspace_fs" in labels
    assert "signal_channel" in labels
    assert all(t["require_approval"] == "never" for t in tools)
```

- [ ] **Step 8: Run all config tests**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/agent_platform/config.py tests/test_config.py
git commit -m "feat: config loader with env var substitution and LiteLLM config generation"
```

---

## Task 3: Session Manager

**Files:**
- Create: `src/agent_platform/session.py`
- Test: `tests/test_session.py`

Manages per-contact JSONL conversation history. Loads history, appends exchanges, truncates oldest complete exchanges when over token budget.

- [ ] **Step 1: Write test for creating and loading empty session**

```python
# tests/test_session.py
import json
from pathlib import Path

from agent_platform.session import SessionManager


def test_load_empty_session(tmp_sessions):
    """Loading a session for a new contact returns empty history."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    history = mgr.load("+11111111111")
    assert history == []


def test_session_file_naming(tmp_sessions):
    """Session files are named by phone number."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    path = mgr._session_path("+11111111111")
    assert path.name == "+11111111111.jsonl"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement SessionManager**

```python
# src/agent_platform/session.py
"""Session manager: JSONL history per contact with token-based truncation."""

import json
from pathlib import Path
from typing import Any

import litellm


class SessionManager:
    def __init__(self, store_dir: Path, max_history_tokens: int = 100000):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_tokens = max_history_tokens

    def _session_path(self, contact: str) -> Path:
        return self.store_dir / f"{contact}.jsonl"

    def load(self, contact: str) -> list[dict[str, Any]]:
        """Load conversation history for a contact."""
        path = self._session_path(contact)
        if not path.exists():
            return []
        messages = []
        for line in path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        return messages

    def append(self, contact: str, messages: list[dict[str, Any]]) -> None:
        """Append messages to a contact's session file."""
        path = self._session_path(contact)
        with path.open("a") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    def load_truncated(self, contact: str, model: str = "claude-opus-4-6") -> list[dict[str, Any]]:
        """Load history, truncating oldest complete exchanges if over token budget.

        An exchange is a (user, assistant) pair, possibly with tool_call/tool messages
        between them. We drop entire exchanges to avoid corrupting conversation structure.
        """
        messages = self.load(contact)
        if not messages:
            return messages

        token_count = self._count_tokens(messages, model)
        if token_count <= self.max_history_tokens:
            return messages

        # Group messages into exchanges: each starts with a "user" message
        exchanges = self._group_exchanges(messages)

        # Drop oldest exchanges until under budget
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
        """Count tokens in a message list using LiteLLM's token counter."""
        try:
            return litellm.token_counter(model=model, messages=messages)
        except Exception:
            # Fallback: rough estimate of 4 chars per token
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            return total_chars // 4
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_session.py -v`
Expected: PASS

- [ ] **Step 5: Write test for appending and reloading**

```python
# Append to tests/test_session.py

def test_append_and_reload(tmp_sessions):
    """Messages appended to session can be reloaded."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    mgr.append("+11111111111", [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    history = mgr.load("+11111111111")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"
```

- [ ] **Step 6: Run test**

Run: `.venv/bin/pytest tests/test_session.py::test_append_and_reload -v`
Expected: PASS

- [ ] **Step 7: Write test for truncation**

```python
# Append to tests/test_session.py

def test_load_truncated_drops_oldest(tmp_sessions):
    """When over token budget, oldest exchanges are dropped."""
    # Use a very low budget to force truncation
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    messages = []
    for i in range(10):
        messages.extend([
            {"role": "user", "content": f"Message {i} " + "x" * 100},
            {"role": "assistant", "content": f"Response {i} " + "y" * 100},
        ])
    mgr.append("+11111111111", messages)

    truncated = mgr.load_truncated("+11111111111")
    # Should have fewer messages than original
    assert len(truncated) < len(messages)
    # Should still have the most recent exchange
    assert truncated[-1]["content"].startswith("Response 9")


def test_load_truncated_preserves_exchange_integrity(tmp_sessions):
    """Truncation drops complete exchanges, not partial ones."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    mgr.append("+11111111111", [
        {"role": "user", "content": "old " + "x" * 200},
        {"role": "assistant", "content": "old reply " + "y" * 200},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new reply"},
    ])

    truncated = mgr.load_truncated("+11111111111")
    # If truncation happened, remaining messages should start with a user message
    if truncated:
        assert truncated[0]["role"] == "user"
```

- [ ] **Step 8: Run all session tests**

Run: `.venv/bin/pytest tests/test_session.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/agent_platform/session.py tests/test_session.py
git commit -m "feat: session manager with JSONL persistence and token-based truncation"
```

---

## Task 4: System Prompt Injection Callback

**Files:**
- Create: `src/agent_platform/callbacks/system_prompt.py`
- Test: `tests/test_system_prompt.py`

A LiteLLM `CustomLogger` callback that reads all `*.md` files from the workspace root and prepends them as a system message on every chat completion request.

- [ ] **Step 1: Write test for markdown discovery**

```python
# tests/test_system_prompt.py
from pathlib import Path

from agent_platform.callbacks.system_prompt import discover_workspace_md, build_system_message


def test_discover_workspace_md(tmp_workspace):
    """Discovers all .md files in workspace root (not subdirectories)."""
    files = discover_workspace_md(tmp_workspace)
    names = [f.name for f in files]
    assert "SOUL.md" in names
    assert "USER.md" in names


def test_discover_workspace_md_sorted(tmp_workspace):
    """Files are returned in alphabetical order."""
    (tmp_workspace / "AGENTS.md").write_text("# Agents\n")
    (tmp_workspace / "ZZZZZ.md").write_text("# Last\n")
    files = discover_workspace_md(tmp_workspace)
    names = [f.name for f in files]
    assert names == sorted(names)


def test_discover_workspace_md_ignores_subdirs(tmp_workspace):
    """Files in subdirectories (like memory/) are not included."""
    (tmp_workspace / "memory" / "2026-03-17.md").write_text("# Daily\n")
    files = discover_workspace_md(tmp_workspace)
    names = [f.name for f in files]
    assert "2026-03-17.md" not in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_system_prompt.py -v`
Expected: FAIL

- [ ] **Step 3: Implement system prompt module**

```python
# src/agent_platform/callbacks/system_prompt.py
"""LiteLLM callback: auto-discovers workspace *.md files and injects as system prompt."""

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.dual_cache import DualCache


def discover_workspace_md(workspace_dir: Path) -> list[Path]:
    """Find all .md files in workspace root (not subdirectories), sorted alphabetically."""
    workspace = Path(workspace_dir)
    return sorted(f for f in workspace.glob("*.md") if f.is_file())


def build_system_message(workspace_dir: Path) -> str:
    """Read and concatenate all workspace markdown files into a system prompt."""
    files = discover_workspace_md(workspace_dir)
    parts = []
    for f in files:
        parts.append(f.read_text())
    return "\n\n".join(parts)


class SystemPromptInjector(CustomLogger):
    """LiteLLM callback that prepends workspace markdown as system prompt.

    Reads WORKSPACE_DIR from environment. Set this in docker-compose.
    """

    def __init__(self):
        super().__init__()
        self.workspace_dir = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion", "text_completion", "embeddings",
            "image_generation", "moderation", "audio_transcription",
            "pass_through_endpoint", "rerank",
        ],
    ) -> Optional[Union[dict, str]]:
        if call_type != "completion":
            return None

        system_content = build_system_message(self.workspace_dir)
        if not system_content:
            return None

        system_msg = {"role": "system", "content": system_content}
        data["messages"] = [system_msg] + data.get("messages", [])
        return None


# Module-level instance — LiteLLM imports this via the callbacks config
proxy_handler_instance = SystemPromptInjector()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_system_prompt.py -v`
Expected: PASS

- [ ] **Step 5: Write test for build_system_message**

```python
# Append to tests/test_system_prompt.py

def test_build_system_message_content(tmp_workspace):
    """System message contains content from all workspace markdown files."""
    msg = build_system_message(tmp_workspace)
    assert "I am a test agent" in msg
    assert "Test user" in msg


def test_build_system_message_empty(tmp_path):
    """Empty workspace produces empty string."""
    empty = tmp_path / "empty_workspace"
    empty.mkdir()
    msg = build_system_message(empty)
    assert msg == ""
```

- [ ] **Step 6: Run all system prompt tests**

Run: `.venv/bin/pytest tests/test_system_prompt.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent_platform/callbacks/system_prompt.py tests/test_system_prompt.py
git commit -m "feat: system prompt injection callback for LiteLLM"
```

---

## Task 5: Signal Channel MCP Server

**Files:**
- Create: `src/adapters/signal/channel_mcp.py`
- Test: `tests/test_channel_mcp.py`

FastMCP HTTP server exposing Signal-specific tools: stage/send attachments, react, typing indicator. Runs in the same process as the adapter.

- [ ] **Step 1: Write test for staging attachments**

```python
# tests/test_channel_mcp.py
import pytest
from adapters.signal.channel_mcp import create_channel_mcp, StagedAttachments


def test_staged_attachments_stage_and_drain():
    """Staging attachments queues them; draining returns and clears the queue."""
    staged = StagedAttachments()
    staged.stage("conv-1", "/tmp/image.png", "image/png")
    staged.stage("conv-1", "/tmp/doc.pdf", "application/pdf")

    items = staged.drain("conv-1")
    assert len(items) == 2
    assert items[0]["file_path"] == "/tmp/image.png"
    assert items[1]["mime_type"] == "application/pdf"

    # Queue should be empty after drain
    assert staged.drain("conv-1") == []


def test_staged_attachments_separate_conversations():
    """Attachments are isolated per conversation_id."""
    staged = StagedAttachments()
    staged.stage("conv-1", "/tmp/a.png", "image/png")
    staged.stage("conv-2", "/tmp/b.png", "image/png")

    assert len(staged.drain("conv-1")) == 1
    assert len(staged.drain("conv-2")) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_channel_mcp.py -v`
Expected: FAIL

- [ ] **Step 3: Implement channel MCP**

```python
# src/adapters/signal/channel_mcp.py
"""Signal channel MCP server: attachments, reactions, typing indicator."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from fastmcp import FastMCP


class StagedAttachments:
    """In-memory attachment staging queue. Safe for single-threaded async (asyncio)."""

    def __init__(self):
        self._queues: dict[str, list[dict[str, str]]] = defaultdict(list)

    def stage(self, conversation_id: str, file_path: str, mime_type: str) -> None:
        self._queues[conversation_id].append({
            "file_path": file_path,
            "mime_type": mime_type,
        })

    def drain(self, conversation_id: str) -> list[dict[str, str]]:
        items = self._queues.pop(conversation_id, [])
        return items


# Global instance — shared between MCP tools and the adapter's message loop
staged_attachments = StagedAttachments()

# Callback for immediate sends — set by the adapter at startup
_send_callback = None


def set_send_callback(callback) -> None:
    """Register a callback for immediate attachment sends. Called by adapter."""
    global _send_callback
    _send_callback = callback


def create_channel_mcp(heartbeat_handler=None) -> FastMCP:
    """Create the Signal channel MCP server with all tools.

    Args:
        heartbeat_handler: async callable that processes a heartbeat.
            If provided, a POST /heartbeat custom route is registered.
    """
    mcp = FastMCP("signal-channel")

    if heartbeat_handler is not None:
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        @mcp.custom_route("/heartbeat", methods=["POST"])
        async def heartbeat_route(request: Request) -> JSONResponse:
            result = await heartbeat_handler()
            return JSONResponse({"status": "ok", "response": result})

    @mcp.tool
    def stage_attachment(conversation_id: str, file_path: str, mime_type: str) -> str:
        """Queue a file to send with the next text response. Multiple files can be staged."""
        staged_attachments.stage(conversation_id, file_path, mime_type)
        return f"Staged {file_path} for conversation {conversation_id}"

    @mcp.tool
    async def send_attachment(
        conversation_id: str, file_path: str, mime_type: str, caption: str | None = None
    ) -> str:
        """Send a file immediately to the current conversation."""
        if _send_callback is None:
            return "Error: send callback not registered"
        await _send_callback(conversation_id, file_path, mime_type, caption)
        return f"Sent {file_path} to conversation {conversation_id}"

    @mcp.tool
    async def react(conversation_id: str, message_id: str, emoji: str) -> str:
        """React to a message with an emoji."""
        if _send_callback is None:
            return "Error: send callback not registered"
        # Note: the adapter registers a general-purpose callback that dispatches
        # based on action type. For now, react is a stub — signal-cli reaction
        # API support will be wired in when the adapter registers its callbacks.
        return f"Reacted with {emoji} to {message_id} (stub — signal-cli wiring TODO)"

    @mcp.tool
    async def set_typing(conversation_id: str, enabled: bool) -> str:
        """Show or hide the typing indicator."""
        status = "started" if enabled else "stopped"
        return f"Typing {status} for {conversation_id} (stub — signal-cli wiring TODO)"

    return mcp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_channel_mcp.py -v`
Expected: PASS

- [ ] **Step 5: Write test for MCP tool registration**

```python
# Append to tests/test_channel_mcp.py

@pytest.mark.asyncio
async def test_mcp_has_expected_tools():
    """The MCP server exposes the expected Signal tools."""
    mcp = create_channel_mcp()
    tools = await mcp.get_tools()
    tool_names = {t.name for t in tools}
    assert "stage_attachment" in tool_names
    assert "send_attachment" in tool_names
    assert "react" in tool_names
    assert "set_typing" in tool_names
```

- [ ] **Step 6: Run all channel MCP tests**

Run: `.venv/bin/pytest tests/test_channel_mcp.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/adapters/signal/channel_mcp.py tests/test_channel_mcp.py
git commit -m "feat: Signal channel MCP server with attachment staging and reactions"
```

---

## Task 6: Signal Adapter

**Files:**
- Create: `src/adapters/signal/adapter.py`
- Create: `src/adapters/signal/main.py`
- Test: `tests/test_adapter.py`

The core message bridge: receives Signal messages, builds LiteLLM requests with session history and MCP tool declarations, sends responses back, handles heartbeat endpoint.

- [ ] **Step 1: Write test for message handling flow**

```python
# tests/test_adapter.py
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from adapters.signal.adapter import SignalAdapter


@pytest.fixture
def adapter_config(sample_config, tmp_sessions):
    """Create adapter with mocked dependencies."""
    from agent_platform.config import load_config
    config = load_config(sample_config)
    return config


@pytest.fixture
def mock_litellm_response():
    """A mock LiteLLM chat completion response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm your agent.",
                }
            }
        ]
    }
```

- [ ] **Step 2: Write test for adapter initialization**

```python
# Append to tests/test_adapter.py

def test_adapter_init(adapter_config, tmp_sessions):
    """Adapter initializes with config and creates session manager."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )
    assert adapter.litellm_url == "http://localhost:4000"
    assert adapter.allow_from == ["+11111111111"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_adapter.py::test_adapter_init -v`
Expected: FAIL

- [ ] **Step 4: Implement SignalAdapter**

```python
# src/adapters/signal/adapter.py
"""Signal adapter: bridges signal-cli JSON-RPC to LiteLLM with session management."""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from agent_platform.config import build_mcp_tool_declarations
from agent_platform.session import SessionManager
from adapters.signal.channel_mcp import staged_attachments

logger = logging.getLogger(__name__)


class SignalAdapter:
    def __init__(
        self,
        config: dict,
        litellm_url: str,
        session_dir: Path,
    ):
        self.config = config
        self.litellm_url = litellm_url.rstrip("/")

        channel_config = config["channels"]["signal"]
        self.signal_cli_url = channel_config["signal_cli"].rstrip("/")
        self.account = channel_config["account"]
        self.allow_from = channel_config.get("allow_from", [])

        session_config = config.get("session", {})
        max_tokens = session_config.get("max_history_tokens", 100000)
        self.session = SessionManager(store_dir=session_dir, max_history_tokens=max_tokens)

        self.mcp_tools = build_mcp_tool_declarations(config)
        self.model = "main"

        self.heartbeat_prompt = config.get("heartbeat", {}).get(
            "prompt", "Check HEARTBEAT.md"
        )

        self._http = httpx.AsyncClient(timeout=300)

    async def handle_message(self, sender: str, text: str, message_id: str | None = None) -> str | None:
        """Process an incoming message and return the agent's response."""
        if self.allow_from and sender not in self.allow_from:
            logger.warning(f"Ignoring message from unauthorized sender: {sender}")
            return None

        conversation_id = sender

        # Load session history
        history = self.session.load_truncated(contact=sender, model=self.config["model"]["model"])

        # Build the new user message
        user_msg = {"role": "user", "content": text}

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
            return f"Error: {e}"

        assistant_msg = result["choices"][0]["message"]
        assistant_text = assistant_msg.get("content", "")

        # Save the exchange to session.
        # Note: LiteLLM handles the full tool execution loop server-side and
        # returns only the final response. Intermediate tool_call/tool_result
        # messages are not available to us. We save the final text exchange.
        # Future: if LiteLLM exposes the full message chain, save all of it.
        self.session.append(sender, [user_msg, {"role": "assistant", "content": assistant_text}])

        # Drain staged attachments
        attachments = staged_attachments.drain(conversation_id)

        # Send response via signal-cli
        await self._send_signal_message(sender, assistant_text, attachments)

        # Clean up attachment files after successful delivery
        for att in attachments:
            try:
                Path(att["file_path"]).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean up attachment {att['file_path']}: {e}")

        return assistant_text

    async def handle_heartbeat(self) -> str | None:
        """Process a heartbeat: send heartbeat prompt to agent, deliver response to all contacts."""
        if not self.allow_from:
            logger.warning("No contacts configured for heartbeat delivery")
            return None

        # Use first allowed contact as the heartbeat conversation target
        contact = self.allow_from[0]
        return await self.handle_message(contact, f"[HEARTBEAT] {self.heartbeat_prompt}")


    async def _send_signal_message(
        self, recipient: str, text: str, attachments: list[dict] | None = None
    ) -> None:
        """Send a message (and optional attachments) via signal-cli REST API."""
        payload: dict[str, Any] = {
            "message": text,
            "number": self.account,
            "recipients": [recipient],
        }
        if attachments:
            payload["base64_attachments"] = []
            for att in attachments:
                file_path = Path(att["file_path"])
                if file_path.exists():
                    data = base64.b64encode(file_path.read_bytes()).decode()
                    payload["base64_attachments"].append(data)

        try:
            await self._http.post(
                f"{self.signal_cli_url}/v2/send",
                json=payload,
            )
        except Exception as e:
            logger.error(f"Failed to send Signal message: {e}")

    async def poll_messages(self) -> None:
        """Poll signal-cli for new messages and process them."""
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
            text = data_msg["message"]
            timestamp = str(data_msg.get("timestamp", ""))

            await self.handle_message(sender, text, message_id=timestamp)

    async def run(self, poll_interval: float = 1.0) -> None:
        """Main loop: poll signal-cli for messages."""
        logger.info(f"Signal adapter started. Account: {self.account}")
        while True:
            await self.poll_messages()
            await asyncio.sleep(poll_interval)

    async def close(self) -> None:
        await self._http.aclose()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_adapter.py::test_adapter_init -v`
Expected: PASS

- [ ] **Step 6: Write test for handle_message flow (mocked HTTP)**

```python
# Append to tests/test_adapter.py

@pytest.mark.asyncio
async def test_handle_message_happy_path(adapter_config, tmp_sessions, mock_litellm_response):
    """handle_message sends to LiteLLM and saves session."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )

    # Mock both HTTP calls: LiteLLM and signal-cli
    async def mock_post(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        if "chat/completions" in url:
            mock_resp.json = MagicMock(return_value=mock_litellm_response)
        else:
            mock_resp.json = MagicMock(return_value={})
        return mock_resp

    adapter._http = AsyncMock()
    adapter._http.post = AsyncMock(side_effect=mock_post)

    result = await adapter.handle_message("+11111111111", "Hello!")
    assert result == "Hello! I'm your agent."

    # Verify session was saved
    history = adapter.session.load("+11111111111")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_handle_message_unauthorized(adapter_config, tmp_sessions):
    """Messages from unauthorized senders are ignored."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )
    result = await adapter.handle_message("+19999999999", "Hello!")
    assert result is None
```

- [ ] **Step 7: Write test for heartbeat flow**

```python
# Append to tests/test_adapter.py

@pytest.mark.asyncio
async def test_handle_heartbeat(adapter_config, tmp_sessions, mock_litellm_response):
    """Heartbeat sends configured prompt to first allowed contact."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )

    async def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "chat/completions" in url:
            resp.json = MagicMock(return_value=mock_litellm_response)
            # Verify heartbeat prompt is in the messages
            payload = kwargs.get("json", {})
            last_msg = payload["messages"][-1]["content"]
            assert "[HEARTBEAT]" in last_msg
            assert "Check HEARTBEAT.md" in last_msg
        else:
            resp.json = MagicMock(return_value={})
        return resp

    adapter._http = AsyncMock()
    adapter._http.post = AsyncMock(side_effect=mock_post)

    result = await adapter.handle_heartbeat()
    assert result == "Hello! I'm your agent."

    # Verify it was saved to the first contact's session
    history = adapter.session.load("+11111111111")
    assert len(history) == 2
    assert "[HEARTBEAT]" in history[0]["content"]
```

- [ ] **Step 8: Write test for poll_messages**

```python
# Append to tests/test_adapter.py

@pytest.mark.asyncio
async def test_poll_messages_processes_envelope(adapter_config, tmp_sessions, mock_litellm_response):
    """poll_messages correctly parses signal-cli envelope format."""
    adapter = SignalAdapter(
        config=adapter_config,
        litellm_url="http://localhost:4000",
        session_dir=tmp_sessions,
    )

    signal_response = [
        {
            "envelope": {
                "source": "+11111111111",
                "dataMessage": {
                    "message": "Test from Signal",
                    "timestamp": 1234567890,
                },
            }
        }
    ]

    call_count = 0

    async def mock_request(method_or_url, url=None, **kwargs):
        nonlocal call_count
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()

        if url and "receive" in url:
            resp.json = MagicMock(return_value=signal_response)
        elif url and "chat/completions" in url:
            resp.json = MagicMock(return_value=mock_litellm_response)
        else:
            resp.json = MagicMock(return_value={})
        call_count += 1
        return resp

    adapter._http = AsyncMock()
    adapter._http.get = AsyncMock(side_effect=lambda url, **kw: mock_request(None, url, **kw))
    adapter._http.post = AsyncMock(side_effect=lambda url, **kw: mock_request(None, url, **kw))

    await adapter.poll_messages()

    # Verify the message was processed and saved
    history = adapter.session.load("+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Test from Signal"
```

- [ ] **Step 9: Run all adapter tests**

Run: `.venv/bin/pytest tests/test_adapter.py -v`
Expected: All PASS

- [ ] **Step 10: Implement main.py entrypoint**

```python
# src/adapters/signal/main.py
"""Entrypoint: starts Signal adapter + channel MCP server (with heartbeat route) in the same process."""

import asyncio
import logging
import os
from pathlib import Path

from agent_platform.config import load_config
from adapters.signal.adapter import SignalAdapter
from adapters.signal.channel_mcp import create_channel_mcp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main():
    config_path = Path(os.environ.get("AGENT_CONFIG", "agent.yaml"))
    config = load_config(config_path)

    litellm_url = os.environ.get("LITELLM_URL", "http://localhost:4000")
    session_dir = Path(config.get("session", {}).get("store", "./sessions"))
    mcp_port = int(os.environ.get("CHANNEL_MCP_PORT", "8100"))

    adapter = SignalAdapter(
        config=config,
        litellm_url=litellm_url,
        session_dir=session_dir,
    )

    # Create channel MCP server with heartbeat route on the same port
    channel_mcp = create_channel_mcp(heartbeat_handler=adapter.handle_heartbeat)

    # Run adapter polling and MCP server concurrently
    async def run_mcp():
        await channel_mcp.run_async(transport="http", host="0.0.0.0", port=mcp_port)

    try:
        await asyncio.gather(
            adapter.run(),
            run_mcp(),
        )
    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 11: Commit**

```bash
git add src/adapters/signal/adapter.py src/adapters/signal/main.py tests/test_adapter.py
git commit -m "feat: Signal adapter with message bridging, session management, and heartbeat"
```

---

## Task 7: Docker Setup

**Files:**
- Create: `src/adapters/signal/Dockerfile`
- Create: `src/adapters/signal/requirements.txt`
- Create: `docker-compose.yaml`
- Create: `agent.yaml.example`

- [ ] **Step 1: Create requirements.txt for Signal adapter**

```
fastmcp>=2.0
litellm>=1.80
httpx>=0.27
pyyaml>=6
uvicorn>=0.30
```

- [ ] **Step 2: Create Dockerfile for Signal adapter**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY src/adapters/signal/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "adapters.signal.main"]
```

- [ ] **Step 3: Create docker-compose.yaml**

```yaml
services:
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
    volumes:
      - ./workspace:/workspace
      - ./sessions:/sessions
      - ./agent.yaml:/app/agent.yaml:ro
    environment:
      - LITELLM_URL=http://litellm:4000
      - AGENT_CONFIG=/app/agent.yaml
      - CHANNEL_MCP_PORT=8100
    ports:
      - "8100:8100"
    depends_on: [litellm, signal-cli]
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
    image: python:3.12-slim
    working_dir: /app
    command: >
      bash -c "pip install fastmcp mcp-server-filesystem &&
      python -c \"
      from fastmcp import FastMCP
      from pathlib import Path
      import os

      mcp = FastMCP('workspace-fs')

      WORKSPACE = Path(os.environ.get('WORKSPACE_DIR', '/workspace'))

      @mcp.tool
      def read_file(path: str) -> str:
          target = (WORKSPACE / path).resolve()
          if not str(target).startswith(str(WORKSPACE.resolve())):
              return 'Error: path traversal not allowed'
          return target.read_text()

      @mcp.tool
      def write_file(path: str, content: str) -> str:
          target = (WORKSPACE / path).resolve()
          if not str(target).startswith(str(WORKSPACE.resolve())):
              return 'Error: path traversal not allowed'
          target.parent.mkdir(parents=True, exist_ok=True)
          target.write_text(content)
          return f'Wrote {len(content)} bytes to {path}'

      @mcp.tool
      def list_directory(path: str = '.') -> list[str]:
          target = (WORKSPACE / path).resolve()
          if not str(target).startswith(str(WORKSPACE.resolve())):
              return ['Error: path traversal not allowed']
          return [str(p.relative_to(WORKSPACE)) for p in sorted(target.iterdir())]

      @mcp.tool
      def search_files(pattern: str, path: str = '.') -> list[str]:
          target = (WORKSPACE / path).resolve()
          if not str(target).startswith(str(WORKSPACE.resolve())):
              return ['Error: path traversal not allowed']
          return [str(p.relative_to(WORKSPACE)) for p in sorted(target.rglob(pattern))]

      mcp.run(transport='http', host='0.0.0.0', port=8000)
      \"
      "
    volumes:
      - ./workspace:/workspace
    environment:
      - WORKSPACE_DIR=/workspace
    restart: unless-stopped

volumes:
  signal-data:
```

**Note:** The workspace-fs-mcp uses an inline Python script for now. Once stable, this should be extracted to its own file. The path traversal guard prevents the agent from reading outside the workspace.

- [ ] **Step 4: Create agent.yaml.example**

```yaml
# agent.yaml — Agent Platform configuration
# Copy to agent.yaml and fill in your values.

model:
  provider: anthropic
  model: claude-opus-4-6
  api_key: ${ANTHROPIC_API_KEY}

# Agent workspace directory. All *.md files in the root are auto-injected
# as the system prompt. The agent can read/write files here via MCP.
workspace: ./workspace

# Optional: explicit ordering for system prompt files.
# If omitted, all *.md files in workspace root are discovered alphabetically.
# system_prompt:
#   - ./workspace/SOUL.md
#   - ./workspace/USER.md

# User-provided MCP servers. Each gets exposed to the agent as tools.
mcp_servers: {}
  # home_assistant:
  #   url: http://ha.home.arpa/mcp
  #   headers:
  #     Authorization: "Bearer ${HA_TOKEN}"
  # custom_tool:
  #   command: python3
  #   args: ["./mcp-servers/custom.py"]

channels:
  signal:
    account: "+1YOURACCOUNT"
    signal_cli: http://signal-cli:8080
    allow_from:
      - "+1YOURPHONE"

session:
  store: ./sessions/
  max_history_tokens: 100000

heartbeat:
  interval: 20m
  prompt: "Check HEARTBEAT.md and decide if anything needs attention."
```

- [ ] **Step 5: Commit**

```bash
git add src/adapters/signal/Dockerfile src/adapters/signal/requirements.txt \
  docker-compose.yaml agent.yaml.example
git commit -m "feat: Docker setup with compose orchestration"
```

---

## Task 8: Config Loader CLI + Generated Config

**Files:**
- Modify: `src/agent_platform/config.py` (add CLI entrypoint)

Add a CLI command so the user can generate LiteLLM config before running docker-compose.

- [ ] **Step 1: Add CLI entrypoint to config.py**

Append to `src/agent_platform/config.py`:

```python
def cli():
    """CLI entrypoint: generate LiteLLM config from agent.yaml."""
    import sys

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agent.yaml")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("generated/litellm-config.yaml")

    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    litellm_config = generate_litellm_config(config)

    # Add system prompt callback reference (module.instance_name)
    # LiteLLM imports this from PYTHONPATH — the callbacks dir is mounted in docker-compose
    litellm_config["litellm_settings"] = {
        "callbacks": "system_prompt.proxy_handler_instance",
    }

    write_litellm_config(litellm_config, output_path)
    print(f"Generated LiteLLM config at {output_path}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Add script entry to pyproject.toml**

Add under `[project.scripts]`:

```toml
[project.scripts]
agent-platform-config = "agent_platform.config:cli"
```

- [ ] **Step 3: Test CLI manually**

```bash
cd /home/max/projects/personal/AgentPlatform
cp agent.yaml.example agent.yaml
# Edit agent.yaml with real values or leave as-is for test
ANTHROPIC_API_KEY=test python -m agent_platform.config agent.yaml generated/litellm-config.yaml
cat generated/litellm-config.yaml
```

Expected: Valid YAML with model_list, mcp_servers, litellm_settings.

- [ ] **Step 4: Add heartbeat cron docs to agent.yaml.example**

Append to `agent.yaml.example`:

```yaml
# Heartbeat: the adapter exposes POST /heartbeat on port 5000.
# Set up a cron job or systemd timer to trigger it:
#
#   */20 * * * * curl -s -X POST http://localhost:8100/heartbeat
#
# In docker-compose, you can use a sidecar or host cron.
# The agent's personality files (HEARTBEAT.md) define behavior.
```

- [ ] **Step 5: Commit**

```bash
git add src/agent_platform/config.py pyproject.toml
git commit -m "feat: config loader CLI for generating LiteLLM config"
```

---

## Task 9: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

Verify the full data flow with mocked external services (signal-cli, LiteLLM).

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration smoke test: verify the full message flow with mocked externals."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from agent_platform.config import load_config, generate_litellm_config, build_mcp_tool_declarations
from agent_platform.session import SessionManager
from agent_platform.callbacks.system_prompt import build_system_message
from adapters.signal.adapter import SignalAdapter
from adapters.signal.channel_mcp import create_channel_mcp, staged_attachments


@pytest.mark.asyncio
async def test_full_message_flow(tmp_path):
    """End-to-end: config → system prompt → session → adapter → response."""
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
    assert "signal_channel" in litellm_config["mcp_servers"]

    # 5. Verify MCP tool declarations
    tools = build_mcp_tool_declarations(config)
    assert len(tools) == 2  # workspace_fs + signal_channel

    # 6. Create adapter and handle a message
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    adapter = SignalAdapter(
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
        if "chat/completions" in url:
            resp.json = MagicMock(return_value=mock_response)
            # Verify the request includes MCP tools
            payload = kwargs.get("json", {})
            assert "tools" in payload
            assert len(payload["tools"]) == 2
        else:
            resp.json = MagicMock(return_value={})
        return resp

    adapter._http = AsyncMock()
    adapter._http.post = AsyncMock(side_effect=mock_post)

    result = await adapter.handle_message("+11111111111", "Hi there!")
    assert result == "I'm here to help!"

    # 7. Verify session persistence
    session = SessionManager(store_dir=sessions_dir, max_history_tokens=100000)
    history = session.load("+11111111111")
    assert len(history) == 2
    assert history[0]["content"] == "Hi there!"
    assert history[1]["content"] == "I'm here to help!"

    # 8. Verify channel MCP tools exist
    mcp = create_channel_mcp()
    mcp_tools = await mcp.get_tools()
    tool_names = {t.name for t in mcp_tools}
    assert "stage_attachment" in tool_names
    assert "send_attachment" in tool_names
```

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke test for full message flow"
```

---

## Task 10: Documentation and Final Cleanup

**Files:**
- Create: `workspace/HEARTBEAT.md` (example)
- Verify: all files exist at expected paths

- [ ] **Step 1: Create example workspace files**

Create `workspace/.gitkeep` (the actual workspace is user-created, but we need the directory in the repo structure).

Create `workspace-example/HEARTBEAT.md`:

```markdown
# Heartbeat

When you receive a heartbeat, check:
1. Read today's date and check if there are any calendar events
2. Check memory for anything you planned to follow up on
3. If nothing needs attention, respond with HEARTBEAT_OK
```

Create `workspace-example/SOUL.md`:

```markdown
# Soul

Define your agent's personality here. This file is auto-injected into the system prompt.
```

- [ ] **Step 2: Create sessions/.gitkeep and generated/.gitkeep**

```bash
mkdir -p sessions generated workspace-example
touch sessions/.gitkeep generated/.gitkeep
```

- [ ] **Step 3: Final full test run**

Run: `.venv/bin/pytest -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit everything**

```bash
git add workspace-example/ sessions/.gitkeep generated/.gitkeep
git commit -m "docs: example workspace files and directory structure"
```

---

## Summary

| Task | Component | New Lines (approx) | Tests |
|------|-----------|-------------------|-------|
| 1 | Scaffolding | ~50 | conftest fixtures |
| 2 | Config Loader | ~90 | 5 tests |
| 3 | Session Manager | ~80 | 5 tests |
| 4 | System Prompt Callback | ~50 | 5 tests |
| 5 | Channel MCP Server | ~70 | 3 tests |
| 6 | Signal Adapter | ~180 | 5 tests (incl. heartbeat, poll) |
| 7 | Docker Setup | ~100 (yaml/dockerfile) | — |
| 8 | Config CLI + Heartbeat docs | ~30 | manual |
| 9 | Integration Test | — | 1 integration test |
| 10 | Docs + Cleanup | ~20 | — |
| **Total** | | **~670 lines** | **24 tests** |
