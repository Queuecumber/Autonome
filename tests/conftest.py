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
