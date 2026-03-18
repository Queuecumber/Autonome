import os
from pathlib import Path

import yaml

from agent_platform.config import load_config, generate_litellm_config, build_mcp_tool_declarations


def test_load_config_basic(sample_config):
    """Config loads and has expected top-level keys."""
    config = load_config(sample_config)
    assert config["model"]["provider"] == "anthropic"
    assert config["model"]["model"] == "claude-opus-4-6"
    assert "channels" in config
    assert "session" in config
    assert "heartbeat" in config


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


def test_generate_litellm_config_basic(sample_config):
    """Generated LiteLLM config has model list and implicit MCP servers."""
    config = load_config(sample_config)
    litellm = generate_litellm_config(config)

    assert len(litellm["model_list"]) == 1
    assert litellm["model_list"][0]["model_name"] == "main"
    assert "workspace_fs" in litellm["mcp_servers"]
    assert "signal" in litellm["mcp_servers"]


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
    assert "signal" in labels
    assert all(t["require_approval"] == "never" for t in tools)
