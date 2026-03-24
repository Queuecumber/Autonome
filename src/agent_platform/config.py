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
    memory_mcp_host: str = "memory-mcp",
    memory_mcp_port: int = 8001,
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
            "memory": {
                "url": f"http://{memory_mcp_host}:{memory_mcp_port}/mcp",
                "transport": "http",
            },
            "signal": {
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
        {"type": "mcp", "server_label": "memory", "require_approval": "never"},
        {"type": "mcp", "server_label": "signal", "require_approval": "never"},
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
