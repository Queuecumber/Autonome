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

    litellm_params: dict[str, Any] = {
        "model": f"{provider}/{model_name}",
        "api_key": api_key,
    }
    if model.get("api_base"):
        litellm_params["api_base"] = model["api_base"]
    if model.get("extra_headers"):
        litellm_params["extra_headers"] = model["extra_headers"]

    litellm_config: dict[str, Any] = {
        "model_list": [
            {
                "model_name": "main",
                "litellm_params": litellm_params,
            }
        ],
        "mcp_servers": {
            "workspace_fs": {
                "url": f"http://{fs_mcp_host}:{fs_mcp_port}/mcp",
                "transport": "http",
                "allow_all_keys": True,
            },
            "memory": {
                "url": f"http://{memory_mcp_host}:{memory_mcp_port}/mcp",
                "transport": "http",
                "allow_all_keys": True,
            },
            "signal": {
                "url": f"http://{adapter_host}:{adapter_mcp_port}/mcp",
                "transport": "http",
                "allow_all_keys": True,
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
        entry["allow_all_keys"] = True
        litellm_config["mcp_servers"][name] = entry

    return litellm_config


def build_mcp_tool_declarations(config: dict) -> list[dict]:
    """Build the MCP tool declarations list for chat completion requests."""
    def _mcp_tool(label: str) -> dict:
        return {
            "type": "mcp",
            "server_label": label,
            "server_url": "litellm_proxy",
            "require_approval": "never",
        }

    tools = [
        _mcp_tool("workspace_fs"),
        _mcp_tool("memory"),
        _mcp_tool("signal"),
    ]
    for name in config.get("mcp_servers", {}):
        tools.append(_mcp_tool(name))
    return tools


def write_litellm_config(litellm_config: dict, output_path: Path) -> None:
    """Write generated LiteLLM config to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"# generated/litellm-config.yaml (DO NOT EDIT — generated from agent.yaml)\n\n"
        + yaml.dump(litellm_config, default_flow_style=False, sort_keys=False)
    )


def _convert_env_refs_to_litellm(obj: Any) -> Any:
    """Convert ${VAR} references to LiteLLM's os.environ/VAR format."""
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", r"os.environ/\1", obj)
    if isinstance(obj, dict):
        return {k: _convert_env_refs_to_litellm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_env_refs_to_litellm(item) for item in obj]
    return obj


def load_config_raw(config_path: Path) -> dict:
    """Load agent.yaml WITHOUT env var substitution (for config generation)."""
    raw = Path(config_path).read_text()
    return yaml.safe_load(raw)


def cli():
    """CLI entrypoint: generate LiteLLM config from agent.yaml."""
    import sys

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agent.yaml")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("generated/litellm-config.yaml")

    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load raw config (no env var substitution) so we can convert
    # ${VAR} to LiteLLM's os.environ/VAR format in the generated output
    config = load_config_raw(config_path)
    litellm_config = generate_litellm_config(config)
    litellm_config = _convert_env_refs_to_litellm(litellm_config)

    write_litellm_config(litellm_config, output_path)
    print(f"Generated LiteLLM config at {output_path}")


if __name__ == "__main__":
    cli()
