"""Entrypoint for the session manager service."""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import uvicorn
import yaml

from session_manager.server import SessionOrchestrator, create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load agent.yaml with ${VAR} environment variable substitution."""
    def _substitute(obj: Any) -> Any:
        if isinstance(obj, str):
            return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
        if isinstance(obj, dict):
            return {k: _substitute(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_substitute(item) for item in obj]
        return obj

    config = yaml.safe_load(Path(config_path).read_text())
    return _substitute(config)


async def startup():
    config_path = Path(os.environ.get("AGENT_CONFIG", "agent.yaml"))
    config = load_config(config_path)

    session_dir = Path(config.get("session", {}).get("store", "./sessions"))
    port = int(os.environ.get("SESSION_MANAGER_PORT", "5000"))

    orchestrator = SessionOrchestrator(
        config=config,
        session_dir=session_dir,
    )

    # Build MCP server URLs
    mcp_urls = {}
    mcp_urls["workspace_fs"] = os.environ.get("WORKSPACE_FS_MCP_URL", "http://workspace-fs-mcp:8000/mcp")
    mcp_urls["memory"] = os.environ.get("MEMORY_MCP_URL", "http://memory-mcp:8001/mcp")
    mcp_urls["signal"] = os.environ.get("SIGNAL_MCP_URL", "http://signal-adapter:8100/mcp")

    for name, server_config in config.get("mcp_servers", {}).items():
        if "url" in server_config:
            mcp_urls[name] = server_config["url"]

    # Connect to MCP servers (retry until available)
    max_retries = 30
    for attempt in range(max_retries):
        try:
            await orchestrator.connect_mcp_servers(mcp_urls)
            if orchestrator.openai_tools:
                break
        except Exception as e:
            logger.warning(f"MCP connection attempt {attempt + 1}/{max_retries} failed: {e}")
        await asyncio.sleep(2)

    if not orchestrator.openai_tools:
        logger.error("No MCP tools discovered after retries. Starting anyway.")

    app = create_app(orchestrator)
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info"))
    await server.serve()


def main():
    asyncio.run(startup())


if __name__ == "__main__":
    main()
