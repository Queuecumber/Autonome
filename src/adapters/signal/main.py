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
