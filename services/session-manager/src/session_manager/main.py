"""Entrypoint for the session manager service."""

import asyncio
import logging
import os
from pathlib import Path

import uvicorn
import yaml

from session_manager.server import SessionOrchestrator, create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def startup():
    config_path = Path(os.environ.get("AGENT_CONFIG", "agent.yaml"))
    config = yaml.safe_load(config_path.read_text())

    session_dir = Path(config.get("session", {}).get("store", "./sessions"))
    port = int(os.environ.get("SESSION_MANAGER_PORT", "5000"))

    orchestrator = SessionOrchestrator(
        config=config,
        session_dir=session_dir,
    )

    # MCP server URLs from config
    mcp_urls = config.get("mcp_servers", {})

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
