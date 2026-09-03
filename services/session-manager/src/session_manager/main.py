"""Entrypoint for the session manager service."""

import asyncio
import logging
import os
from pathlib import Path

import uvicorn
import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from session_manager import platform_mcp
from session_manager.event import Event
from session_manager.orchestrator import SessionOrchestrator

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
    platform_mcp_port = int(os.environ.get("PLATFORM_MCP_PORT", "5001"))

    orchestrator = SessionOrchestrator(
        config=config,
        session_dir=session_dir,
    )

    platform_mcp.binary_store = orchestrator.binaries
    platform_mcp.orchestrator = orchestrator
    asyncio.create_task(
        platform_mcp.mcp.run_async(transport="http", host="0.0.0.0", port=platform_mcp_port),
        name="platform-mcp",
    )
    # Give the MCP server a moment to bind before the orchestrator tries
    # to connect to it.
    await asyncio.sleep(0.5)

    # Connect to MCP servers (retry until available)
    mcp_servers = dict(config.get("mcp_servers", {}) or {})
    mcp_servers.setdefault("session", f"http://localhost:{platform_mcp_port}/mcp")
    max_retries = 30
    for attempt in range(max_retries):
        try:
            await orchestrator.connect_mcp_servers(mcp_servers)
            if orchestrator.openai_tools:
                break
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning("MCP connection attempt %d/%d failed: %s", attempt + 1, max_retries, e)
        await asyncio.sleep(2)

    if not orchestrator.openai_tools:
        logger.error("No MCP tools discovered after retries. Starting anyway.")

    # Events now arrive over MCP, pushed by the adapters as notifications
    # (see MCPConnection._on_log). The HTTP listener stays because the
    # deployment's readiness and liveness probes are tcpSocket checks
    # against this port.
    async def health(request: Request) -> Response:
        return Response(status_code=200)

    app = Starlette(routes=[Route("/health", health, methods=["GET"])])
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info"))

    asyncio.create_task(orchestrator.run_binary_gc())

    await server.serve()


def main():
    asyncio.run(startup())


if __name__ == "__main__":
    main()
