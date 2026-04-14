"""Entrypoint for the session manager service."""

import asyncio
import logging
import os
from pathlib import Path

import uvicorn
import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

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

    orchestrator = SessionOrchestrator(
        config=config,
        session_dir=session_dir,
    )

    # Connect to MCP servers (retry until available)
    mcp_urls = config.get("mcp_servers", {})
    max_retries = 30
    for attempt in range(max_retries):
        try:
            await orchestrator.connect_mcp_servers(mcp_urls)
            if orchestrator.openai_tools:
                break
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning(f"MCP connection attempt {attempt + 1}/{max_retries} failed: {e}")
        await asyncio.sleep(2)

    if not orchestrator.openai_tools:
        logger.error("No MCP tools discovered after retries. Starting anyway.")

    async def event_endpoint(request: Request) -> Response:
        body = await request.json()
        asyncio.create_task(orchestrator.handle_event(body))
        return Response(status_code=202)

    app = Starlette(routes=[Route("/event", event_endpoint, methods=["POST"])])
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info"))
    await server.serve()


def main():
    asyncio.run(startup())


if __name__ == "__main__":
    main()
