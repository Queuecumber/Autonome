"""Entrypoint for the session manager service."""

import logging
import os
from pathlib import Path

import uvicorn

from agent_platform.config import load_config
from session_manager.server import SessionOrchestrator, create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def main():
    config_path = Path(os.environ.get("AGENT_CONFIG", "agent.yaml"))
    config = load_config(config_path)

    litellm_url = os.environ.get("LITELLM_URL", "http://localhost:4000")
    session_dir = Path(config.get("session", {}).get("store", "./sessions"))
    port = int(os.environ.get("SESSION_MANAGER_PORT", "5000"))

    orchestrator = SessionOrchestrator(
        config=config,
        litellm_url=litellm_url,
        session_dir=session_dir,
    )

    app = create_app(orchestrator)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
