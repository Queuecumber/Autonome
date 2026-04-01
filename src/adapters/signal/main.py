"""Signal adapter: runs inbound listener + outbound MCP server over a shared SignalClient."""

import asyncio
import logging
import os

from adapters.signal.model import SignalClient
from adapters.signal import mcp_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    signal_cli_url = os.environ.get("SIGNAL_CLI_URL", "http://localhost:8080")
    session_manager_url = os.environ.get("SESSION_MANAGER_URL", "http://localhost:5000")
    account = os.environ.get("SIGNAL_ACCOUNT", "")
    allow_from = os.environ.get("ALLOW_FROM", "").split(",") if os.environ.get("ALLOW_FROM") else []
    mcp_port = int(os.environ.get("CHANNEL_MCP_PORT", "8100"))

    client = SignalClient(
        signal_cli_url=signal_cli_url,
        account=account,
        allow_from=allow_from,
    )

    mcp_server.init(client, session_manager_url)

    try:
        await asyncio.gather(
            mcp_server.run_inbound(),
            mcp_server.run_mcp(port=mcp_port),
        )
    finally:
        await mcp_server.close()
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
