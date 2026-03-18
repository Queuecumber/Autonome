"""Signal adapter: runs inbound listener + outbound MCP server."""

import asyncio
import logging
import os

from adapters.signal.inbound import SignalInbound
from adapters.signal.outbound_mcp import SignalSender, create_signal_mcp

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

    inbound = SignalInbound(
        signal_cli_url=signal_cli_url,
        session_manager_url=session_manager_url,
        account=account,
        allow_from=allow_from,
    )

    sender = SignalSender(signal_cli_url=signal_cli_url, account=account)
    mcp = create_signal_mcp(sender)

    async def run_mcp():
        await mcp.run_async(transport="http", host="0.0.0.0", port=mcp_port)

    try:
        await asyncio.gather(
            inbound.run(),
            run_mcp(),
        )
    finally:
        await inbound.close()
        await sender.close()


if __name__ == "__main__":
    asyncio.run(main())
