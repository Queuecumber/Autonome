"""Session manager: central orchestrator that receives events and drives LLM calls.

Sends requests to LiteLLM proxy /v1/responses endpoint which handles model routing,
MCP tool discovery, tool execution loop (with require_approval=never), and returns
the final response.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_platform.config import build_mcp_tool_declarations
from agent_platform.session import SessionManager

logger = logging.getLogger(__name__)


class SessionOrchestrator:
    """Receives events from adapters, maintains session history, calls LiteLLM proxy."""

    def __init__(self, config: dict, litellm_url: str, session_dir: Path):
        self.config = config
        self.litellm_url = litellm_url.rstrip("/")

        session_config = config.get("session", {})
        max_tokens = session_config.get("max_history_tokens", 100000)
        self.session = SessionManager(store_dir=session_dir, max_history_tokens=max_tokens)

        self.mcp_tools = build_mcp_tool_declarations(config)
        self.workspace_dir = Path(config.get("workspace", "./workspace"))
        self.model = "main"

        self.heartbeat_prompt = config.get("heartbeat", {}).get(
            "prompt", "Check HEARTBEAT.md"
        )

        # Heartbeat routes to the primary Signal contact's session
        channels = config.get("channels", {})
        signal_config = channels.get("signal", {})
        allow_from = signal_config.get("allow_from", [])
        self.heartbeat_source = "signal"
        self.heartbeat_session_id = allow_from[0] if allow_from else "system"

        # Per-session locks for concurrency safety
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

        self._http = httpx.AsyncClient(timeout=300)

    def _get_lock(self, channel: str, session_id: str) -> asyncio.Lock:
        key = (channel, session_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _build_system_prompt(self) -> str:
        """Read AGENTS.md from workspace for system prompt. Agent bootstraps the rest itself."""
        agents_path = self.workspace_dir / "AGENTS.md"
        if agents_path.exists():
            return agents_path.read_text()
        return ""

    async def handle_event(self, event: dict[str, Any]) -> str | None:
        """Process an inbound event from any adapter."""
        source = event["source"]
        session_id = event["session_id"]
        text = event.get("text", "")
        metadata = event.get("metadata", {})

        async with self._get_lock(source, session_id):
            # Load session history, filter non-conversational entries, sanitize
            raw_history = self.session.load_truncated(
                channel=source,
                session_id=session_id,
                model=self.config["model"]["model"],
            )
            history = []
            for msg in raw_history:
                if msg.get("role") not in ("user", "assistant", "tool", "system"):
                    continue
                if not msg.get("content"):
                    if msg.get("tool_calls"):
                        msg["content"] = "(calling tools)"
                    elif msg.get("role") == "assistant":
                        msg["content"] = "(no text response)"
                for key in ("reasoning_content", "thinking_blocks", "provider_specific_fields", "function_call"):
                    msg.pop(key, None)
                history.append(msg)

            # Build user message with metadata context
            context_prefix = f"[{source}]"
            if metadata:
                context_prefix = f"[{source} | {json.dumps(metadata)}]"
            enriched_content = f"{context_prefix} {text}"
            stored_msg = {"role": "user", "content": text}

            # Build system prompt
            system_content = self._build_system_prompt()

            # Build Responses API input
            input_items = []

            # System prompt as first input
            if system_content:
                input_items.append({
                    "role": "system",
                    "content": system_content,
                })

            # History
            for msg in history:
                input_items.append(msg)

            # New user message
            input_items.append({
                "role": "user",
                "content": enriched_content,
            })

            # Build request payload for /v1/responses
            payload: dict[str, Any] = {
                "model": self.model,
                "input": input_items,
                "tools": self.mcp_tools,
            }

            logger.info(f"Calling LiteLLM proxy /v1/responses: {len(input_items)} input items, {len(self.mcp_tools)} MCP tools")

            try:
                resp = await self._http.post(
                    f"{self.litellm_url}/v1/responses",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                logger.error(f"LiteLLM proxy request failed: {e}")
                return None

            # Extract text from response output
            assistant_text = ""
            output = result.get("output", [])
            for item in output:
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            assistant_text += content.get("text", "")

            # Save exchange to session
            all_messages = [stored_msg]
            if assistant_text:
                all_messages.append({"role": "assistant", "content": assistant_text})
            self.session.append(source, session_id, all_messages)

            logger.info(f"Response: {assistant_text[:200]}")
            return assistant_text

    async def handle_heartbeat(self) -> str | None:
        """Process a heartbeat routed to the primary contact's session."""
        event = {
            "source": self.heartbeat_source,
            "session_id": self.heartbeat_session_id,
            "text": f"[HEARTBEAT] {self.heartbeat_prompt}",
            "metadata": {},
        }
        return await self.handle_event(event)

    async def close(self) -> None:
        await self._http.aclose()


def create_app(orchestrator: SessionOrchestrator) -> Starlette:
    """Create the HTTP application for the session manager."""

    async def event_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        result = await orchestrator.handle_event(body)
        if result is None:
            return JSONResponse(
                {"status": "error", "response": None},
                status_code=502,
            )
        return JSONResponse({"status": "ok", "response": result})

    async def heartbeat_endpoint(request: Request) -> JSONResponse:
        result = await orchestrator.handle_heartbeat()
        if result is None:
            return JSONResponse(
                {"status": "error", "response": None},
                status_code=502,
            )
        return JSONResponse({"status": "ok", "response": result})

    return Starlette(routes=[
        Route("/event", event_endpoint, methods=["POST"]),
        Route("/heartbeat", heartbeat_endpoint, methods=["POST"]),
    ])
