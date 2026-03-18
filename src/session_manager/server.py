"""Session manager: central orchestrator that receives events and drives LLM calls."""

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
    """Receives events from adapters, maintains session history, drives LLM calls."""

    def __init__(self, config: dict, litellm_url: str, session_dir: Path):
        self.config = config
        self.litellm_url = litellm_url.rstrip("/")
        self.model = "main"

        session_config = config.get("session", {})
        max_tokens = session_config.get("max_history_tokens", 100000)
        self.session = SessionManager(store_dir=session_dir, max_history_tokens=max_tokens)

        self.mcp_tools = build_mcp_tool_declarations(config)

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
        """Get or create a lock for a (channel, session_id) pair."""
        key = (channel, session_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def handle_event(self, event: dict[str, Any]) -> str | None:
        """Process an inbound event from any adapter.

        Event format:
            {
                "source": "signal",
                "session_id": "+16092409191",
                "text": "hey what's up",
                "metadata": {"message_id": "ts_123", "sender": "+16092409191"}
            }
        """
        source = event["source"]
        session_id = event["session_id"]
        text = event.get("text", "")
        metadata = event.get("metadata", {})

        async with self._get_lock(source, session_id):
            # Load session history
            history = self.session.load_truncated(
                channel=source,
                session_id=session_id,
                model=self.config["model"]["model"],
            )

            # Build user message — enriched with metadata for the LLM
            context_prefix = f"[{source}]"
            if metadata:
                context_prefix = f"[{source} | {json.dumps(metadata)}]"
            enriched_content = f"{context_prefix} {text}"
            enriched_msg = {"role": "user", "content": enriched_content}

            # Store clean text in history (no metadata noise on replay)
            stored_msg = {"role": "user", "content": text}

            # Build chat completion request
            messages = history + [enriched_msg]
            payload = {
                "model": self.model,
                "messages": messages,
                "tools": self.mcp_tools,
            }

            # Call LiteLLM
            try:
                resp = await self._http.post(
                    f"{self.litellm_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                logger.error(f"LiteLLM request failed: {e}")
                return None

            assistant_msg = result["choices"][0]["message"]
            assistant_text = assistant_msg.get("content", "")

            # Save clean exchange to session history
            self.session.append(source, session_id, [
                stored_msg,
                {"role": "assistant", "content": assistant_text},
            ])

            return assistant_text

    async def handle_heartbeat(self) -> str | None:
        """Process a heartbeat by generating an event routed to the primary contact's session."""
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
