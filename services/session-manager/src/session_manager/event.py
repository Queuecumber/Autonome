"""Canonical event schema for the session manager.

Adapters (signal, matrix, time, etc.) produce events that conform to this
shape and push them to session-manager as MCP notifications. Events without a session_id
route to the default session — the agent has one unified history. An
event can target a different session by setting session_id explicitly.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


Energy = Literal["active", "passive"]

DEFAULT_SESSION_ID = "main"


@dataclass
class Event:
    """An inbound event.

    Attributes:
        session_id: Routing key. Empty/omitted routes to the default
            session.
        source: Where the event came from ("matrix", "signal", "time", etc.).
            Metadata only, not used for routing.
        event_type: What kind of event this is ("message", "cron", "continuity",
            etc.). Surfaced to the agent as the "event" field in the developer
            context message.
        text: The event content. May be plain text or JSON-encoded structured
            content (e.g. reactions).
        energy: "active" cancels in-progress generation and processes immediately.
            "passive" queues if the session is busy, processes if idle.
        metadata: Source-specific extras (sender, room_id, emoji, schedule_id, etc.).
            Surfaced to the agent in the developer context message.
    """
    session_id: str = DEFAULT_SESSION_ID
    source: str = "unknown"
    event_type: str = "message"
    text: str = ""
    energy: Energy = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Build an Event from a JSON payload. Raises ValueError on invalid input."""
        energy = data.get("energy", "active")
        if energy not in ("active", "passive"):
            raise ValueError(f"Event energy must be 'active' or 'passive', got {energy!r}")
        return cls(
            session_id=data.get("session_id") or DEFAULT_SESSION_ID,
            source=data.get("source", "unknown"),
            event_type=data.get("event_type", "message"),
            text=data.get("text", ""),
            energy=energy,
            metadata=data.get("metadata") or {},
        )
