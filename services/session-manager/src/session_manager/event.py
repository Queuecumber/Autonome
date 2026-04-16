"""Canonical event schema for the session manager.

Adapters (signal, matrix, time, etc.) produce events that conform to this
shape and POST them to the /event endpoint. The session manager validates
on receipt and routes to the session identified by session_id.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


Energy = Literal["active", "passive"]


@dataclass
class Event:
    """An inbound event targeting a specific session.

    Attributes:
        session_id: Routing key — the session to append this event to.
            Adapters should namespace their session_ids (e.g. "matrix:!roomid",
            "signal:+1234") to avoid collisions.
        source: Where the event came from ("matrix", "signal", "time", etc.).
            Metadata only, not used for routing.
        text: The event content. May be plain text or JSON-encoded structured
            content (e.g. reactions).
        energy: "active" cancels in-progress generation and processes immediately.
            "passive" queues if the session is busy, processes if idle.
        metadata: Source-specific extras (sender, room_id, emoji, schedule_id, etc.).
            Surfaced to the agent in the developer context message.
    """
    session_id: str
    source: str = "unknown"
    text: str = ""
    energy: Energy = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Build an Event from a JSON payload. Raises ValueError on invalid input."""
        if "session_id" not in data:
            raise ValueError("Event missing required field: session_id")
        energy = data.get("energy", "active")
        if energy not in ("active", "passive"):
            raise ValueError(f"Event energy must be 'active' or 'passive', got {energy!r}")
        return cls(
            session_id=data["session_id"],
            source=data.get("source", "unknown"),
            text=data.get("text", ""),
            energy=energy,
            metadata=data.get("metadata") or {},
        )
