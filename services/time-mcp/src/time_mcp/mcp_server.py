"""Time adapter — cron-based scheduled events + continuity wakeup.

Events fire to the session manager just like other adapters. The continuity
cron is always registered (configurable interval and message) to give the
agent regular opportunities to check in.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from croniter import croniter
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

session_manager_url: str
_http: httpx.AsyncClient
_store_path: Path
_schedules: dict[str, "Schedule"] = {}
_fire_event = asyncio.Event()

mcp = FastMCP("time", instructions=(
    "Time and scheduling. Use get_current_time for wall clock. "
    "Use schedule_cron to set a recurring wakeup (cron syntax), or "
    "cancel_schedule to remove one. Scheduled events arrive with "
    "source='time'. A 'continuity' schedule runs at a configured interval "
    "to give you regular check-ins."
))


@dataclass
class Schedule:
    id: str
    cron: str
    message: str
    session_id: str
    label: str | None = None
    energy: str = "passive"
    next_fire: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "cron": self.cron, "message": self.message,
            "session_id": self.session_id,
            "label": self.label, "energy": self.energy,
        }

    def compute_next(self, base: datetime | None = None) -> None:
        base = base or datetime.now().astimezone()
        self.next_fire = croniter(self.cron, base).get_next(float)


def _save() -> None:
    _store_path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() for s in _schedules.values() if s.id != "continuity"]
    _store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load() -> None:
    if not _store_path.exists():
        return
    for entry in json.loads(_store_path.read_text()):
        sched = Schedule(**entry)
        sched.compute_next()
        _schedules[sched.id] = sched
    logger.info(f"Loaded {len(_schedules)} schedules")


# ── Tools ────────────────────────────────────────────────

@mcp.tool
def get_current_time() -> str:
    """Return the current wall-clock time with timezone and weekday."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


@mcp.tool
def schedule_cron(
    cron: str, message: str, session_id: str,
    label: str | None = None, energy: str = "passive",
) -> str:
    """Schedule a recurring wakeup. cron is a standard cron expression
    (e.g. '*/20 * * * *' for every 20 minutes). The message is delivered
    to the given session_id when the schedule fires.

    energy is "active" (interrupts current generation) or "passive" (queues
    if busy). Most scheduled events should be passive — use active only when
    the schedule genuinely needs immediate attention.

    Returns the schedule id."""
    if not croniter.is_valid(cron):
        raise ValueError(f"Invalid cron expression: {cron}")
    if energy not in ("active", "passive"):
        raise ValueError(f"energy must be 'active' or 'passive', got {energy!r}")
    sched = Schedule(
        id=str(uuid.uuid4())[:8],
        cron=cron, message=message, session_id=session_id,
        label=label, energy=energy,
    )
    sched.compute_next()
    _schedules[sched.id] = sched
    _save()
    _fire_event.set()
    return sched.id


@mcp.tool
def list_schedules() -> list[dict]:
    """List all active schedules with their next fire time."""
    out = []
    for s in _schedules.values():
        out.append({
            **s.to_dict(),
            "next_fire": datetime.fromtimestamp(s.next_fire).astimezone().isoformat() if s.next_fire else None,
        })
    return out


@mcp.tool
def cancel_schedule(schedule_id: str) -> str:
    """Cancel a schedule by id. The continuity schedule cannot be cancelled."""
    if schedule_id == "continuity":
        raise ValueError("Continuity schedule cannot be cancelled")
    if schedule_id not in _schedules:
        raise ValueError(f"No schedule with id {schedule_id}")
    del _schedules[schedule_id]
    _save()
    return f"Cancelled {schedule_id}"


# ── Scheduler loop ───────────────────────────────────────

async def _scheduler() -> None:
    """Fire schedules as they come due."""
    while True:
        now = datetime.now().astimezone().timestamp()

        # Find next schedule to fire
        due = [s for s in _schedules.values() if s.next_fire and s.next_fire <= now]
        for sched in due:
            await _fire(sched)
            sched.compute_next()

        # Sleep until next fire or until _fire_event is signaled
        pending = [s.next_fire for s in _schedules.values() if s.next_fire]
        if pending:
            sleep_until = min(pending)
            sleep_for = max(1.0, sleep_until - datetime.now().astimezone().timestamp())
        else:
            sleep_for = 60.0

        try:
            await asyncio.wait_for(_fire_event.wait(), timeout=sleep_for)
        except asyncio.TimeoutError:
            pass
        _fire_event.clear()


async def _fire(sched: Schedule) -> None:
    """POST a scheduled event to the session manager."""
    event = {
        "source": "time",
        "session_id": sched.session_id,
        "text": sched.message,
        "energy": sched.energy,
        "metadata": {
            "schedule_id": sched.id,
            "label": sched.label,
            "cron": sched.cron,
        },
    }
    logger.info(f"Firing schedule {sched.id} ({sched.label or ''}, {sched.energy}) → {sched.session_id}")
    try:
        await _http.post(f"{session_manager_url}/event", json=event)
    except Exception as e:
        logger.error(f"Failed to fire schedule {sched.id}: {e}")


# ── Entrypoint ───────────────────────────────────────────

async def main():
    global session_manager_url, _http, _store_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    session_manager_url = os.environ.get("SESSION_MANAGER_URL", "http://localhost:5000")
    mcp_port = int(os.environ.get("TIME_MCP_PORT", "8300"))
    _store_path = Path(os.environ.get("SCHEDULE_STORE", "/data/schedules.json"))

    continuity_cron = os.environ.get("CONTINUITY_CRON", "*/20 * * * *")
    continuity_message = os.environ.get("CONTINUITY_MESSAGE", "continuity check")
    continuity_session = os.environ.get("CONTINUITY_SESSION", "")

    _http = httpx.AsyncClient(timeout=600)

    _load()

    if continuity_session:
        cont = Schedule(
            id="continuity",
            cron=continuity_cron,
            message=continuity_message,
            session_id=continuity_session,
            label="continuity",
        )
        cont.compute_next()
        _schedules["continuity"] = cont
        logger.info(f"Continuity schedule registered: {continuity_cron} → {continuity_session}")
    else:
        logger.warning("CONTINUITY_SESSION not set — skipping continuity schedule")

    try:
        await asyncio.gather(
            _scheduler(),
            mcp.run_async(transport="http", host="0.0.0.0", port=mcp_port),
        )
    finally:
        await _http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
