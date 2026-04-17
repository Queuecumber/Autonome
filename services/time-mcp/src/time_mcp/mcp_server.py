"""Time adapter — cron-based scheduled events + continuity wakeup.

Events fire to the session manager just like other adapters. The continuity
cron is always registered (configurable interval and message) to give the
agent regular opportunities to check in.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastmcp import FastMCP
from pydantic import BaseModel, computed_field

logger = logging.getLogger(__name__)

session_manager_url: str
_http: httpx.AsyncClient
_store_path: Path
_scheduler: AsyncIOScheduler | None = None
_schedules: dict[str, "Schedule"] = {}

mcp = FastMCP("time", instructions=(
    "Time and scheduling. Use get_current_time for wall clock. "
    "Use schedule_cron to set a recurring wakeup (cron syntax), or "
    "cancel_schedule to remove one. Scheduled events arrive with "
    "source='time'. A 'continuity' schedule runs at a configured interval "
    "to give you regular check-ins."
))


class Schedule(BaseModel):
    id: str
    cron: str
    message: str
    session_id: str
    energy: str = "passive"

    @computed_field
    @property
    def next_fire(self) -> str | None:
        job = _scheduler.get_job(self.id) if _scheduler else None
        return job.next_run_time.isoformat() if job and job.next_run_time else None


def _save() -> None:
    _store_path.parent.mkdir(parents=True, exist_ok=True)
    fields = set(Schedule.model_fields)
    data = [s.model_dump(include=fields) for s in _schedules.values() if s.id != "continuity"]
    _store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load() -> None:
    if not _store_path.exists():
        return
    for entry in json.loads(_store_path.read_text()):
        sched = Schedule(**entry)
        _schedules[sched.id] = sched
        _add_job(sched)
    logger.info(f"Loaded {len(_schedules)} schedules")


def _add_job(sched: Schedule) -> None:
    _scheduler.add_job(
        _fire,
        CronTrigger.from_crontab(sched.cron),
        args=[sched.id],
        id=sched.id,
        replace_existing=True,
    )


# ── Tools ────────────────────────────────────────────────

@mcp.tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S %Z (%A)") -> str:
    """Return the current wall-clock time. format is a strftime format string."""
    return datetime.now().astimezone().strftime(format)


@mcp.tool
def schedule_cron(
    schedule_id: str, cron: str, message: str, session_id: str,
    energy: str = "passive",
) -> None:
    """Schedule a recurring wakeup. cron is a standard cron expression
    (e.g. '*/20 * * * *' for every 20 minutes). The message is delivered
    to the given session_id when the schedule fires.

    schedule_id is a short, memorable name you choose (e.g. 'morning-checkin').
    Use it later to cancel. Rejected if already in use.

    energy is "active" (interrupts current generation) or "passive" (queues
    if busy). Most scheduled events should be passive — use active only when
    the schedule genuinely needs immediate attention.
    """
    if energy not in ("active", "passive"):
        raise ValueError(f"energy must be 'active' or 'passive', got {energy!r}")
    if schedule_id in _schedules:
        raise ValueError(f"Schedule id {schedule_id!r} already exists — cancel it first or pick a different name")
    try:
        CronTrigger.from_crontab(cron)
    except Exception as e:
        raise ValueError(f"Invalid cron expression {cron!r}: {e}")
    sched = Schedule(
        id=schedule_id,
        cron=cron, message=message, session_id=session_id,
        energy=energy,
    )
    _schedules[sched.id] = sched
    _add_job(sched)
    _save()


@mcp.tool
def list_schedules() -> list[Schedule]:
    """List all active schedules with their next fire time."""
    return list(_schedules.values())


@mcp.tool
def cancel_schedule(schedule_id: str) -> None:
    """Cancel a schedule by id. The continuity schedule cannot be cancelled."""
    if schedule_id == "continuity":
        raise ValueError("Continuity schedule cannot be cancelled")
    if schedule_id not in _schedules:
        raise ValueError(f"No schedule with id {schedule_id}")
    _scheduler.remove_job(schedule_id)
    del _schedules[schedule_id]
    _save()


# ── Firing ───────────────────────────────────────────────

async def _fire(schedule_id: str) -> None:
    """POST a scheduled event to the session manager."""
    sched = _schedules.get(schedule_id)
    if sched is None:
        logger.warning(f"Schedule {schedule_id} fired but is no longer registered")
        return
    event = {
        "source": "time",
        "session_id": sched.session_id,
        "event_type": "continuity" if sched.id == "continuity" else "cron",
        "text": sched.message,
        "energy": sched.energy,
        "metadata": {
            "schedule_id": sched.id,
            "cron": sched.cron,
        },
    }
    logger.info(f"Firing schedule {sched.id} ({sched.energy}) → {sched.session_id}")
    try:
        await _http.post(f"{session_manager_url}/event", json=event)
    except Exception as e:
        logger.error(f"Failed to fire schedule {sched.id}: {e}")


# ── Entrypoint ───────────────────────────────────────────

async def main():
    global session_manager_url, _http, _store_path, _scheduler

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
    _scheduler = AsyncIOScheduler()
    _scheduler.start()

    _load()

    if continuity_session:
        cont = Schedule(
            id="continuity",
            cron=continuity_cron,
            message=continuity_message,
            session_id=continuity_session,
        )
        _schedules["continuity"] = cont
        _add_job(cont)
        logger.info(f"Continuity schedule registered: {continuity_cron} → {continuity_session}")
    else:
        logger.warning("CONTINUITY_SESSION not set — skipping continuity schedule")

    try:
        await mcp.run_async(transport="http", host="0.0.0.0", port=mcp_port)
    finally:
        _scheduler.shutdown()
        await _http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
