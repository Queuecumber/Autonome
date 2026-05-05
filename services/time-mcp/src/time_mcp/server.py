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
  """
# Time Tools

These tools allow you to manage time by scheduling wakeups and querying the wall clock.
This allows you to schedule specific tasks to be run on a schedule using cron syntax. After
configuring a task, an event will fire from the MCP on the schedule you chose with whatever
additional information you provided and allow you to take actions.

You will also receive periodic minimal wakeups from this tool for continuity. These messages
are called "continuitons" and give you a chance to take actions without an explicit event
from a user or other external source. These messages are minimal to reduce token consumption,
currently they are set to the string "✨". Use these events to do whatever you want including
check in with a human.
"""
)) # TODO substitute actual continuity message if overridden


class Schedule(BaseModel):
    id: str
    cron: str
    message: str
    session_id: str | None = None
    energy: str = "passive"

    @computed_field
    @property
    def next_fire(self) -> str | None:
        job = _scheduler.get_job(self.id) if _scheduler else None
        return job.next_run_time.isoformat() if job and job.next_run_time else None


def _save() -> None:
    _store_path.parent.mkdir(parents=True, exist_ok=True)
    fields = set(Schedule.model_fields)
    data = [s.model_dump(include=fields) for s in _schedules.values()]
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
        args=[sched],
        id=sched.id,
        replace_existing=True,
    )


# ── Tools ────────────────────────────────────────────────

@mcp.tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S %Z (%A)") -> str:
    """Read the wall clock.

    Args:
        format: strftime format. Default produces e.g.
            `2026-05-05 14:31:09 EDT (Friday)`.

    Returns:
        The current local time formatted per the strftime spec.
    """
    return datetime.now().astimezone().strftime(format)


@mcp.tool
def schedule_cron(
    schedule_id: str, cron: str, message: str,
    session_id: str | None = None,
    energy: str = "passive",
) -> None:
    """Schedule a recurring wakeup that fires events.

    Persists across restarts. When the cron fires, an event is dispatched
    with `text=message` for the agent to act on (or ignore). Defaults to
    landing in your current session — pass `session_id` only to target a
    different one.

    Args:
        schedule_id: A short memorable name (e.g. `morning-checkin`),
            used later to cancel. `continuity` is reserved by the platform.
        cron: A standard cron expression (e.g. `*/20 * * * *` for every
            20 minutes).
        message: Text delivered when the schedule fires.
        session_id: Optional explicit target session. Omit to land in the
            default session.
        energy: `"active"` preempts in-progress generation; `"passive"`
            queues if busy. Use `active` only when immediate attention
            is genuinely required.

    Raises:
        ValueError: If `energy` is not a valid value, `schedule_id` is
            already in use or reserved, or `cron` isn't valid.
    """
    if energy not in ("active", "passive"):
        raise ValueError(f"energy must be 'active' or 'passive', got {energy!r}")
    if schedule_id == "continuity":
        raise ValueError("'continuity' is a reserved schedule id")
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
    """List all active schedules.

    Returns:
        Each schedule's id, cron, message, session, energy, and computed
        next-fire timestamp.
    """
    return list(_schedules.values())


@mcp.tool
def cancel_schedule(schedule_id: str) -> None:
    """Cancel a previously-created schedule.

    Args:
        schedule_id: The id you passed to `schedule_cron`.

    Raises:
        ValueError: If no schedule with that id exists.
    """
    if schedule_id not in _schedules:
        raise ValueError(f"No schedule with id {schedule_id}")
    _scheduler.remove_job(schedule_id)
    del _schedules[schedule_id]
    _save()


# ── Firing ───────────────────────────────────────────────

async def _fire(sched: Schedule) -> None:
    """POST a scheduled event to the session manager."""
    event: dict = {
        "source": "time",
        "event_type": "continuity" if sched.id == "continuity" else "cron",
        "text": sched.message,
        "energy": sched.energy,
        "metadata": {
            "schedule_id": sched.id,
            "cron": sched.cron,
        },
    }
    if sched.session_id:
        event["session_id"] = sched.session_id
    logger.info(f"Firing schedule {sched.id} ({sched.energy}) → {sched.session_id or 'default'}")
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
    continuity_message = os.environ.get("CONTINUITY_MESSAGE", "✨")

    _http = httpx.AsyncClient(timeout=600)
    _scheduler = AsyncIOScheduler()
    _scheduler.start()

    _load()

    _add_job(Schedule(
        id="continuity",
        cron=continuity_cron,
        message=continuity_message,
    ))
    logger.info(f"Continuity schedule registered: {continuity_cron}")

    try:
        await mcp.run_async(transport="http", host="0.0.0.0", port=mcp_port)
    finally:
        _scheduler.shutdown()
        await _http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
