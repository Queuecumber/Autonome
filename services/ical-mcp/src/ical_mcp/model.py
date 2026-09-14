"""Calendar parsing and bounded recurrence queries, adapted from aibs/ical/model.py."""

import base64
import copy
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from icalendar import Calendar
from pydantic import BaseModel, Field, field_validator
import recurring_ical_events


@dataclass(frozen=True)
class Settings:
    """Named private feeds and refresh/query limits; URLs are omitted from repr.

    Args:
        urls: Calendar name to HTTPS feed URL, normally supplied through a Secret.
        timezone_name: Zone for floating times and all-day query boundaries.
        refresh_seconds: Adapter-side feed refresh interval.
        timeout_seconds: HTTP timeout.
        max_feed_bytes: Maximum downloaded/decoded feed size.
        session_id: Optional explicit session route.
        energy: Passive by default; active preempts ongoing generation.

    Raises:
        ValueError: Invalid URLs, timezone, routing, or bounds.
    """
    urls: dict[str, str] = field(repr=False)
    timezone_name: str = "UTC"
    refresh_seconds: float = 300
    timeout_seconds: float = 20
    max_feed_bytes: int = 5 * 1024 * 1024
    session_id: str = ""
    energy: str = "passive"

    def __post_init__(self):
        """Validate configuration without fetching any private feed."""
        if not isinstance(self.urls, dict) or not 1 <= len(self.urls) <= 32:
            raise ValueError("ICAL_URLS must contain 1..32 named feeds")
        for name, value in self.urls.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(value, str):
                raise ValueError("Calendar names and URLs must be nonempty strings")
            url = urlparse(value)
            if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
                raise ValueError("Calendar feeds must be HTTPS URLs without userinfo or fragments")
        try:
            ZoneInfo(self.timezone_name)
        except (KeyError, ValueError) as error:
            raise ValueError("Invalid ICAL_TIMEZONE") from error
        if min(self.refresh_seconds, self.timeout_seconds, self.max_feed_bytes) <= 0:
            raise ValueError("Calendar refresh and size limits must be positive")
        if self.energy not in {"active", "passive"}:
            raise ValueError("ICAL_EVENT_ENERGY must be passive or active")

    def feed_id(self, name: str) -> str:
        """Return an opaque name/URL identity; unknown calendar names raise KeyError."""
        return hashlib.sha256(json.dumps([name, self.urls[name]]).encode()).hexdigest()

    @classmethod
    def from_env(cls) -> "Settings":
        """Read ICAL_* variables; malformed configuration raises ValueError."""
        return cls(json.loads(os.environ.get("ICAL_URLS", "{}")),
                   os.environ.get("ICAL_TIMEZONE", "UTC"),
                   float(os.environ.get("ICAL_REFRESH_SECONDS", "300")),
                   float(os.environ.get("ICAL_TIMEOUT_SECONDS", "20")),
                   int(os.environ.get("ICAL_MAX_FEED_BYTES", str(5 * 1024 * 1024))),
                   os.environ.get("ICAL_SESSION_ID", ""), os.environ.get("ICAL_EVENT_ENERGY", "passive"))


class EventKey(BaseModel):
    """An opaque feed-scoped source component or expanded occurrence identifier."""
    feed: str = Field(min_length=1)
    uid: str = Field(min_length=1)
    recurrence: str = ""
    occurrence: str = ""

    def encode(self) -> str:
        """Return a URL-safe ID without exposing the feed's URL or access tokens."""
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")

    @classmethod
    def decode(cls, value: str) -> "EventKey":
        """Parse a full opaque ID; raises ValueError for malformed or oversized input."""
        try:
            if not value or len(value) > 8192:
                raise ValueError("Invalid length")
            return cls.model_validate_json(base64.b64decode(value + "=" * (-len(value) % 4),
                                                           altchars=b"-_", validate=True))
        except ValueError as error:
            raise ValueError("Invalid calendar event ID") from error


class Event(BaseModel):
    """One source event or occurrence, preserving all-day dates and explicit freshness."""
    id: str
    uid: str
    calendar: str
    subject: str
    organizer: str
    start: datetime | date | None
    end: datetime | date | None
    all_day: bool
    calendar_timezone: str = "UTC"
    recurrence_id: str = ""
    recurrence_rule: str = ""
    location: str = ""
    attendees: list[str] = Field(default_factory=list)
    body: str = ""
    status: str = ""
    as_of: str = ""
    stale: bool = False

    @field_validator("start", "end", mode="before")
    @classmethod
    def preserve_date_type(cls, value):
        """Restore ISO dates/times to their original type; malformed strings raise ValueError."""
        if isinstance(value, str):
            return date.fromisoformat(value) if len(value) == 10 else datetime.fromisoformat(value)
        return value


def instant(value: datetime | date, zone: ZoneInfo) -> datetime:
    """Interpret dates/floating times in zone and return an aware UTC instant."""
    if not isinstance(value, datetime):
        value = datetime.combine(value, time.min)
    if value.utcoffset() is None:
        value = value.replace(tzinfo=zone)
    return value.astimezone(timezone.utc)


def recurrence_id(component) -> str:
    """Return a canonical recurrence identity, or empty for a series master."""
    value = component.get("RECURRENCE-ID")
    if value is None:
        return ""
    value = value.dt
    if isinstance(value, datetime) and value.utcoffset() is not None:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def component_key(component, feed: str) -> EventKey:
    """Identify a UID/RECURRENCE-ID source component without expanding its series."""
    return EventKey(feed=feed, uid=str(component.get("UID", "")), recurrence=recurrence_id(component))


def calendar_zone(cal: Calendar, fallback: ZoneInfo) -> ZoneInfo:
    """Use a feed's declared timezone for floating/all-day values, or the configured fallback."""
    try:
        return ZoneInfo(str(cal.get("X-WR-TIMEZONE", fallback.key)))
    except (KeyError, ValueError) as error:
        raise ValueError("Invalid calendar timezone") from error


def recurrence_query(cal: Calendar, zone: ZoneInfo):
    """Build a validated recurrence engine, retaining sparse cancellation exceptions.

    Args:
        cal: Parsed source calendar, left unchanged.
        zone: Effective feed timezone.

    Returns:
        A recurring-ical-events query; invalid recurrence rules raise ValueError.
    """
    prepared = copy.deepcopy(cal)
    for component in prepared.walk("VEVENT"):
        if component.get("DTSTART") is None and component.get("RECURRENCE-ID") is not None:
            component.add("DTSTART", component.decoded("RECURRENCE-ID"))
    prepared.subcomponents = [component for component in prepared.subcomponents
                              if component.name != "VEVENT" or component.get("DTSTART") is not None]
    prepared["X-WR-TIMEZONE"] = zone.key
    return recurring_ical_events.of(prepared, keep_recurrence_attributes=True)


def render(component, name: str, feed: str, zone: ZoneInfo, occurrence: bool = False) -> Event:
    """Convert one parsed component into caller-facing event data.

    Args:
        component: An icalendar VEVENT or recurrence-expanded component.
        name: Configured calendar name.
        feed: Opaque feed identity.
        zone: Interpretation of floating datetimes; DATE values remain dates.
        occurrence: Include the expanded start in the ID, distinguishing repetitions.

    Returns:
        Full event details. A cancellation without dates has null start/end.
    """
    start = component.decoded("DTSTART", None)
    end = component.decoded("DTEND", None)
    all_day = isinstance(start, date) and not isinstance(start, datetime)
    if start is not None and end is None:
        end = start + component.decoded("DURATION", timedelta(days=1) if all_day else timedelta())
    if isinstance(start, datetime) and start.utcoffset() is None:
        start = start.replace(tzinfo=zone)
    if isinstance(end, datetime) and end.utcoffset() is None:
        end = end.replace(tzinfo=zone)
    key = component_key(component, feed)
    if occurrence and start is not None:
        key.occurrence = start.isoformat()
    attendees = component.get("ATTENDEE", [])
    if not isinstance(attendees, list):
        attendees = [attendees]
    rule = component.get("RRULE")
    return Event(id=key.encode(), uid=key.uid, calendar=name,
                 subject=str(component.get("SUMMARY", "")),
                 organizer=str(component.get("ORGANIZER", "")).removeprefix("mailto:"),
                 start=start, end=end, all_day=all_day, calendar_timezone=zone.key, recurrence_id=key.recurrence,
                 recurrence_rule=rule.to_ical().decode() if rule else "",
                 location=str(component.get("LOCATION", "")),
                 attendees=[str(value).removeprefix("mailto:") for value in attendees],
                 body=str(component.get("DESCRIPTION", "")), status=str(component.get("STATUS", "")))


def parse_feed(raw: bytes, name: str, feed: str, zone: ZoneInfo) -> tuple[Calendar, dict]:
    """Validate a complete feed and build a source-component snapshot, never expanding recurrences.

    Args:
        raw: Complete iCalendar document.
        name: Calendar name.
        feed: Opaque identity.
        zone: Floating-time interpretation.

    Returns:
        Parsed calendar and keyed fingerprints/full event data for change detection.

    Raises:
        ValueError: Malformed calendar, ambiguous components, missing required dates,
            or an invalid interval. No previous snapshot should be replaced on error.
    """
    cal = Calendar.from_ical(raw)
    if cal.name != "VCALENDAR" or any(component.errors for component in cal.walk()):
        raise ValueError("Malformed iCalendar feed")
    zone = calendar_zone(cal, zone)
    snapshot = {}
    for component in cal.walk("VEVENT"):
        item = render(component, name, feed, zone)
        if item.start is None and item.status != "CANCELLED":
            raise ValueError("Event is missing DTSTART")
        if item.start is not None and item.end is not None:
            if isinstance(item.start, datetime) != isinstance(item.end, datetime):
                raise ValueError("DTSTART and DTEND must use the same date type")
            duration = instant(item.end, zone) - instant(item.start, zone)
            if duration < timedelta() or (item.all_day and duration == timedelta()):
                raise ValueError("Event interval is invalid")
        if item.id in snapshot:
            raise ValueError("Duplicate UID and RECURRENCE-ID in one feed")
        stable = copy.deepcopy(component)
        for attribute in ["DTSTAMP", "LAST-MODIFIED", "SEQUENCE"]:
            stable.pop(attribute, None)
        snapshot[item.id] = {"fingerprint": hashlib.sha256(stable.to_ical()).hexdigest(),
                             "event": item.model_dump(mode="json")}
    recurrence_query(cal, zone)
    return cal, snapshot


def query_range(raw: bytes, name: str, feed: str, zone: ZoneInfo,
                start: datetime, end: datetime, limit: int, identifier: str = "") -> list[Event]:
    """Return at most limit eligible occurrences in a half-open time interval.

    Args:
        raw: Previously validated feed.
        name: Calendar name.
        feed: Opaque identity.
        zone: Zone for floating times and all-day boundaries.
        start: Aware inclusive query start.
        end: Aware exclusive query end.
        limit: Caller-enforced positive result budget, including any truncation probe.
        identifier: Optional exact occurrence ID, filtered before applying the limit.

    Returns:
        Start-ordered occurrences with distinct retrievable IDs. Recurrence logic
        and exception handling are delegated to recurring-ical-events.
    """
    cal = Calendar.from_ical(raw)
    zone = calendar_zone(cal, zone)
    query = recurrence_query(cal, zone)
    result, seen = [], set()
    cursor = start
    while cursor < end:
        boundary = min(cursor + timedelta(days=1), end)
        components = query.between(cursor.astimezone(zone), boundary.astimezone(zone))
        components.sort(key=lambda item: instant(item.decoded("DTSTART"), zone))
        for component in components:
            item = render(component, name, feed, zone, occurrence=True)
            beginning, ending = instant(item.start, zone), instant(item.end, zone)
            if (item.id not in seen and (not identifier or item.id == identifier)
                    and item.status != "CANCELLED" and beginning < end
                    and (ending > start or beginning == ending == start)):
                result.append(item)
                seen.add(item.id)
                if len(result) == limit:
                    return result
        cursor = boundary
    return result
