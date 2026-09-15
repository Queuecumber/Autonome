"""Calendar parsing and bounded recurrence queries, adapted from aibs/ical/model.py."""

import base64
import copy
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import logging
import os
import re
from typing import Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from icalendar import Calendar, vText
from icalendar.parser import Contentline, Contentlines
from icalendar.prop import TypesFactory
from pydantic import BaseModel, Field, field_validator
import recurring_ical_events

logger = logging.getLogger(__name__)


class CalendarValidationError(ValueError):
    """A calendar validation failure with a safe, source-free diagnostic code.

    Args:
        code: Fixed category suitable for status responses and logs. Event
            contents, URLs, and the underlying parser message are never included.
    """
    def __init__(self, code: Literal["malformed_content", "invalid_timezone", "missing_start",
                                    "mixed_date_types", "invalid_interval", "ambiguous_event_id",
                                    "invalid_recurrence"]):
        """Retain the diagnostic code while remaining compatible with ValueError callers."""
        self.code = code
        super().__init__(code)


def repair_description_lines(raw: bytes) -> tuple[bytes, int]:
    """Recover malformed continuation text immediately following a VEVENT DESCRIPTION.

    Args:
        raw: Original feed bytes, left unchanged.

    Returns:
        Parser-normalized bytes and the number of recovered lines. Proper
        properties remain properties; text uses icalendar's TEXT escaping.

    Raises:
        CalendarValidationError: A malformed line is outside DESCRIPTION, looks
            like a broken standard/extension property, or has unsafe structure.
    """
    lines = Contentlines()
    stack = []
    description = None
    repaired = 0
    reserved = set(TypesFactory.types_map) | {"BEGIN", "END"}
    for line in Contentlines.from_ical(raw):
        if not line:
            continue
        try:
            name, params, value = line.parts()
        except ValueError:
            token = re.match(r"[A-Za-z0-9-]+", str(line))
            prefix = token.group().upper() if token else ""
            if (description is None or not stack or stack[-1] != "VEVENT"
                    or prefix in reserved or prefix.startswith("X-")):
                raise CalendarValidationError("malformed_content") from None
            index, params, text = description
            text += "\n" + str(vText.from_ical(str(line)))
            lines[index] = Contentline.from_parts("DESCRIPTION", params, vText(text))
            description = (index, params, text)
            repaired += 1
            continue
        name = name.upper()
        if name == "BEGIN":
            stack.append(value.upper())
        elif name == "END":
            if not stack or stack.pop() != value.upper():
                raise CalendarValidationError("malformed_content")
        description = ((len(lines), params, str(vText.from_ical(value)))
                       if name == "DESCRIPTION" and stack and stack[-1] == "VEVENT" else None)
        lines.append(line)
    if stack:
        raise CalendarValidationError("malformed_content")
    return lines.to_ical(), repaired


def read_calendar(raw: bytes) -> Calendar:
    """Parse a calendar, narrowly recovering malformed DESCRIPTION continuations.

    Args:
        raw: Original complete feed bytes; callers can retain them for auditing.

    Returns:
        Parsed calendar with recovered text, without dropping malformed properties.
        Recovery logs contain counts only, never description text or feed URLs.

    Raises:
        CalendarValidationError: Unsafe or unrecoverable syntax; previous cached
            data must remain in use until a valid refresh succeeds.
    """
    try:
        cal = Calendar.from_ical(raw)
        if any(component.errors for component in cal.walk()):
            repaired, count = repair_description_lines(raw)
            cal = Calendar.from_ical(repaired)
            if count:
                logger.warning("Recovered %d malformed DESCRIPTION continuation line(s)", count)
        if cal.name != "VCALENDAR" or any(component.errors for component in cal.walk()):
            raise CalendarValidationError("malformed_content")
        return cal
    except (ValueError, IndexError) as error:
        if isinstance(error, CalendarValidationError):
            raise
        raise CalendarValidationError("malformed_content") from None


def notification_date(value: str) -> datetime | None:
    """Parse startup/all modes or an explicit ISO date/aware timestamp for notifications.

    Returns:
        UTC timestamp for a date/timestamp, or None for either named mode.

    Raises:
        ValueError: Invalid input or an ISO timestamp without a timezone.
    """
    if value in {"startup", "all"}:
        return None
    result = datetime.fromisoformat(value)
    if len(value) == 10:
        result = result.replace(tzinfo=timezone.utc)
    if result.utcoffset() is None:
        raise ValueError("ICAL_NOTIFY_SINCE requires a date or timezone-aware timestamp")
    return result.astimezone(timezone.utc)


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
        notify_since: startup persists first activation; all disables the date
            filter; an ISO date or aware timestamp sets an explicit cutoff.

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
    notify_since: str = "startup"

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
        notification_date(self.notify_since)

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
                   os.environ.get("ICAL_SESSION_ID", ""), os.environ.get("ICAL_EVENT_ENERGY", "passive"),
                   os.environ.get("ICAL_NOTIFY_SINCE", "startup"))


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


def notification_time(item: dict, since: datetime) -> bool:
    """Return whether an event/occurrence starts on or after the floor or overlaps it."""
    zone = ZoneInfo(item.get("calendar_timezone", "UTC"))
    start = Event.preserve_date_type(item.get("start"))
    end = Event.preserve_date_type(item.get("end"))
    return bool((start is not None and instant(start, zone) >= since)
                or (end is not None and instant(end, zone) > since))


def notification_contexts(old_raw: bytes | None, previous: dict, raw: bytes, current: dict,
                          since: datetime, zone: ZoneInfo) -> dict[str, dict]:
    """Find qualifying dates for changed source records without filtering stored history.

    Args:
        old_raw: Previous feed, or None for a source lookup.
        previous: Previous source snapshot.
        raw: Current complete feed.
        current: Current source snapshot, possibly narrowed to queued source IDs.
        since: Inclusive notification date floor.
        zone: Fallback zone for floating/all-day dates.

    Returns:
        Source IDs mapped to a qualifying source/recurrence interval. Both old
        and new dates are checked, preserving cancellations and moves out of the
        window. Past recurring masters qualify only if the recurrence engine
        finds an occurrence after the floor; a past individual exception does not
        qualify merely because its parent still recurs.
    """
    calendars = {}
    result = {}
    for identifier in previous.keys() | current.keys():
        if previous.get(identifier) == current.get(identifier):
            continue
        for item, content in [(current.get(identifier), raw), (previous.get(identifier), old_raw)]:
            if item is None:
                continue
            event = item["event"]
            candidate = event
            if not notification_time(candidate, since):
                if content is None:
                    continue
                if content not in calendars:
                    calendars[content] = read_calendar(content)
                cal = calendars[content]
                related = [entry for entry in cal.walk("VEVENT") if str(entry.get("UID")) == event["uid"]]
                key = EventKey.decode(identifier)
                source = next((entry for entry in related if component_key(entry, key.feed) == key), None)
                if source is None:
                    continue
                rid = source.get("RECURRENCE-ID")
                if rid is not None and str(rid.params.get("RANGE", "")) != "THISANDFUTURE":
                    continue
                if not any(entry.get("RRULE") or entry.get("RDATE") for entry in related):
                    continue
                selected = copy.copy(cal)
                selected.subcomponents = [entry for entry in cal.subcomponents
                                          if entry.name != "VEVENT" or entry in related]
                effective_zone = calendar_zone(cal, zone)
                query = recurrence_query(selected, effective_zone)
                candidate = None
                for occurrence in query.after(since.astimezone(effective_zone)):
                    value = render(occurrence, event["calendar"], key.feed, effective_zone, occurrence=True)
                    data = value.model_dump(mode="json")
                    if notification_time(data, since):
                        candidate = data
                        break
                if candidate is None:
                    continue
            result[identifier] = {key: candidate.get(key) for key in
                                  ["start", "end", "all_day", "calendar_timezone", "recurrence_id"]}
            break
    return result


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
    except (KeyError, ValueError):
        raise CalendarValidationError("invalid_timezone") from None


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
    try:
        return recurring_ical_events.of(prepared, keep_recurrence_attributes=True)
    except ValueError:
        raise CalendarValidationError("invalid_recurrence") from None


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
    cal = read_calendar(raw)
    zone = calendar_zone(cal, zone)
    snapshot = {}
    for component in cal.walk("VEVENT"):
        item = render(component, name, feed, zone)
        if item.start is None and item.status != "CANCELLED":
            raise CalendarValidationError("missing_start")
        if item.start is not None and item.end is not None:
            if isinstance(item.start, datetime) != isinstance(item.end, datetime):
                raise CalendarValidationError("mixed_date_types")
            duration = instant(item.end, zone) - instant(item.start, zone)
            if duration < timedelta() or (item.all_day and duration == timedelta()):
                raise CalendarValidationError("invalid_interval")
        if item.id in snapshot:
            raise CalendarValidationError("ambiguous_event_id")
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
    cal = read_calendar(raw)
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
