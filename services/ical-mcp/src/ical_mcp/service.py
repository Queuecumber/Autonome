"""Conditional feed refreshes, persistent snapshots, and retried calendar change events."""

import asyncio
from datetime import date, datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sqlite3
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

from ical_mcp.model import (Event, EventKey, Settings, calendar_zone, instant, notification_contexts,
                            notification_date, notification_time, parse_feed, query_range)

logger = logging.getLogger(__name__)


class Store:
    """Persist complete validated feeds and pending change events in SQLite.

    Args:
        path: Persistent database path, used by one service instance.

    Raises:
        sqlite3.Error: Storage failure; snapshot/outbox transactions roll back together.
    """
    def __init__(self, path: Path):
        """Open a state database without discarding existing snapshots or queued events."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS feeds (
                id TEXT PRIMARY KEY, raw BLOB, snapshot TEXT NOT NULL DEFAULT '{}',
                etag TEXT NOT NULL DEFAULT '', modified TEXT NOT NULL DEFAULT '',
                as_of TEXT NOT NULL DEFAULT '', checked TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS outbox (
                id TEXT PRIMARY KEY, feed TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS notification_floors (
                feed TEXT PRIMARY KEY, since TEXT NOT NULL);
        """)

    def notification_floor(self, feed: str) -> datetime:
        """Return a persisted first-use cutoff; restarts never move this boundary forward."""
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO notification_floors VALUES (?, ?)",
                            (feed, datetime.now(timezone.utc).isoformat()))
            value = self.db.execute("SELECT since FROM notification_floors WHERE feed=?", (feed,)).fetchone()[0]
            return datetime.fromisoformat(value)

    def get(self, feed: str) -> dict | None:
        """Return the cached feed/status or None before its first refresh attempt."""
        row = self.db.execute("SELECT * FROM feeds WHERE id=?", (feed,)).fetchone()
        return dict(row) if row is not None else None

    def checked(self, feed: str, when: str, error: str = "") -> None:
        """Record a refresh result without replacing cached bytes; failures retain old data."""
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO feeds(id) VALUES (?)", (feed,))
            if error:
                self.db.execute("UPDATE feeds SET checked=?, error=? WHERE id=?", (when, error, feed))
            else:
                self.db.execute("UPDATE feeds SET checked=?, as_of=?, error='' WHERE id=?", (when, when, feed))

    def replace(self, feed: str, name: str, raw: bytes, snapshot: dict, etag: str,
                modified: str, when: str, settings: Settings,
                eligible: dict[str, dict] | None = None) -> None:
        """Atomically retain a valid snapshot and queue source-level changes in batches of 20.

        Args:
            feed: Current feed identity.
            name: Configured calendar name.
            raw: Complete validated iCalendar bytes.
            snapshot: Source component fingerprints and event data.
            etag: HTTP validator, or empty.
            modified: Last-Modified validator, or empty.
            when: UTC observation timestamp.
            settings: Event energy and explicit session routing.
            eligible: Date-qualified source IDs and intervals, or None to notify all changes.

        Returns:
            None. First successful use establishes a quiet baseline, even after failures.
            Removal means absent from this feed, not necessarily cancelled in reality.
        """
        old = self.get(feed)
        changes = []
        if old and old["raw"] is not None:
            previous = json.loads(old["snapshot"])
            for identifier in sorted(previous.keys() | snapshot.keys()):
                before, after = previous.get(identifier), snapshot.get(identifier)
                if before == after:
                    continue
                if eligible is not None and identifier not in eligible:
                    continue
                change = ("removed_from_feed" if after is None else "added" if before is None
                          else "cancelled" if after["event"]["status"] == "CANCELLED" else "updated")
                item = (after or before)["event"]
                brief = {key: item[key] for key in ["id", "uid", "start", "end", "all_day", "status", "recurrence_id"]}
                brief["subject"] = item["subject"][:500]
                change_record = {"change": change, "event": brief}
                if eligible is not None:
                    change_record["notification_context"] = eligible[identifier]
                changes.append(change_record)
        with self.db:
            for offset in range(0, len(changes), 20):
                identifier = str(uuid4())
                event = {"source": "ical", "event_type": "calendar_changed", "energy": settings.energy,
                         "text": json.dumps({"calendar": name, "changes": changes[offset:offset + 20]}),
                         "metadata": {"event_id": identifier, "calendar": name, "observed_at": when}}
                if settings.session_id:
                    event["session_id"] = settings.session_id
                self.db.execute("INSERT INTO outbox VALUES (?, ?, ?)", (identifier, feed, json.dumps(event)))
            self.db.execute("INSERT OR REPLACE INTO feeds VALUES (?, ?, ?, ?, ?, ?, ?, '')",
                            (feed, raw, json.dumps(snapshot, sort_keys=True), etag, modified, when, when))

    def pending(self, feeds: list[str]) -> list[tuple[str, dict]]:
        """Return up to 100 pending events belonging only to currently configured feeds."""
        placeholders = ",".join("?" for _ in feeds)
        return [(row["id"], json.loads(row["payload"])) for row in self.db.execute(
            f"SELECT id, payload FROM outbox WHERE feed IN ({placeholders}) ORDER BY rowid LIMIT 100", feeds)]

    def acknowledge(self, identifier: str) -> None:
        """Remove one HTTP-accepted event; a lost response may cause a later duplicate."""
        with self.db:
            self.db.execute("DELETE FROM outbox WHERE id=?", (identifier,))

    def close(self) -> None:
        """Close the database after refresh and delivery tasks have stopped."""
        self.db.close()


class Calendars:
    """Read named feeds and notify the agent when their source records change.

    Args:
        settings: Feed URLs and operational policy.
        store: Single-instance persistent state.
        http: HTTP client with finite timeouts and TLS validation.
    """
    def __init__(self, settings: Settings, store: Store, http: httpx.AsyncClient):
        """Configure locks and shutdown state without fetching feeds."""
        self.settings, self.store, self.http = settings, store, http
        self.zone = ZoneInfo(settings.timezone_name)
        self.locks = {name: asyncio.Lock() for name in settings.urls}
        self.stop = asyncio.Event()
        for name in settings.urls:
            self.notification_floor(name)

    def notification_floor(self, name: str) -> datetime | None:
        """Return a feed's effective notification cutoff, or None when filtering is disabled."""
        if self.settings.notify_since == "startup":
            return self.store.notification_floor(self.settings.feed_id(name))
        return notification_date(self.settings.notify_since)

    def status(self, name: str) -> dict:
        """Return availability/freshness for a calendar without exposing its private URL."""
        row = self.store.get(self.settings.feed_id(name))
        floor = self.notification_floor(name)
        return {"calendar": name, "available": bool(row and row["raw"] is not None),
                "notify_since": floor.isoformat() if floor else None,
                "as_of": row["as_of"] if row else "", "last_checked": row["checked"] if row else "",
                "error": row["error"] if row else "", "stale": self.stale(row)}

    def stale(self, row: dict | None) -> bool:
        """Return whether a cached feed is missing, failed its refresh, or aged beyond two intervals."""
        return (not row or not row["as_of"] or bool(row["error"])
                or (datetime.now(timezone.utc) - datetime.fromisoformat(row["as_of"])).total_seconds()
                > self.settings.refresh_seconds * 2)

    async def refresh(self, name: str) -> None:
        """Conditionally fetch one complete feed, preserving old data on every failed refresh.

        Args:
            name: Configured calendar name; unknown names raise KeyError.

        Returns:
            None. Failures set a sanitized status error, never a fabricated empty calendar.
        """
        feed = self.settings.feed_id(name)
        async with self.locks[name]:
            old = self.store.get(feed)
            headers = {}
            if old and old["raw"] is not None:
                if old["etag"]:
                    headers["If-None-Match"] = old["etag"]
                if old["modified"]:
                    headers["If-Modified-Since"] = old["modified"]
            when = datetime.now(timezone.utc).isoformat()
            try:
                async with asyncio.timeout(self.settings.timeout_seconds), self.http.stream(
                        "GET", self.settings.urls[name], headers=headers, follow_redirects=False) as response:
                    if response.status_code == 304:
                        if not old or old["raw"] is None:
                            raise ValueError("No cached feed for HTTP 304")
                        self.store.checked(feed, when)
                        return
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise ValueError("Expected a complete HTTP 200 calendar feed")
                    raw = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        raw.extend(chunk)
                        if len(raw) > self.settings.max_feed_bytes:
                            raise ValueError("Calendar feed exceeds configured size limit")
                    _, snapshot = await asyncio.to_thread(parse_feed, bytes(raw), name, feed, self.zone)
                    floor = self.notification_floor(name)
                    eligible = None
                    if floor is not None and old and old["raw"] is not None:
                        eligible = await asyncio.to_thread(notification_contexts, old["raw"],
                            json.loads(old["snapshot"]), bytes(raw), snapshot, floor, self.zone)
                    self.store.replace(feed, name, bytes(raw), snapshot, response.headers.get("etag", ""),
                                       response.headers.get("last-modified", ""), when, self.settings, eligible)
            except Exception as error:
                self.store.checked(feed, when, type(error).__name__)
                logger.warning("Calendar refresh failed (%s); cached data retained", type(error).__name__)

    async def cached(self, name: str) -> dict:
        """Return cached bytes, refreshing when due; unavailable feeds raise RuntimeError."""
        feed = self.settings.feed_id(name)
        row = self.store.get(feed)
        if (not row or not row["checked"] or
                (datetime.now(timezone.utc) - datetime.fromisoformat(row["checked"])).total_seconds()
                >= self.settings.refresh_seconds):
            await self.refresh(name)
            row = self.store.get(feed)
        if not row or row["raw"] is None:
            raise RuntimeError("Calendar unavailable; inspect list_calendars for refresh status")
        return row

    async def range(self, start: datetime, end: datetime, name: str | None, limit: int) -> dict:
        """Return a bounded merged occurrence range with freshness and explicit truncation.

        Args:
            start: Inclusive aware start.
            end: Exclusive aware end; at most 366 days after start.
            name: One calendar, or None/empty for all configured calendars.
            limit: Maximum returned events, from 1 to 100.

        Raises:
            ValueError: Invalid dates or bounds.
            RuntimeError: Any selected calendar has no valid snapshot.
            KeyError: Unknown calendar name.
        """
        if start.utcoffset() is None or end.utcoffset() is None or not 1 <= limit <= 100:
            raise ValueError("Use timezone-aware dates and a limit from 1 to 100")
        start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        if not timedelta() < end - start <= timedelta(days=366):
            raise ValueError("Calendar range must be positive and no longer than 366 days")
        names = [name] if name else list(self.settings.urls)
        found, statuses = [], []
        for calendar in names:
            row = await self.cached(calendar)
            items = await asyncio.to_thread(query_range, row["raw"], calendar, row["id"], self.zone, start, end, limit + 1)
            for item in items:
                item.as_of, item.stale = row["as_of"], self.stale(row)
            found.extend(items)
            statuses.append(self.status(calendar))
        found.sort(key=lambda event: (instant(event.start, ZoneInfo(event.calendar_timezone)), event.calendar, event.id))
        return {"events": [item.model_dump(mode="json") for item in found[:limit]],
                "truncated": len(found) > limit, "calendars": statuses}

    async def get(self, identifier: str) -> Event:
        """Resolve a source or occurrence ID, rejecting foreign or rescheduled stale IDs.

        Args:
            identifier: Opaque ID from a range query or change notification.

        Returns:
            Full event data with snapshot freshness.

        Raises:
            ValueError: Malformed ID or a feed no longer configured.
            KeyError: Removed component, excluded occurrence, or stale occurrence start.
            RuntimeError: No valid cached feed.
        """
        key = EventKey.decode(identifier)
        name = next((name for name in self.settings.urls if self.settings.feed_id(name) == key.feed), None)
        if name is None:
            raise ValueError("Event ID belongs to an unconfigured calendar feed")
        row = await self.cached(name)
        if not key.occurrence:
            item = json.loads(row["snapshot"]).get(key.encode())
            if item is None:
                raise KeyError("Event is no longer in the feed")
            event = Event.model_validate(item["event"])
        else:
            start = date.fromisoformat(key.occurrence) if len(key.occurrence) == 10 else datetime.fromisoformat(key.occurrence)
            # Query only the requested UID, so unrelated overlapping events cannot hide it behind the limit.
            cal, _ = await asyncio.to_thread(parse_feed, row["raw"], name, key.feed, self.zone)
            when = instant(start, calendar_zone(cal, self.zone))
            cal.subcomponents = [component for component in cal.subcomponents
                                 if component.name != "VEVENT" or str(component.get("UID")) == key.uid]
            items = await asyncio.to_thread(query_range, cal.to_ical(), name, key.feed, self.zone,
                                           when, when + timedelta(seconds=1), 1, key.encode())
            event = next((item for item in items if item.id == key.encode()), None)
            if event is None:
                raise KeyError("Occurrence was removed or rescheduled; query its current range")
        event.as_of, event.stale = row["as_of"], self.stale(row)
        return event

    async def refresh_loop(self) -> None:
        """Refresh every configured feed until stopped, isolating provider failures per feed."""
        while not self.stop.is_set():
            for name in self.settings.urls:
                if self.stop.is_set():
                    return
                await self.refresh(name)
            try:
                await asyncio.wait_for(self.stop.wait(), self.settings.refresh_seconds)
            except TimeoutError:
                pass

    async def deliver_once(self, url: str) -> int:
        """Process a pending page, suppressing excluded history and retrying eligible HTTP failures."""
        pending = self.store.pending([self.settings.feed_id(name) for name in self.settings.urls])
        for identifier, event in pending:
            name = event["metadata"]["calendar"]
            floor = self.notification_floor(name)
            if floor is not None:
                payload = json.loads(event["text"])
                retained = []
                row = self.store.get(self.settings.feed_id(name))
                snapshot = json.loads(row["snapshot"]) if row else {}
                unresolved = {change["event"]["id"]: snapshot[change["event"]["id"]]
                              for change in payload["changes"]
                              if change["event"]["id"] in snapshot
                              and not notification_time(change.get("notification_context") or change["event"], floor)}
                contexts = {}
                if unresolved and row["raw"] is not None:
                    contexts = await asyncio.to_thread(notification_contexts, None, {}, row["raw"],
                                                       unresolved, floor, self.zone)
                for change in payload["changes"]:
                    context = change.get("notification_context") or change["event"]
                    if notification_time(context, floor):
                        retained.append(change)
                        continue
                    key = change["event"]["id"]
                    if key in contexts:
                        retained.append({**change, "notification_context": contexts[key]})
                if not retained:
                    self.store.acknowledge(identifier)
                    continue
                payload["changes"] = retained
                event = {**event, "text": json.dumps(payload)}
            response = await self.http.post(f"{url.rstrip('/')}/event", json=event)
            response.raise_for_status()
            self.store.acknowledge(identifier)
        return len(pending)

    async def delivery_loop(self, url: str) -> None:
        """Retry HTTP/storage failures until stopped; acceptance is not proof of agent processing."""
        while not self.stop.is_set():
            try:
                if await self.deliver_once(url):
                    continue
            except (httpx.HTTPError, sqlite3.Error) as error:
                logger.warning("Calendar event delivery retry (%s)", type(error).__name__)
            try:
                await asyncio.wait_for(self.stop.wait(), 5)
            except TimeoutError:
                pass
