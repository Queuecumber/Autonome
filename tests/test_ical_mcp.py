"""Calendar recurrence, source snapshots, refresh failure safety, and pushed changes."""

import asyncio
import base64
from datetime import date, datetime, timedelta, timezone
import json
import logging
import sqlite3
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from fastmcp import Client
import httpx
from icalendar import Calendar, Event as ICalEvent
import pytest

from ical_mcp import model, server
from ical_mcp.service import Calendars, Store

START = datetime(2026, 9, 14, tzinfo=timezone.utc)


def event(uid="meeting", start=START + timedelta(hours=9), end=None, **fields):
    """Build a synthetic VEVENT with optional recurrence and source metadata."""
    value = ICalEvent()
    value.add("uid", uid)
    if start is not None:
        value.add("dtstart", start)
    if end is not None:
        value.add("dtend", end)
    value.add("summary", "Planning")
    for name, content in fields.items():
        value.add(name.replace("_", "-"), content)
    return value


def feed(*components, zone=None):
    """Serialize a complete synthetic calendar with an optional floating-time zone."""
    cal = Calendar()
    cal.add("version", "2.0")
    cal.add("prodid", "-//Tests//Calendar//EN")
    if zone:
        cal.add("x-wr-timezone", zone)
    for component in components:
        cal.add_component(component)
    return cal.to_ical()


@pytest.fixture
def settings():
    """A named calendar whose fake token must never appear in tool results or logs."""
    return model.Settings({"Personal": "https://calendar.test/private.ics?token=private-test-token"})


@pytest.fixture
def store(tmp_path):
    """A real per-test persistent database."""
    value = Store(tmp_path / "state" / "ical.sqlite3")
    yield value
    value.close()


@pytest.fixture
async def service(settings, store):
    """A service backed by configurable synthetic HTTP responses and real parsing/storage."""
    backend = {"body": feed(event()), "code": 200, "headers": {"etag": '"v1"'},
               "requests": [], "deliveries": [], "delivery_code": 202}

    def respond(request):
        """Record feed refreshes and event deliveries without external network access."""
        if request.method == "POST":
            backend["deliveries"].append(json.loads(request.content))
            return httpx.Response(backend["delivery_code"])
        backend["requests"].append(request)
        return httpx.Response(backend["code"], content=backend["body"], headers=backend["headers"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        value = Calendars(settings, store, http)
        value.backend = backend
        yield value


@pytest.mark.parametrize("changes", [
    {"urls": {}}, {"urls": []}, {"urls": {"": "https://calendar.test"}},
    {"urls": {"x": "http://calendar.test"}}, {"urls": {"x": "https://u:p@calendar.test"}},
    {"urls": {"x": "https://calendar.test/#fragment"}}, {"urls": {"x": 12}},
    {"timezone_name": "invalid-zone"}, {"refresh_seconds": 0}, {"energy": "invalid"},
])
def test_invalid_calendar_settings(changes):
    """Invalid URLs, zones, and refresh policy fail before any feed access."""
    with pytest.raises(ValueError):
        model.Settings(**{"urls": {"x": "https://calendar.test"}, **changes})


def test_settings_env_and_opaque_ids(settings, monkeypatch):
    """Private URL tokens are excluded from repr and opaque event IDs."""
    monkeypatch.setenv("ICAL_URLS", json.dumps(settings.urls))
    assert model.Settings.from_env() == settings
    assert "private-test-token" not in repr(settings)
    key = model.EventKey(feed=settings.feed_id("Personal"), uid="shared UID / special")
    assert model.EventKey.decode(key.encode()) == key
    assert "private-test-token" not in key.encode()
    for invalid in ["", "?", "x" * 8193, base64.b64encode(b"{}").decode()]:
        with pytest.raises(ValueError, match="event ID"):
            model.EventKey.decode(invalid)


def test_full_event_fields_all_day_and_duration():
    """Dates stay dates, end dates are exclusive, and singular ATTENDEE fields are supported."""
    component = event(start=date(2026, 9, 14), description="Plain <text>", organizer="mailto:owner@example.test",
                      attendee="mailto:guest@example.test", location="Office")
    raw = feed(component)
    _, snapshot = model.parse_feed(raw, "Personal", "feed", ZoneInfo("UTC"))
    item = next(iter(snapshot.values()))["event"]
    assert item["start"] == "2026-09-14" and item["end"] == "2026-09-15" and item["all_day"]
    assert item["attendees"] == ["guest@example.test"] and item["organizer"] == "owner@example.test"
    assert item["body"] == "Plain <text>" and item["location"] == "Office"
    timed = event(duration=timedelta(hours=2))
    parsed = model.render(timed, "Personal", "feed", ZoneInfo("UTC"))
    assert parsed.end - parsed.start == timedelta(hours=2)


@pytest.mark.parametrize("start", [date(2026, 9, 14), START, START + timedelta(hours=9)])
def test_source_event_dates_survive_snapshot_serialization(start):
    """Restoring a snapshot preserves DATE versus DATE-TIME, including midnight events."""
    original = model.render(event(start=start), "Personal", "feed", ZoneInfo("UTC"))
    restored = model.Event.model_validate(original.model_dump(mode="json"))
    assert type(restored.start) is type(original.start)
    assert restored.model_dump(mode="json") == original.model_dump(mode="json")


def test_floating_time_uses_feed_zone_then_configured_fallback():
    """Floating appointments are not silently interpreted as UTC when a zone is declared."""
    floating = event(start=datetime(2026, 9, 14, 9), end=datetime(2026, 9, 14, 10))
    raw = feed(floating, zone="America/New_York")
    _, snapshot = model.parse_feed(raw, "Personal", "feed", ZoneInfo("UTC"))
    assert next(iter(snapshot.values()))["event"]["start"].endswith("-04:00")
    found = model.query_range(raw, "Personal", "feed", ZoneInfo("UTC"),
                             START + timedelta(hours=13), START + timedelta(hours=14), 10)
    assert len(found) == 1 and found[0].start.hour == 9
    fallback = model.render(floating, "Personal", "feed", ZoneInfo("America/Los_Angeles"))
    assert fallback.start.utcoffset() == timedelta(hours=-7)
    with pytest.raises(ValueError, match="timezone"):
        model.parse_feed(feed(floating, zone="invalid"), "Personal", "feed", ZoneInfo("UTC"))


def test_recurrence_exceptions_exdates_dst_and_distinct_ids():
    """The library expands recurrence rules while preserving moved and excluded instances."""
    zone = ZoneInfo("America/New_York")
    begin = datetime(2026, 10, 30, 9, tzinfo=zone)
    base = event(start=begin, end=begin + timedelta(hours=1), rrule={"freq": "daily", "count": 5},
                 exdate=begin + timedelta(days=1))
    moved = event(start=begin + timedelta(days=2, hours=3), end=begin + timedelta(days=2, hours=4),
                  recurrence_id=begin + timedelta(days=2))
    raw = feed(base, moved)
    found = model.query_range(raw, "Personal", "feed", ZoneInfo("UTC"),
                             begin.astimezone(timezone.utc), (begin + timedelta(days=6)).astimezone(timezone.utc), 20)
    assert len(found) == 4 and len({item.id for item in found}) == 4
    assert [item.start.hour for item in found] == [9, 12, 9, 9]
    assert found[0].start.utcoffset() == timedelta(hours=-4)
    assert found[-1].start.utcoffset() == timedelta(hours=-5)


def test_sparse_cancellation_excludes_only_its_occurrence():
    """A cancellation lacking DTSTART still suppresses the indicated recurring instance."""
    base = event(rrule={"freq": "daily", "count": 3})
    cancelled = event(start=None, recurrence_id=START + timedelta(days=1, hours=9), status="CANCELLED")
    raw = feed(base, cancelled)
    model.parse_feed(raw, "Personal", "feed", ZoneInfo("UTC"))
    found = model.query_range(raw, "Personal", "feed", ZoneInfo("UTC"), START, START + timedelta(days=4), 10)
    assert [item.start.day for item in found] == [14, 16]


@pytest.mark.parametrize("raw", [b"not a calendar", b"BEGIN:VEVENT\r\nEND:VEVENT\r\n",
                                    feed(event(start=None)), feed(event(), event()),
                                    feed(event(end=START)),
                                    feed(event(start=date(2026, 9, 14), end=date(2026, 9, 14))),
                                    feed(event(start=date(2026, 9, 14), end=START + timedelta(days=1))),
                                    feed(event(rrule={"freq": "DAILY"})).replace(b"FREQ=DAILY", b"FREQ=INVALID")])
def test_invalid_feed_is_rejected(raw):
    """Malformed/ambiguous calendars and invalid intervals cannot become accepted snapshots."""
    with pytest.raises((ValueError, KeyError)):
        model.parse_feed(raw, "Personal", "feed", ZoneInfo("UTC"))


@pytest.mark.asyncio
async def test_quiet_baseline_conditional_refresh_and_persistence(service, store, tmp_path):
    """A first successful load is quiet, validators are reused, and cached state survives restart."""
    await service.refresh("Personal")
    feed_id = service.settings.feed_id("Personal")
    assert store.pending([feed_id]) == []
    original = store.get(feed_id)
    service.backend.update(code=304, body=b"")
    await service.refresh("Personal")
    assert service.backend["requests"][-1].headers["if-none-match"] == '"v1"'
    assert store.get(feed_id)["raw"] == original["raw"]
    reopened = Store(tmp_path / "state" / "ical.sqlite3")
    try:
        assert reopened.get(feed_id)["raw"] == original["raw"]
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_changes_are_batched_and_metadata_churn_is_quiet(service, store):
    """Source additions/updates/removals notify once, while timestamp-only churn does not."""
    await service.refresh("Personal")
    fid = service.settings.feed_id("Personal")
    metadata_only = event(dtstamp=START, last_modified=START, sequence=2)
    service.backend["body"] = feed(metadata_only)
    await service.refresh("Personal")
    assert store.pending([fid]) == []
    revised = event()
    revised["SUMMARY"] = "Changed meeting"
    service.backend["body"] = feed(revised, *[event(uid=f"new-{index}") for index in range(24)])
    await service.refresh("Personal")
    pending = store.pending([fid])
    assert len(pending) == 2
    changes = [change for _, value in pending for change in json.loads(value["text"])["changes"]]
    assert len(changes) == 25 and {change["change"] for change in changes} == {"added", "updated"}
    assert all(value["energy"] == "passive" and value["source"] == "ical" for _, value in pending)
    await service.refresh("Personal")
    assert store.pending([fid]) == pending
    for identifier, _ in pending:
        store.acknowledge(identifier)
    service.backend["body"] = feed()
    await service.refresh("Personal")
    removed = [change for _, value in store.pending([fid]) for change in json.loads(value["text"])["changes"]]
    assert len(removed) == 25 and all(change["change"] == "removed_from_feed" for change in removed)


@pytest.mark.asyncio
async def test_failed_refresh_keeps_cached_data_and_never_leaks_url(service, store, caplog):
    """HTTP and parser errors expose stale data explicitly without publishing private URLs."""
    await service.refresh("Personal")
    previous = store.get(service.settings.feed_id("Personal"))["raw"]
    for code, body in [(503, b"failure"), (200, b"invalid"), (204, b""), (206, feed(event()))]:
        service.backend.update(code=code, body=body)
        with caplog.at_level(logging.INFO):
            await service.refresh("Personal")
        assert service.status("Personal")["stale"]
        assert store.get(service.settings.feed_id("Personal"))["raw"] == previous
        result = await service.range(START, START + timedelta(days=1), None, 10)
        assert result["events"][0]["stale"] and result["calendars"][0]["error"]
    assert "private-test-token" not in caplog.text
    assert "private-test-token" not in json.dumps(service.status("Personal"))


@pytest.mark.asyncio
async def test_first_failure_does_not_become_empty_calendar_or_noisy_baseline(service, store):
    """Unavailable feeds raise explicitly; first eventual success still establishes a quiet baseline."""
    service.backend.update(code=304, body=b"")
    await service.refresh("Personal")
    with pytest.raises(RuntimeError):
        await service.range(START, START + timedelta(days=1), None, 10)
    service.backend.update(code=200, body=feed(event()))
    await service.refresh("Personal")
    assert store.pending([service.settings.feed_id("Personal")]) == []


@pytest.mark.asyncio
async def test_feed_size_limit_and_last_modified_header(settings, store):
    """Oversized feeds are not committed and Last-Modified is sent on subsequent refreshes."""
    headers = []

    def endpoint(request):
        """Serve a fixed synthetic complete feed with a modification validator."""
        headers.append(request.headers)
        return httpx.Response(200, content=feed(event()), headers={"last-modified": "Mon, 14 Sep 2026 09:00:00 GMT"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as http:
        service = Calendars(settings, store, http)
        await service.refresh("Personal")
        await service.refresh("Personal")
        assert headers[-1]["if-modified-since"] == "Mon, 14 Sep 2026 09:00:00 GMT"
        tiny = model.Settings(settings.urls, max_feed_bytes=10)
        await Calendars(tiny, store, http).refresh("Personal")
        assert store.get(settings.feed_id("Personal"))["error"] == "ValueError"


@pytest.mark.asyncio
async def test_outbox_retries_same_event_and_isolates_changed_feed(service, store):
    """HTTP errors retain a stable event ID; other feed identities cannot drain it."""
    await service.refresh("Personal")
    service.backend["body"] = feed(event(), event("new"))
    await service.refresh("Personal")
    fid = service.settings.feed_id("Personal")
    service.backend["delivery_code"] = 500
    with pytest.raises(httpx.HTTPStatusError):
        await service.deliver_once("http://session.test/")
    assert len(store.pending([fid])) == 1 and store.pending(["different-feed"]) == []
    service.backend["delivery_code"] = 202
    assert await service.deliver_once("http://session.test") == 1
    assert service.backend["deliveries"][0] == service.backend["deliveries"][1]
    assert store.pending([fid]) == []


@pytest.mark.asyncio
async def test_snapshot_and_outbox_are_atomic_on_storage_failure(service, store):
    """A failed enqueue does not advance the source snapshot and lose its change notification."""
    await service.refresh("Personal")
    fid = service.settings.feed_id("Personal")
    before = store.get(fid)["raw"]
    store.db.execute("CREATE TRIGGER refuse_event BEFORE INSERT ON outbox BEGIN SELECT RAISE(ABORT, 'test'); END")
    service.backend["body"] = feed(event(), event("new"))
    await service.refresh("Personal")
    assert store.get(fid)["raw"] == before and store.pending([fid]) == []
    store.db.execute("DROP TRIGGER refuse_event")
    await service.refresh("Personal")
    assert len(store.pending([fid])) == 1


@pytest.mark.asyncio
async def test_range_truncation_occurrence_lookup_and_source_lookup(service, store):
    """Repeated occurrences have distinct retrievable IDs and ranges expose truncation."""
    service.backend["body"] = feed(event(rrule={"freq": "daily", "count": 10}))
    result = await service.range(START, START + timedelta(days=15), None, 3)
    assert result["truncated"] and len(result["events"]) == 3
    assert len({item["id"] for item in result["events"]}) == 3
    for item in result["events"]:
        loaded = await service.get(item["id"])
        assert loaded.model_dump(mode="json") == item
    snapshot = json.loads(store.get(service.settings.feed_id("Personal"))["snapshot"])
    source = await service.get(next(iter(snapshot)))
    assert source.recurrence_rule == "FREQ=DAILY;COUNT=10"
    with pytest.raises(ValueError, match="unconfigured"):
        await service.get(model.EventKey(feed="different", uid="meeting").encode())
    service.backend["body"] = feed()
    await service.refresh("Personal")
    with pytest.raises(KeyError):
        await service.get(source.id)
    with pytest.raises(KeyError):
        await service.get(result["events"][0]["id"])


@pytest.mark.asyncio
async def test_all_day_occurrence_can_be_retrieved(service):
    """All-day IDs round-trip as dates rather than midnight timestamps."""
    service.backend["body"] = feed(event(start=date(2026, 9, 14)), zone="America/New_York")
    result = await service.range(START, START + timedelta(days=2), None, 10)
    assert len(result["events"]) == 1 and result["events"][0]["all_day"]
    loaded = await service.get(result["events"][0]["id"])
    assert loaded.model_dump(mode="json") == result["events"][0]


@pytest.mark.parametrize("start,end,limit", [(START.replace(tzinfo=None), START, 10),
                                           (START, START, 10), (START, START - timedelta(days=1), 10),
                                           (START, START + timedelta(days=367), 10),
                                           (START, START + timedelta(days=1), 0)])
@pytest.mark.asyncio
async def test_invalid_ranges_fail_before_feed_requests(service, start, end, limit):
    """Caller date and size bounds are validated before network work."""
    with pytest.raises(ValueError):
        await service.range(start, end, None, limit)
    assert service.backend["requests"] == []


@pytest.mark.asyncio
async def test_multiple_calendars_do_not_collide_on_shared_uids(settings, store):
    """Results are globally start-sorted and identical provider UIDs remain calendar-scoped."""
    config = model.Settings({"Later": "https://calendar.test/later", "Earlier": "https://calendar.test/earlier"})

    def endpoint(request):
        """Provide the same UID at different times in independently named feeds."""
        hour = 10 if request.url.path == "/later" else 8
        return httpx.Response(200, content=feed(event(start=START + timedelta(hours=hour))))

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as http:
        service = Calendars(config, store, http)
        result = await service.range(START, START + timedelta(days=1), None, 10)
        assert [item["calendar"] for item in result["events"]] == ["Earlier", "Later"]
        assert len({item["id"] for item in result["events"]}) == 2
        with pytest.raises(KeyError):
            await service.range(START, START + timedelta(days=1), "missing", 10)


@pytest.mark.asyncio
async def test_explicit_routing_and_cancellation_change(settings, store):
    """Confirmed source cancellation is distinct from removal and preserves configured routing."""
    config = model.Settings(settings.urls, session_id="calendar-session", energy="active")
    data = {"body": feed(event())}
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=data["body"]))) as http:
        service = Calendars(config, store, http)
        await service.refresh("Personal")
        data["body"] = feed(event(status="CANCELLED"))
        await service.refresh("Personal")
        pending = store.pending([config.feed_id("Personal")])
        assert len(pending) == 1
        sent = pending[0][1]
        assert sent["session_id"] == "calendar-session" and sent["energy"] == "active"
        assert json.loads(sent["text"])["changes"][0]["change"] == "cancelled"
        assert (await service.range(START, START + timedelta(days=1), None, 10))["events"] == []


@pytest.mark.asyncio
async def test_refresh_has_an_overall_download_timeout(settings, store):
    """A feed that trickles indefinitely cannot hold the refresh operation forever."""
    class SlowFeed(httpx.AsyncByteStream):
        """Synthetic response whose body never arrives within the configured deadline."""
        async def __aiter__(self):
            """Yield a valid feed only after an intentionally excessive delay."""
            await asyncio.sleep(1)
            yield feed(event())

    config = model.Settings(settings.urls, timeout_seconds=0.01)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=SlowFeed()))) as http:
        service = Calendars(config, store, http)
        await service.refresh("Personal")
        assert service.status("Personal")["error"] == "TimeoutError"
        assert not service.status("Personal")["available"]


@pytest.mark.asyncio
async def test_refresh_and_delivery_loops_retry_and_shutdown(service, monkeypatch):
    """Background loops retry transient errors and release promptly when stopped."""
    refreshes = []
    original_refresh = service.refresh

    async def refresh(name):
        """Stop after the refresh loop has completed a timeout-and-retry cycle."""
        refreshes.append(name)
        await original_refresh(name)

    waits = []

    async def wait():
        """First interval elapses; the next interval receives the shutdown signal."""
        waits.append(True)
        if len(waits) == 1:
            raise TimeoutError
        service.stop.set()
        return True

    monkeypatch.setattr(service, "refresh", refresh)
    monkeypatch.setattr(service.stop, "wait", wait)
    await service.refresh_loop()
    assert len(refreshes) == 2
    service.stop.clear()
    waits.clear()
    attempts = AsyncMock(side_effect=[httpx.ConnectError("unavailable"), 1, 0])
    monkeypatch.setattr(service, "deliver_once", attempts)
    await service.delivery_loop("http://session.test")
    assert attempts.await_count == 3


@pytest.mark.asyncio
async def test_stop_between_calendar_refreshes_skips_remaining_feeds(settings, store):
    """Shutdown does not initiate another network request for the next configured calendar."""
    config = model.Settings({"first": "https://calendar.test/first", "second": "https://calendar.test/second"})
    async with httpx.AsyncClient() as http:
        service = Calendars(config, store, http)

        async def stop_after_refresh(name):
            """Request shutdown immediately after the first feed."""
            assert name == "first"
            service.stop.set()

        service.refresh = stop_after_refresh
        await service.refresh_loop()


@pytest.mark.asyncio
async def test_mcp_lifespan_and_calendar_tools(settings, monkeypatch, tmp_path):
    """Real MCP discovery, range/get calls, and status run inside the managed service lifecycle."""
    original_client = httpx.AsyncClient
    monkeypatch.setattr(server.Settings, "from_env", lambda: settings)
    monkeypatch.setenv("ICAL_STATE_PATH", str(tmp_path / "lifecycle.sqlite3"))
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=feed(event()))), **kwargs))
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert set(tools) == {"list_calendars", "get_event", "range_events"}
        assert all(tool.annotations.readOnlyHint for tool in tools.values())
        result = await client.call_tool("range_events", {"start_date_time": START.isoformat(),
            "end_date_time": (START + timedelta(days=1)).isoformat()})
        item = result.structured_content["events"][0]
        detail = await client.call_tool("get_event", {"event_id": item["id"]})
        assert detail.structured_content["id"] == item["id"]
        status = await client.call_tool("list_calendars", {})
        assert "private-test-token" not in str(status)
    assert server.calendars is None
    with pytest.raises(RuntimeError):
        await server.list_calendars()


@pytest.mark.asyncio
async def test_time_passing_alone_does_not_generate_calendar_changes(service, store):
    """Expanding a new query window is not a source update and cannot create change events."""
    service.backend["body"] = feed(event(rrule={"freq": "daily"}))
    await service.refresh("Personal")
    for day in range(3):
        result = await service.range(START + timedelta(days=day), START + timedelta(days=day + 1), None, 10)
        assert len(result["events"]) == 1
    assert store.pending([service.settings.feed_id("Personal")]) == []


@pytest.mark.asyncio
async def test_half_open_range_and_long_event_deduplication(service):
    """Multi-day events appear once; an event starting at the exclusive end is omitted."""
    service.backend["body"] = feed(event("long", START - timedelta(days=1), START + timedelta(days=4)),
                                   event("boundary", START + timedelta(days=3)))
    result = await service.range(START, START + timedelta(days=3), None, 10)
    assert [item["uid"] for item in result["events"]] == ["long"]
