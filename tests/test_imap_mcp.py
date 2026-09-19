"""Read-only mail, mailbox epoch safety, and durable IDLE event delivery."""

import asyncio
import base64
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from imap_mcp import events, model, server


@pytest.fixture
def settings():
    """Synthetic account configuration; no real credentials or mail servers are used."""
    return model.Settings("imaps://mail.test", "test-user", "test-password", notify_since="all")


@pytest.fixture
def key(settings):
    """A stable synthetic message identity."""
    return model.MessageKey(account=settings.account, folder="Projects:Mixed Case/Work", validity=21, uid=7)


@pytest.fixture
def mime():
    """An email with distinct recipients, plain/HTML alternatives, and binary content."""
    mail = EmailMessage()
    mail["Subject"] = "Status update"
    mail["From"] = "Sender <sender@example.test>"
    mail["To"] = "One <one@example.test>, Two <two@example.test>"
    mail["Cc"] = "Copy <copy@example.test>"
    mail["Date"] = "Mon, 14 Sep 2026 09:00:00 -0400"
    mail.set_content("Plain text with <literal> content")
    mail.add_alternative("<p>HTML version</p>", subtype="html")
    mail.add_attachment(b"\x00\xffbinary", maintype="application", subtype="octet-stream", filename="report.bin")
    return mail.as_bytes()


@pytest.fixture
def client(mime):
    """An IMAPClient-shaped mock with independent header and full-body responses."""
    result = Mock()
    result.has_capability.return_value = False
    result.select_folder.return_value = {b"UIDVALIDITY": 21, b"UIDNEXT": 8}
    result.list_folders.return_value = [((), b"/", "Projects:Mixed Case/Work"),
                                      ((b"\\Noselect",), b"/", "Container")]
    result.search.return_value = [7]

    def fetch(uids, fields):
        """Return requested records keyed by UID, not sequence number."""
        return {uid: {b"RFC822.SIZE": len(mime), b"BODY[]": mime, b"BODY[HEADER]": mime.split(b"\n\n", 1)[0],
                      b"INTERNALDATE": datetime(2026, 9, 14, 13, tzinfo=timezone.utc)}
                for uid in uids}

    result.fetch.side_effect = fetch
    return result


@pytest.fixture
def mailbox(settings, client, monkeypatch):
    """Mailbox whose connection factory uses a synthetic read-only protocol client."""
    box = model.Mailbox(settings)

    @contextmanager
    def connect():
        """Yield the isolated client without network access."""
        yield client

    monkeypatch.setattr(box, "connect", connect)
    yield box
    box.close()


@pytest.fixture
def store(tmp_path):
    """A real SQLite event store with per-test lifetime."""
    state = events.EventStore(tmp_path / "state" / "imap.sqlite3")
    yield state
    state.close()


def test_message_ids_round_trip_folder_names_and_account(key, settings):
    """IDs survive separators in folder names and distinguish accounts and epochs."""
    assert model.MessageKey.decode(key.encode()) == key
    assert "/" not in key.encode()
    assert settings.account == model.Settings(settings.server, settings.username, "rotated").account
    assert settings.account != model.Settings(settings.server, "different", "secret").account
    assert "test-password" not in repr(settings)


@pytest.mark.parametrize("value", ["", "bad id", "x" * 4097, "e30", base64.b64encode(b"null").decode()])
def test_invalid_message_ids_fail(value):
    """Malformed or incomplete IDs never become mailbox commands."""
    with pytest.raises(ValueError, match="message ID"):
        model.MessageKey.decode(value)


@pytest.mark.parametrize("changes", [
    {"server": "http://mail.test"}, {"server": "imaps://u:p@mail.test"},
    {"server": "imaps://mail.test/path"}, {"server": "imaps://mail.test:0"},
    {"username": ""}, {"password": ""}, {"folders": ()},
    {"folders": ("INBOX", "INBOX")}, {"folders": ("bad\nfolder",)}, {"timeout": 0},
    {"max_message_bytes": 0},
])
def test_account_validation_precedes_network(changes):
    """Unsafe or incomplete settings fail without attempting authentication."""
    with pytest.raises(ValueError):
        model.Settings(**{"server": "imaps://mail.test", "username": "u", "password": "p", **changes})


def test_environment_configuration(monkeypatch):
    """Account settings parse explicit JSON folders and finite connection limits."""
    for name, value in {"IMAP_SERVER": "starttls://mail.test:1143", "IMAP_USERNAME": "user",
                        "IMAP_PASSWORD": "secret", "IMAP_FOLDERS": '["INBOX", "Mixed Case"]'}.items():
        monkeypatch.setenv(name, value)
    configured = model.Settings.from_env()
    assert configured.folders == ("INBOX", "Mixed Case")
    monkeypatch.setenv("IMAP_FOLDERS", '"INBOX"')
    with pytest.raises(ValueError):
        model.Settings.from_env()


@pytest.mark.parametrize("scheme,port", [("imaps", 993), ("tls", 993), ("starttls", 143), ("imap", 143)])
def test_connection_uses_tls_before_login(settings, monkeypatch, scheme, port):
    """Every authentication uses verified TLS and bounded sockets, then closes the connection."""
    protocol = Mock()
    protocol.__enter__ = Mock(return_value=protocol)
    protocol.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=protocol)
    monkeypatch.setattr(model, "IMAPClient", factory)
    configured = model.Settings(f"{scheme}://mail.test", "user", "secret")
    box = model.Mailbox(configured)
    try:
        with box.connect() as connected:
            assert connected is protocol
    finally:
        box.close()
    assert factory.call_args.kwargs["port"] == port
    assert factory.call_args.kwargs["use_uid"] is True
    assert factory.call_args.kwargs["timeout"] == 20
    if scheme == "starttls":
        assert [call[0] for call in protocol.method_calls][:2] == ["starttls", "login"]
    else:
        protocol.starttls.assert_not_called()
        assert factory.call_args.kwargs["ssl"] is (scheme != "imap")
    assert protocol.normalise_times is False
    protocol.__exit__.assert_called_once()


def test_mime_details_and_summaries(mime, key):
    """Plain text is not mangled as HTML; attachment sizes reflect decoded bytes."""
    detail = model.parse_message(mime, key)
    assert detail.body.strip() == "Plain text with <literal> content"
    assert len(detail.to) == 2 and len(detail.cc) == 1
    assert detail.date_time.utcoffset().total_seconds() == -4 * 3600
    assert detail.has_attachments
    assert detail.attachment_metadata[0].size == len(b"\x00\xffbinary")
    assert detail.attachment_metadata[0].uri.startswith("imap://attachments/")
    summary = model.parse_message(mime, key, summary=True)
    assert summary.body is summary.to is summary.has_attachments is None
    assert summary.subject == detail.subject


def test_missing_headers_html_and_attached_message(key):
    """Missing headers, HTML-only bodies, and attached emails remain readable."""
    empty = model.parse_message(b"Date: malformed\n\n", key)
    assert empty.subject == "" and empty.date_time is None and empty.body == ""
    html = model.parse_message(b"Content-Type: text/html\n\n<p>Hello <b>there</b></p>", key)
    assert "**there**" in html.body
    part = EmailMessage()
    nested = EmailMessage()
    nested.set_content("Nested message")
    part.set_type("message/rfc822")
    part.set_payload([nested])
    assert b"Nested message" in model.attachment_bytes(part)
    blank = EmailMessage()
    assert model.attachment_bytes(blank) == b""


def test_mail_tools_are_readonly_and_keep_folder_case(mailbox, client, key):
    """Read and search use UID fetches, BODY.PEEK, and read-only selection."""
    details = mailbox.get(key.encode())
    assert details.subject == "Status update"
    assert details.folder == key.folder
    assert details.received_at == datetime(2026, 9, 14, 13, tzinfo=timezone.utc)
    data, metadata = mailbox.attachment(key.encode(), "0")
    assert data == b"\x00\xffbinary" and metadata.name == "report.bin"
    with pytest.raises(KeyError):
        mailbox.attachment(key.encode(), "99")
    results = mailbox.search("ALL", key.folder)
    assert results[0].id == key.encode() and results[0].has_attachments is None
    assert mailbox.folders() == [key.folder]
    assert mailbox.search("ALL", None)[0].id == key.encode()
    for call in client.select_folder.call_args_list:
        assert call.args == (key.folder,) and call.kwargs == {"readonly": True}
    assert all("RFC822" not in call.args[1] for call in client.fetch.call_args_list)
    assert all(field in {"RFC822.SIZE", "INTERNALDATE"} or field.startswith("BODY.PEEK")
               for call in client.fetch.call_args_list for field in call.args[1])


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_search_limits_are_validated_before_connect(mailbox, client, limit):
    """Unbounded searches are rejected without protocol requests."""
    with pytest.raises(ValueError):
        mailbox.search("ALL", limit=limit)
    client.select_folder.assert_not_called()


@pytest.mark.parametrize("query", ["", "ALL\r\nLOGOUT", "ALL\x00"])
def test_search_control_characters_are_rejected(mailbox, query):
    """Search expressions cannot introduce another protocol command."""
    with pytest.raises(ValueError):
        mailbox.search(query)


def test_search_is_bounded_newest_first_and_handles_empty_results(mailbox, client):
    """Limits apply across folders and empty search results do not cause empty FETCH commands."""
    client.search.return_value = [1, 3, 2]
    result = mailbox.search("ALL", limit=2)
    assert [model.MessageKey.decode(item.id).uid for item in result] == [3, 2]
    client.search.return_value = []
    client.fetch.reset_mock()
    assert mailbox.search("ALL") == []
    client.fetch.assert_not_called()


def _search_folders(client, mime, folders):
    """Configure read-only folder records keyed by UID with independently controlled receipt dates."""
    current = {}
    client.list_folders.return_value = [((), b"/", name) for name in folders]

    def select(name, readonly=False):
        """Select only a known fake folder and preserve the readonly contract."""
        assert readonly is True
        current["folder"] = name
        return {b"UIDVALIDITY": 21, b"UIDNEXT": max(folders[name], default=0) + 1}

    def fetch(uids, fields):
        """Return only requested metadata or headers, never full bodies during search."""
        assert set(fields) <= {"INTERNALDATE", "BODY.PEEK[HEADER]"}
        return {uid: {b"INTERNALDATE": folders[current["folder"]][uid],
                      b"BODY[HEADER]": mime.split(b"\n\n", 1)[0]}
                for uid in uids if uid in folders[current["folder"]]}

    client.select_folder.side_effect = select
    client.search.side_effect = lambda criteria: list(folders[current["folder"]])
    client.fetch.side_effect = fetch
    return current


def test_search_defaults_to_all_selectable_folders(mailbox, client, mime):
    """An omitted folder must not hide archived mail from an agent's search."""
    _search_folders(client, mime, {
        "INBOX": {}, "Archive": {434: datetime(2026, 8, 28, tzinfo=timezone.utc)}})
    result = mailbox.search('SUBJECT "Weekly Update"')
    assert len(result) == 1
    assert model.MessageKey.decode(result[0].id).folder == "Archive"


def test_search_orders_by_receipt_date_before_applying_limit(mailbox, client, mime):
    """Imported high UIDs from old years must not push recent low UIDs below the result cap."""
    _search_folders(client, mime, {"Archive": {
        434: datetime(2026, 8, 28, tzinfo=timezone.utc),
        100001: datetime(2025, 8, 28, tzinfo=timezone.utc),
        100002: datetime(2024, 8, 28, tzinfo=timezone.utc),
    }})
    result = mailbox.search('SUBJECT "Weekly Update"', "Archive", limit=1)
    assert [model.MessageKey.decode(item.id).uid for item in result] == [434]


def test_all_folder_search_chooses_globally_newest_matches(mailbox, client, mime):
    """Folder enumeration order cannot consume the limit before a later folder is searched."""
    _search_folders(client, mime, {
        "First": {999999: datetime(2024, 1, 1, tzinfo=timezone.utc)},
        "Archive": {434: datetime(2026, 8, 28, tzinfo=timezone.utc)},
        "INBOX": {3: datetime(2026, 9, 16, tzinfo=timezone.utc)},
    })
    result = mailbox.search("ALL", None, limit=2)
    assert [(model.MessageKey.decode(item.id).folder, model.MessageKey.decode(item.id).uid)
            for item in result] == [("INBOX", 3), ("Archive", 434)]


def test_explicit_folder_search_does_not_expand_scope(mailbox, client, mime):
    """Callers can still restrict a search to INBOX rather than the new all-folder default."""
    _search_folders(client, mime, {
        "Archive": {434: datetime(2026, 9, 16, tzinfo=timezone.utc)},
        "INBOX": {3: datetime(2026, 8, 1, tzinfo=timezone.utc)},
    })
    result = mailbox.search("ALL", "INBOX")
    assert [(item.folder, model.MessageKey.decode(item.id).uid) for item in result] == [("INBOX", 3)]
    client.list_folders.assert_not_called()


def test_search_normalizes_receipt_timezones_and_sorts_unknown_dates_last(mailbox, client, mime):
    """Receipt instants determine order; sender dates, naive values, and large UIDs cannot override it."""
    _search_folders(client, mime, {"Archive": {
        1: datetime(2026, 9, 16, 1, tzinfo=timezone(timedelta(hours=2))),
        2: datetime(2026, 9, 15, 23, 30, tzinfo=timezone.utc),
        1000001: None,
        1000002: "not-a-date",
        1000003: datetime(2026, 9, 17),
    }})
    result = mailbox.search("ALL", limit=5)
    assert [model.MessageKey.decode(item.id).uid for item in result] == [2, 1, 1000003, 1000002, 1000001]
    assert result[0].received_at == datetime(2026, 9, 15, 23, 30, tzinfo=timezone.utc)
    assert result[1].received_at == datetime(2026, 9, 15, 23, tzinfo=timezone.utc)
    assert all(item.received_at is None for item in result[2:])
    assert result[0].date_time == datetime(2026, 9, 14, 9, tzinfo=timezone(timedelta(hours=-4)))


def test_search_dates_are_batched_and_only_winning_headers_are_fetched(mailbox, client, mime):
    """Global newest selection considers every match but bounds each request and retained headers."""
    count = model.DATE_FETCH_BATCH_SIZE * 2 + 3
    records = {uid: datetime(2025, 1, 1, tzinfo=timezone.utc) for uid in range(1, count + 1)}
    records[7] = datetime(2026, 9, 16, tzinfo=timezone.utc)
    _search_folders(client, mime, {"Archive": records})
    result = mailbox.search("ALL", limit=3)
    assert model.MessageKey.decode(result[0].id).uid == 7
    date_calls = [call for call in client.fetch.call_args_list if call.args[1] == ["INTERNALDATE"]]
    header_calls = [call for call in client.fetch.call_args_list if call.args[1] == ["BODY.PEEK[HEADER]"]]
    assert [len(call.args[0]) for call in date_calls] == [model.DATE_FETCH_BATCH_SIZE, model.DATE_FETCH_BATCH_SIZE, 3]
    assert sum(len(call.args[0]) for call in header_calls) == 3
    assert all(item.body is None and item.attachment_metadata is None for item in result)


def test_search_skips_messages_expunged_during_metadata_or_header_reads(mailbox, client, mime):
    """Concurrent removals cannot fabricate messages or prevent the remaining results from returning."""
    _search_folders(client, mime, {"Archive": {
        1: datetime(2026, 9, 14, tzinfo=timezone.utc),
        2: datetime(2026, 9, 15, tzinfo=timezone.utc),
        3: datetime(2026, 9, 16, tzinfo=timezone.utc),
    }})
    fetch = client.fetch.side_effect

    def disappearing(uids, fields):
        """Remove one UID before dates are read, and another before headers are read."""
        values = fetch(uids, fields)
        values.pop(2 if fields == ["INTERNALDATE"] else 3, None)
        return values

    client.fetch.side_effect = disappearing
    result = mailbox.search("ALL", limit=3)
    assert [model.MessageKey.decode(item.id).uid for item in result] == [1]


def test_search_rejects_uidvalidity_change_before_fetching_headers(mailbox, client, mime):
    """Re-selecting a reset folder must not bind an old UID to an unrelated new message."""
    _search_folders(client, mime, {"Archive": {1: datetime(2026, 9, 16, tzinfo=timezone.utc)}})
    select = client.select_folder.side_effect
    calls = []

    def reset(name, readonly=False):
        """Change epochs on the second selection, after metadata ranking has completed."""
        calls.append(name)
        selected = select(name, readonly)
        selected[b"UIDVALIDITY"] += len(calls) - 1
        return selected

    client.select_folder.side_effect = reset
    with pytest.raises(RuntimeError, match="Mailbox changed"):
        mailbox.search("ALL")
    assert all(call.args[1] == ["INTERNALDATE"] for call in client.fetch.call_args_list)


def test_detail_rejects_foreign_stale_missing_and_oversized_ids(mailbox, client, key):
    """A reused UID cannot retrieve the wrong message after an epoch/account change."""
    foreign = key.model_copy(update={"account": "foreign"})
    with pytest.raises(ValueError, match="account"):
        mailbox.get(foreign.encode())
    client.select_folder.return_value[b"UIDVALIDITY"] = 22
    with pytest.raises(ValueError, match="UIDVALIDITY"):
        mailbox.get(key.encode())
    client.select_folder.return_value[b"UIDVALIDITY"] = 21
    client.fetch.side_effect = None
    client.fetch.return_value = {}
    with pytest.raises(KeyError):
        mailbox.get(key.encode())
    client.fetch.return_value = {7: {b"RFC822.SIZE": 100_000_000}}
    with pytest.raises(ValueError, match="exceeds"):
        mailbox.get(key.encode())
    client.fetch.side_effect = [{7: {b"RFC822.SIZE": 1}}, {}]
    with pytest.raises(KeyError):
        mailbox.get(key.encode())
    client.fetch.side_effect = [{7: {b"RFC822.SIZE": 1}}, {7: {b"BODY[]": b"x" * 26_214_401}}]
    with pytest.raises(ValueError, match="exceeds"):
        mailbox.get(key.encode())


def test_initial_baseline_catchup_and_persistent_outbox(mailbox, client, store, tmp_path):
    """First startup is quiet; future UIDs are staged exactly once and survive reopening."""
    monitor = events.Monitor(mailbox, store)
    monitor.scan(client, "INBOX", 21, 7)
    assert store.pending(mailbox.settings.account) == []
    client.search.return_value = [7, 9, 8]
    monitor.scan(client, "INBOX", 21, 7)
    monitor.scan(client, "INBOX", 21, 99)
    pending = store.pending(mailbox.settings.account)
    assert [event["metadata"]["uid"] for _, event in pending] == [8, 9]
    event = pending[0][1]
    assert event["source"] == "imap" and event["event_type"] == "mail_received"
    assert event["energy"] == "passive" and "session_id" not in event
    assert "body" not in json.loads(event["text"])
    reopened = events.EventStore(tmp_path / "state" / "imap.sqlite3")
    try:
        assert reopened.pending(mailbox.settings.account) == pending
        assert reopened.checkpoint(mailbox.settings.account, "INBOX", 21, 99) == 9
        assert reopened.pending("different-account") == []
    finally:
        reopened.close()


def test_epoch_reset_expunges_and_staging_deduplication(mailbox, client, store):
    """A new UIDVALIDITY starts a fresh baseline, while old queued events stay identifiable."""
    monitor = events.Monitor(mailbox, store, session_id="mail-session", energy="active")
    store.checkpoint(mailbox.settings.account, "INBOX", 21, 0)
    monitor.scan(client, "INBOX", 21, 7)
    pending = store.pending(mailbox.settings.account)
    assert pending[0][1]["session_id"] == "mail-session"
    assert pending[0][1]["energy"] == "active"
    old_key = model.MessageKey.decode(pending[0][0])
    store.stage(old_key, pending[0][1])
    assert len(store.pending(mailbox.settings.account)) == 1
    assert store.checkpoint(mailbox.settings.account, "INBOX", 22, 40) == 40
    with pytest.raises(RuntimeError):
        store.stage(old_key, {})
    assert len(store.pending(mailbox.settings.account)) == 1
    client.search.return_value = [41]
    client.fetch.side_effect = None
    client.fetch.return_value = {}
    monitor.scan(client, "INBOX", 22, 40)
    assert store.checkpoint(mailbox.settings.account, "INBOX", 22, 0) == 41
    assert len(store.pending(mailbox.settings.account)) == 1


def test_scan_failure_does_not_skip_unstaged_uids(mailbox, client, store):
    """After a fetch failure, reconnect resumes at the failed UID rather than the next one."""
    monitor = events.Monitor(mailbox, store)
    store.checkpoint(mailbox.settings.account, "INBOX", 21, 0)
    client.search.return_value = [1, 2, 3]
    original = client.fetch.side_effect

    def fetch(uids, fields):
        """Fail on the second message of a synthetic arrival burst."""
        if uids == [2]:
            raise OSError("Disconnected")
        return original(uids, fields)

    client.fetch.side_effect = fetch
    with pytest.raises(OSError):
        monitor.scan(client, "INBOX", 21, 0)
    assert store.checkpoint(mailbox.settings.account, "INBOX", 21, 0) == 1
    client.fetch.side_effect = original
    monitor.scan(client, "INBOX", 21, 0)
    assert [event["metadata"]["uid"] for _, event in store.pending(mailbox.settings.account)] == [1, 2, 3]


@pytest.mark.asyncio
async def test_http_delivery_retries_errors_and_acknowledges_acceptance(mailbox, client, store):
    """Failed HTTP responses retain the event, and retries carry the same stable event ID."""
    monitor = events.Monitor(mailbox, store)
    store.checkpoint(mailbox.settings.account, "INBOX", 21, 0)
    monitor.scan(client, "INBOX", 21, 0)
    received = []

    def endpoint(request):
        """First attempt fails; the next is accepted by a synthetic session endpoint."""
        assert request.url.path == "/event"
        received.append(json.loads(request.content))
        return httpx.Response(503 if len(received) == 1 else 202)

    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await monitor.deliver_once(http, "http://session.test/")
        assert len(store.pending(mailbox.settings.account)) == 1
        assert await monitor.deliver_once(http, "http://session.test") == 1
    assert received[0] == received[1]
    assert store.pending(mailbox.settings.account) == []


def test_idle_stops_before_fetching_and_polling_is_only_a_fallback(mailbox, client, store):
    """No IMAP command runs while IDLE is active, and shutdown remains interruptible."""
    monitor = events.Monitor(mailbox, store)
    client.idle_check.return_value = [(1, b"EXISTS")]
    monitor.wait_for_change(client, True)
    assert [call[0] for call in client.method_calls] == ["idle", "idle_check", "idle_done"]
    client.idle_check.assert_called_once_with(timeout=5)
    monitor.stop = Mock()
    monitor.stop.is_set.return_value = False
    monitor.wait_for_change(client, False)
    monitor.stop.wait.assert_called_once_with(60)
    client.noop.assert_called_once()
    monitor.stop.is_set.return_value = True
    client.noop.reset_mock()
    monitor.wait_for_change(client, False)
    client.noop.assert_not_called()


def test_idle_errors_exit_idle_and_watcher_reconnects(mailbox, client, store, monkeypatch):
    """A failed connection is retried; IDLE errors do not leave the client in IDLE mode."""
    monitor = events.Monitor(mailbox, store)
    client.idle_check.side_effect = OSError("closed")
    with pytest.raises(OSError):
        monitor.wait_for_change(client, True)
    client.idle_done.assert_called_once()
    monitor.scan = Mock(side_effect=OSError("disconnected"))
    monitor.stop = Mock()
    monitor.stop.is_set.side_effect = [False, False, True]
    monitor.watch("INBOX")
    monitor.stop.wait.assert_called_once_with(5)
    client.select_folder.assert_called_with("INBOX", readonly=True)


@pytest.mark.parametrize("kwargs", [{"energy": "wrong"}, {"poll_seconds": 0}, {"retry_seconds": -1}])
def test_invalid_monitor_settings(mailbox, store, kwargs):
    """Invalid routing energy or intervals fail before background workers start."""
    with pytest.raises(ValueError):
        events.Monitor(mailbox, store, **kwargs)


def test_attachment_tools_return_portable_binary_resources(mailbox, key, monkeypatch):
    """Attachments are embedded resources, never paths built from untrusted email filenames."""
    monkeypatch.setattr(server, "mailbox", mailbox)
    result = server.get_attachment(key.encode(), "0")
    assert base64.b64decode(result.resource.blob) == b"\x00\xffbinary"
    resource = server.attachment_resource(key.encode(), "0").contents[0]
    assert resource.content == b"\x00\xffbinary"
    assert resource.mime_type == "application/octet-stream"
    assert server.get_mail(key.encode()).subject == "Status update"
    assert server.search_mail("ALL", key.folder)[0].id == key.encode()
    assert server.list_folders() == [key.folder]
    monkeypatch.setattr(server, "mailbox", None)
    with pytest.raises(RuntimeError):
        server.list_folders()


@pytest.mark.asyncio
async def test_mcp_lifespan_tools_resources_and_worker_shutdown(mailbox, key, monkeypatch, tmp_path):
    """The real MCP lifecycle starts/stops watchers and exposes portable binary resources."""
    from fastmcp import Client

    monkeypatch.setattr(server.Settings, "from_env", lambda: mailbox.settings)
    monkeypatch.setattr(server, "Mailbox", lambda settings, state_path=None: mailbox)
    monkeypatch.setenv("IMAP_STATE_PATH", str(tmp_path / "lifespan.sqlite3"))
    stopped = []

    def wait_until_shutdown(self, client, idle):
        """Simulate an idle server, releasing its watcher on lifespan shutdown."""
        self.stop.wait(5)
        stopped.append(self.stop.is_set())

    monkeypatch.setattr(events.Monitor, "wait_for_change", wait_until_shutdown)
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert {"get_mail", "search_mail", "get_attachment", "list_folders"} == set(tools)
        assert all(tool.annotations.readOnlyHint for tool in tools.values())
        assert tools["search_mail"].inputSchema["properties"]["folder"]["default"] is None
        search = await client.call_tool("search_mail", {"search": "ALL", "limit": 1})
        summary = search.structured_content["result"][0]
        assert summary["folder"] == key.folder
        assert summary["received_at"] == "2026-09-14T13:00:00Z"
        details = await client.call_tool("get_mail", {"message_id": key.encode()})
        assert not details.is_error
        binary = await client.call_tool("get_attachment", {"message_id": key.encode(), "attachment_id": "0"})
        resources = [block for block in binary.content if block.type == "resource"]
        assert base64.b64decode(resources[0].resource.blob) == b"\x00\xffbinary"
        contents = await client.read_resource(f"imap://attachments/{key.encode()}/0")
        assert base64.b64decode(contents[0].blob) == b"\x00\xffbinary"
        assert contents[0].mimeType == "application/octet-stream"
    assert server.mailbox is None
    assert stopped and all(stopped)


@pytest.mark.asyncio
async def test_delivery_loop_waits_retries_and_stops(mailbox, store, monkeypatch):
    """A failed delivery is retried and a stopped adapter does not keep dispatching."""
    monitor = events.Monitor(mailbox, store, retry_seconds=0.001)
    calls = []

    async def attempt(http, url):
        """Fail once then request shutdown after a successful empty poll."""
        calls.append(url)
        if len(calls) == 1:
            raise httpx.ConnectError("Unavailable")
        monitor.stop.set()
        return 0

    monkeypatch.setattr(monitor, "deliver_once", attempt)
    async with httpx.AsyncClient() as http:
        await asyncio.wait_for(monitor.deliver(http, "http://session.test"), 2)
    assert len(calls) == 2


def test_stopped_scans_and_idle_keepalives(mailbox, client, store):
    """Shutdown prevents new notifications and ends IDLE even without a change response."""
    monitor = events.Monitor(mailbox, store)
    store.checkpoint(mailbox.settings.account, "INBOX", 21, 0)
    monitor.stop.set()
    monitor.scan(client, "INBOX", 21, 0)
    client.fetch.assert_not_called()
    monitor.wait_for_change(client, True)
    client.idle_done.assert_called_once()


def test_watcher_supports_servers_without_idle(mailbox, client, store, monkeypatch):
    """A server lacking IDLE still produces adapter-side events without agent polling."""
    monitor = events.Monitor(mailbox, store)
    client.has_capability.return_value = False

    def end_after_first_scan(connection, idle):
        """Stop the test after verifying the fallback mode selected by the watcher."""
        assert connection is client and idle is False
        monitor.stop.set()

    monkeypatch.setattr(monitor, "wait_for_change", end_after_first_scan)
    monitor.watch("INBOX")
    assert store.checkpoint(mailbox.settings.account, "INBOX", 21, 0) == 7


@pytest.mark.parametrize("value,expected", [
    ("startup", None), ("all", None),
    ("2026-09-14", datetime(2026, 9, 14, tzinfo=timezone.utc)),
    ("2026-09-14T09:00:00-04:00", datetime(2026, 9, 14, 13, tzinfo=timezone.utc)),
])
def test_imap_notification_date_modes(value, expected):
    """Modes and explicit dates have predictable, timezone-safe meanings."""
    assert model.notification_date(value) == expected


@pytest.mark.parametrize("value", ["", "yesterday", "2026-02-31", "2026-09-14T09:00:00"])
def test_invalid_imap_notification_dates(settings, value):
    """Malformed cutoffs fail configuration rather than silently disabling filtering."""
    with pytest.raises(ValueError):
        replace(settings, notify_since=value)


def test_bridge_backfill_is_suppressed_by_receipt_date_without_hiding_mail(mailbox, client, store, tmp_path):
    """Increasing UIDs for old imported mail do not bypass the default persisted cutoff."""
    mailbox.settings = replace(mailbox.settings, notify_since="startup")
    monitor = events.Monitor(mailbox, store)
    floor = monitor.notification_floor("INBOX")
    monitor.scan(client, "INBOX", 21, 7)
    original_fetch = client.fetch.side_effect

    def fetch(uids, fields):
        """Supply old and new receipt times while leaving the sender's Date header unchanged."""
        records = original_fetch(uids, fields)
        for uid in uids:
            records[uid][b"INTERNALDATE"] = floor + (timedelta(seconds=1) if uid == 9 else -timedelta(days=365))
        return records

    client.fetch.side_effect = fetch
    client.search.return_value = [7, 8, 9]
    monitor.scan(client, "INBOX", 21, 7)
    pending = store.pending(mailbox.settings.account)
    assert [event["metadata"]["uid"] for _, event in pending] == [9]
    assert pending[0][1]["metadata"]["received_at"] == (floor + timedelta(seconds=1)).isoformat()
    assert store.checkpoint(mailbox.settings.account, "INBOX", 21, 0) == 9
    assert len(mailbox.search("ALL", "INBOX")) == 3
    reopened = events.EventStore(tmp_path / "state" / "imap.sqlite3")
    try:
        assert events.Monitor(mailbox, reopened).notification_floor("INBOX") == floor
        reopened.checkpoint(mailbox.settings.account, "INBOX", 22, 0)
        assert reopened.notification_floor(mailbox.settings.account, "INBOX") == floor
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_imap_legacy_outbox_is_filtered_using_server_dates(mailbox, client, store):
    """An upgrade suppresses historical notifications already queued by the old adapter."""
    original = events.Monitor(mailbox, store)
    store.checkpoint(mailbox.settings.account, "INBOX", 21, 0)
    client.search.return_value = [7, 8]
    original.scan(client, "INBOX", 21, 0)
    for identifier, event in store.pending(mailbox.settings.account):
        del event["metadata"]["received_at"]
        with store.db:
            store.db.execute("UPDATE outbox SET payload=? WHERE id=?", (json.dumps(event), identifier))
    fetch = client.fetch.side_effect

    def dates(uids, fields):
        """Receipt dates, not sender headers, determine eligibility of legacy queued records."""
        values = fetch(uids, fields)
        for uid in uids:
            values[uid][b"INTERNALDATE"] = datetime(2025 if uid == 7 else 2026, 9, 14, 13, tzinfo=timezone.utc)
        return values

    client.fetch.side_effect = dates
    mailbox.settings = replace(mailbox.settings, notify_since="2026-09-14")
    monitor = events.Monitor(mailbox, store)
    posted = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: (posted.append(json.loads(request.content)), httpx.Response(202))[1])) as http:
        assert await monitor.deliver_once(http, "http://session.test") == 2
    assert [event["metadata"]["uid"] for event in posted] == [8]
    assert store.pending(mailbox.settings.account) == []


@pytest.mark.asyncio
async def test_imap_legacy_receipt_lookup_failure_retains_queue(mailbox, client, store, monkeypatch):
    """Transient IMAP failure during legacy filtering cannot discard an eligible notification."""
    monitor = events.Monitor(mailbox, store)
    store.checkpoint(mailbox.settings.account, "INBOX", 21, 0)
    monitor.scan(client, "INBOX", 21, 0)
    identifier, event = store.pending(mailbox.settings.account)[0]
    event["metadata"].pop("received_at")
    with store.db:
        store.db.execute("UPDATE outbox SET payload=? WHERE id=?", (json.dumps(event), identifier))
    mailbox.settings = replace(mailbox.settings, notify_since="startup")
    monitor = events.Monitor(mailbox, store)
    monkeypatch.setattr(mailbox, "received_at", Mock(side_effect=OSError("offline")))
    async with httpx.AsyncClient() as http:
        with pytest.raises(OSError):
            await monitor.deliver_once(http, "http://session.test")
        assert len(store.pending(mailbox.settings.account)) == 1
        monkeypatch.setattr(mailbox, "received_at", Mock(side_effect=ValueError("stale ID")))
        await monitor.deliver_once(http, "http://session.test")
    assert store.pending(mailbox.settings.account) == []


def test_receipt_lookup_handles_unknown_foreign_and_reset_ids(mailbox, client, key):
    """Receipt rechecks use readonly selection and never accept ambiguous naive timestamps."""
    assert mailbox.received_at(key.encode()) == datetime(2026, 9, 14, 13, tzinfo=timezone.utc)
    client.select_folder.assert_called_with(key.folder, readonly=True)
    with pytest.raises(ValueError):
        mailbox.received_at(key.model_copy(update={"account": "foreign"}).encode())
    client.select_folder.return_value[b"UIDVALIDITY"] = 22
    assert mailbox.received_at(key.encode()) is None
    assert model.received_time("2026-09-14T13:00:00Z") == datetime(2026, 9, 14, 13, tzinfo=timezone.utc)
    assert model.received_time("invalid") is None
    assert model.received_time(datetime(2026, 9, 14)) is None


def test_missing_receipt_date_does_not_emit_or_block_backfill_cursor(mailbox, client, store):
    """Unreadable receipt dates do not create notifications, but the mail remains searchable."""
    mailbox.settings = replace(mailbox.settings, notify_since="startup")
    monitor = events.Monitor(mailbox, store)
    store.checkpoint(mailbox.settings.account, "INBOX", 21, 0)
    client.fetch.side_effect = lambda uids, fields: {uid: {b"BODY[HEADER]": b"Subject: Test\n"} for uid in uids}
    monitor.scan(client, "INBOX", 21, 0)
    assert store.pending(mailbox.settings.account) == []
    assert store.checkpoint(mailbox.settings.account, "INBOX", 21, 0) == 7
