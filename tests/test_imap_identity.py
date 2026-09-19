"""Native mail identity selection, bounded relocation, and durable location observations."""

import base64
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
import json
from unittest.mock import Mock

import pytest

from imap_mcp import events, model, server
from imap_mcp.identity import IdentityStore, LookupIncompleteError, MessageKey, NativeKey, decode_id


def mail(native="native-1", subject="Synthetic mail"):
    """Return synthetic RFC822 mail with a native provider header and a binary attachment."""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = "sender@example.test"
    if native is not None:
        message[model.PROTON_HEADER] = native
    message.set_content("Synthetic body")
    message.add_attachment(b"attachment bytes", maintype="application", subtype="octet-stream", filename="data.bin")
    return message.as_bytes()


class FakeIMAP:
    """A folder-aware synthetic server honoring requested FETCH fields and read-only selection."""

    def __init__(self, emailid=False, proton=True):
        """Initialize two folders with no network or external account state."""
        self.caps = {"OBJECTID"} if emailid else set()
        self.caps.add("ID")
        self.identification = ((b"name", b"Proton Mail Bridge" if proton else b"Generic IMAP"),)
        self.messages = {"INBOX": {7: (mail(), "Email_1")}, "Archive": {}}
        self.epochs = {"INBOX": 21, "Archive": 30}
        self.selected = ""
        self.fetches = []
        self.searches = []

    def has_capability(self, capability):
        """Return whether this synthetic connection supports the named extension."""
        return capability in self.caps

    def id_(self):
        """Return a realistic IMAPClient parsed ID response."""
        return self.identification

    def list_folders(self):
        """List selectable folders plus one unselectable container."""
        return [((), b"/", name) for name in self.messages] + [((b"\\Noselect",), b"/", "Container")]

    def select_folder(self, folder, readonly=False):
        """Select a known folder and require that callers never request write access."""
        assert readonly
        self.selected = folder
        return {b"UIDVALIDITY": self.epochs[folder],
                b"UIDNEXT": max(self.messages[folder], default=0) + 1}

    def search(self, criteria):
        """Support standard EMAILID and UID/ALL searches, never expensive provider HEADER searches."""
        self.searches.append((self.selected, criteria))
        rows = self.messages[self.selected]
        if isinstance(criteria, list) and criteria[0] == "EMAILID":
            return [uid for uid, (_, identity) in rows.items() if identity == criteria[1]]
        if isinstance(criteria, list) and criteria[0] == "UID":
            return [uid for uid in rows if uid >= int(criteria[1].split(":")[0])]
        assert criteria == "ALL" or criteria == ["ALL"]
        return list(rows)

    def fetch(self, uids, fields):
        """Return only requested metadata or body fields for currently existing UIDs."""
        self.fetches.append((self.selected, list(uids), list(fields)))
        result = {}
        for uid in uids:
            if uid not in self.messages[self.selected]:
                continue
            raw, emailid = self.messages[self.selected][uid]
            row = {}
            for item in fields:
                if item == "EMAILID":
                    assert "OBJECTID" in self.caps
                    row[b"EMAILID"] = (emailid.encode(),) if emailid is not None else None
                elif item == "RFC822.SIZE":
                    row[b"RFC822.SIZE"] = len(raw)
                elif item == "INTERNALDATE":
                    row[b"INTERNALDATE"] = datetime(2026, 9, 19, tzinfo=timezone.utc)
                elif item == "BODY.PEEK[]":
                    row[b"BODY[]"] = raw
                elif item == "BODY.PEEK[HEADER]":
                    row[b"BODY[HEADER]"] = raw.split(b"\n\n", 1)[0] + b"\n\n"
                elif item == model.PROTON_FETCH:
                    headers = BytesParser(policy=policy.default).parsebytes(raw, headersonly=True)
                    row[b"BODY[HEADER.FIELDS (X-PM-INTERNAL-ID)]"] = b"".join(
                        f"{model.PROTON_HEADER}: {value}\r\n".encode()
                        for value in headers.get_all(model.PROTON_HEADER, [])) + b"\r\n"
                else:
                    raise AssertionError(f"Unexpected fetch field: {item}")
            result[uid] = row
        return result


@pytest.fixture
def protocol():
    """Provide a fresh recognized Proton-like server."""
    return FakeIMAP()


@pytest.fixture
def mailbox(protocol, tmp_path):
    """Use a persistent test-only cache and a fake connection for all mailbox operations."""
    box = model.Mailbox(model.Settings("imap://test", "user", "password", notify_since="all"),
                        state_path=tmp_path / "imap.sqlite3")
    wire(box, protocol)
    yield box
    box.close()


def wire(box, protocol):
    """Connect a Mailbox to one synthetic server without modifying the adapter's global state."""
    @contextmanager
    def connection():
        """Yield the fake server as a per-call connection."""
        yield protocol
    box.connect = connection


def fresh_mcp():
    """Use a fresh server lifetime with the production handlers for each transport test."""
    from fastmcp import FastMCP

    app = FastMCP("imap-test", lifespan=server.lifespan, mask_error_details=True)
    for handler in [server.get_mail, server.search_mail, server.get_attachment, server.list_folders]:
        app.tool(handler)
    app.resource("imap://attachments/{message_id}/{attachment_id}")(server.attachment_resource)
    return app


@pytest.mark.parametrize("objectid,provider,expected", [
    (True, True, "emailid"), (True, False, "emailid"),
    (False, True, "proton"), (False, False, "mailbox"),
])
def test_identity_precedence_and_generic_fallback(mailbox, protocol, objectid, provider, expected):
    """Standard IDs win; only server-recognized providers may supply fallback header IDs."""
    protocol.caps = {"ID", "OBJECTID"} if objectid else {"ID"}
    protocol.identification = ((b"name", b"Proton Mail Bridge" if provider else b"Other"),)
    message = mailbox.search("ALL", "INBOX")[0]
    assert message.identity_kind == expected
    identity = decode_id(message.id)
    assert identity.account == mailbox.settings.account
    if expected != "mailbox":
        assert identity.value == ("Email_1" if objectid else "native-1")
    assert mailbox.get(message.id).body.strip() == "Synthetic body"


def test_native_id_is_identical_in_multiple_folders(mailbox, protocol):
    """Location is not encoded in native IDs, but each summary reports the observed folder."""
    protocol.messages["Archive"][101] = protocol.messages["INBOX"][7]
    first = mailbox.search("ALL", "INBOX")[0]
    second = mailbox.search("ALL", "Archive")[0]
    assert first.id == second.id and first.folder != second.folder
    assert len(mailbox.identities.locations(decode_id(first.id))) == 2


@pytest.mark.parametrize("objectid", [False, True])
def test_old_native_ids_and_attachment_uris_survive_moves(mailbox, protocol, objectid):
    """An issued ID resolves a new folder/UID without changing the ID or attachment bytes."""
    if objectid:
        protocol.caps.add("OBJECTID")
    original = mailbox.search("ALL", "INBOX")[0]
    attachment_uri = mailbox.get(original.id).attachment_metadata[0].uri
    protocol.messages["Archive"][99] = protocol.messages["INBOX"].pop(7)
    moved = mailbox.get(original.id)
    assert moved.id == original.id and moved.folder == "Archive"
    assert moved.attachment_metadata[0].uri == attachment_uri
    assert mailbox.attachment(original.id, "0")[0] == b"attachment bytes"
    assert not any(isinstance(query, list) and query[0] == "HEADER" for _, query in protocol.searches)
    if objectid:
        assert ("Archive", ["EMAILID", "Email_1"]) in protocol.searches


def test_cached_fetch_does_not_search_mailboxes(mailbox, protocol):
    """The normal read path validates a cached UID and avoids any mailbox search."""
    message = mailbox.search("ALL", "INBOX")[0]
    protocol.searches.clear()
    protocol.fetches.clear()
    assert mailbox.get(message.id).subject == "Synthetic mail"
    assert protocol.searches == []
    assert len(protocol.fetches) == 2
    assert model.PROTON_FETCH in protocol.fetches[0][2]
    assert "BODY.PEEK[]" in protocol.fetches[1][2]


def test_native_id_recovery_survives_renamed_folder_and_epoch_reset(mailbox, protocol):
    """Old folder names and reused UIDs cannot redirect a native ID to a different message."""
    original = mailbox.search("ALL", "INBOX")[0]
    row = protocol.messages.pop("INBOX")[7]
    protocol.messages["Renamed"] = {3: row}
    protocol.epochs["Renamed"] = 55
    assert mailbox.get(original.id).folder == "Renamed"
    protocol.messages["Renamed"] = {3: (mail("other", "Wrong message"), "Other"), 8: row}
    protocol.epochs["Renamed"] = 56
    result = mailbox.get(original.id)
    assert result.subject == "Synthetic mail"
    assert mailbox.identities.locations(decode_id(original.id))[0].validity == 56


def test_matching_header_is_rechecked_after_body_fetch(mailbox, protocol, monkeypatch):
    """A message changing between preflight and body read cannot be returned under the old ID."""
    original = mailbox.search("ALL", "INBOX")[0]
    fetch = protocol.fetch

    def replace_during_body(uids, fields):
        """Simulate a bad server reassigning the selected UID between FETCH commands."""
        if "BODY.PEEK[]" in fields:
            protocol.messages["INBOX"][7] = (mail("other", "Wrong"), "Other")
        return fetch(uids, fields)

    monkeypatch.setattr(protocol, "fetch", replace_during_body)
    with pytest.raises(KeyError):
        mailbox.get(original.id)


def test_recovery_budget_is_resumable_and_survives_restart(mailbox, protocol, tmp_path):
    """Recovery reads a bounded number of new headers, preserving progress across process lifetimes."""
    mailbox.settings = replace(mailbox.settings, lookup_max_messages=1)
    protocol.messages = {"INBOX": {1: (mail("target"), "T"), 2: (mail("second"), "S"), 3: (mail("third"), "D")}}
    identity = NativeKey(account=mailbox.settings.account, kind="proton", value="target")
    with pytest.raises(LookupIncompleteError, match="incomplete"):
        mailbox.get(identity.encode())
    assert [entry[1] for entry in protocol.fetches] == [[3]]
    reopened = model.Mailbox(mailbox.settings, state_path=tmp_path / "imap.sqlite3")
    wire(reopened, protocol)
    try:
        with pytest.raises(LookupIncompleteError):
            reopened.get(identity.encode())
        assert reopened.get(identity.encode()).id == identity.encode()
        assert [entry[1] for entry in protocol.fetches if entry[2] == [model.PROTON_FETCH]] == [[3], [2], [1]]
    finally:
        reopened.close()


def test_native_ids_reject_other_accounts_and_unavailable_mechanisms(mailbox, protocol):
    """Neither cache hits nor opaque ID contents can cross accounts or bypass provider configuration."""
    message = mailbox.search("ALL", "INBOX")[0]
    key = decode_id(message.id)
    with pytest.raises(ValueError, match="account"):
        mailbox.get(key.model_copy(update={"account": "other"}).encode())
    mailbox.settings = replace(mailbox.settings, id_provider="generic")
    with pytest.raises(ValueError, match="mechanism"):
        mailbox.get(message.id)
    with pytest.raises(ValueError, match="mechanism"):
        mailbox.get(NativeKey(account=key.account, kind="emailid", value="Email_1").encode())


def test_old_mailbox_ids_keep_their_original_safety_guards(mailbox, protocol):
    """Legacy IDs are still accepted, but cannot claim move stability they never contained."""
    key = MessageKey(account=mailbox.settings.account, folder="INBOX", validity=21, uid=7)
    assert mailbox.get(key.encode()).id == key.encode()
    protocol.epochs["INBOX"] = 22
    with pytest.raises(ValueError, match="UIDVALIDITY"):
        mailbox.get(key.encode())


def test_notifications_use_native_ids_without_changing_outbox_identity(mailbox, protocol, tmp_path):
    """Native mail references do not change per-folder checkpoint or notification deduplication keys."""
    state = events.EventStore(tmp_path / "imap.sqlite3")
    try:
        state.checkpoint(mailbox.settings.account, "INBOX", 21, 0)
        protocol.select_folder("INBOX", readonly=True)
        events.Monitor(mailbox, state).scan(protocol, "INBOX", 21, 0)
        event_id, event = state.pending(mailbox.settings.account)[0]
        assert isinstance(decode_id(event_id), MessageKey)
        assert event["metadata"]["event_id"] == event_id
        native_id = event["metadata"]["message_id"]
        assert isinstance(decode_id(native_id), NativeKey)
        assert json.loads(event["text"])["id"] == native_id
        protocol.messages["Archive"][12] = protocol.messages["INBOX"].pop(7)
        assert mailbox.get(native_id).folder == "Archive"
        assert state.checkpoint(mailbox.settings.account, "INBOX", 21, 0) == 7
    finally:
        state.close()


@pytest.mark.parametrize("kind,value", [("emailid", "M_Case-Sensitive"), ("proton", "a+/base64==")])
def test_native_key_round_trip_preserves_native_value(kind, value):
    """Opaque wrappers preserve backend IDs exactly while keeping paths URL-safe."""
    key = NativeKey(account="account", kind=kind, value=value)
    assert decode_id(key.encode()) == key
    assert "/" not in key.encode() and "+" not in key.encode()


@pytest.mark.parametrize("value", ["", "legacy", "v2.bad id", "v2.e30", "v2." + "x" * 4097])
def test_invalid_native_key_encodings(value):
    """Unversioned, incomplete, malformed, and oversized native IDs are rejected."""
    with pytest.raises(ValueError):
        NativeKey.decode(value)


@pytest.mark.parametrize("kind,value", [("unknown", "id"), ("emailid", "a/b"),
                                         ("emailid", "a" * 256), ("proton", "bad\r\nvalue")])
def test_invalid_native_id_values(kind, value):
    """Unsupported kinds and command-control characters never become IMAP commands."""
    with pytest.raises(ValueError):
        NativeKey(account="account", kind=kind, value=value)


@pytest.mark.parametrize("value,expected", [((b"Ab_C",), "Ab_C"), (b"Ab", "Ab"),
                                            ((42,), "42"), ((b"0042",), "0042"),
                                            ((b"a", b"b"), None), ((b"\xff",), None), (None, None)])
def test_emailid_fetch_parsing(value, expected):
    """The IMAPClient tuple representation is decoded without case folding or losing leading zeros."""
    assert model.native_value({b"EMAILID": value}, "emailid") == expected


def test_duplicate_or_missing_provider_headers_do_not_become_native_ids(mailbox, protocol):
    """Ambiguous and absent provider metadata use the honest mailbox-scoped fallback."""
    for raw in [mail(None), b"X-Pm-Internal-Id: one\nX-Pm-Internal-Id: two\n\nbody"]:
        protocol.messages["INBOX"][7] = (raw, None)
        assert mailbox.search("ALL", "INBOX")[0].identity_kind == "mailbox"


@pytest.mark.parametrize("response,expected", [
    ({b"name": b"Proton Mail Bridge"}, True), ({"name": "protonmail-bridge"}, True),
    ((None,), False), (((b"name",),), False), ({"name": None}, False),
])
def test_provider_id_response_shapes(mailbox, protocol, response, expected):
    """Provider recognition depends on valid server metadata, never message content."""
    protocol.identification = response
    assert mailbox.identity_capabilities(protocol).proton is expected


def test_provider_override_and_id_command_failure(mailbox, protocol, monkeypatch):
    """Explicit provider configuration works without ID; unsupported ID safely stays generic."""
    monkeypatch.setattr(protocol, "id_", Mock(side_effect=model.IMAPClientError("ID unavailable")))
    assert not mailbox.identity_capabilities(protocol).proton
    mailbox.settings = replace(mailbox.settings, id_provider="proton")
    assert mailbox.identity_capabilities(protocol).proton


def test_identity_cache_scopes_accounts_and_prunes_only_old_epochs(tmp_path):
    """Cache eviction cannot erase unrelated accounts or other locations of a message."""
    cache = IdentityStore(tmp_path / "cache.sqlite3")
    location = MessageKey(account="a", folder="INBOX", validity=1, uid=1)
    key = NativeKey(account="a", kind="proton", value="native")
    try:
        cache.remember(location, "proton", key.value)
        foreign = location.model_copy(update={"account": "b"})
        cache.remember(foreign, "proton", key.value)
        assert cache.locations(key) == [location]
        assert cache.unseen("a", "INBOX", 1, "proton", [1, 2, 3]) == [3, 2]
        cache.prune_epoch("a", "INBOX", 2)
        assert cache.locations(key) == []
        assert cache.locations(key.model_copy(update={"account": "b"})) == [foreign]
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_native_mail_and_attachment_ids_work_through_mcp(mailbox, protocol, monkeypatch, tmp_path):
    """MCP tool/resource consumers can keep the same opaque ID across a simulated move."""
    from fastmcp import Client

    monkeypatch.setattr(server.Settings, "from_env", lambda: mailbox.settings)
    monkeypatch.setattr(server, "Mailbox", lambda settings, state_path=None: mailbox)
    monkeypatch.setattr(events.Monitor, "watch", lambda self, folder: self.stop.wait(5))
    monkeypatch.setenv("IMAP_STATE_PATH", str(tmp_path / "events.sqlite3"))
    async with Client(fresh_mcp()) as client:
        search = await client.call_tool("search_mail", {"search": "ALL"})
        identifier = search.structured_content["result"][0]["id"]
        assert search.structured_content["result"][0]["identity_kind"] == "proton"
        protocol.messages["Archive"][9] = protocol.messages["INBOX"].pop(7)
        result = await client.call_tool("get_mail", {"message_id": identifier})
        assert result.structured_content["id"] == identifier
        assert result.structured_content["folder"] == "Archive"
        contents = await client.read_resource(result.structured_content["attachment_metadata"][0]["uri"])
        assert base64.b64decode(contents[0].blob) == b"attachment bytes"


@pytest.mark.parametrize("tool", ["get_mail", "get_attachment", "attachment_resource"])
def test_incomplete_recovery_is_not_hidden_by_mcp_error_masking(monkeypatch, tool):
    """Budget exhaustion stays actionable rather than becoming a generic not-found error."""
    from fastmcp.exceptions import ResourceError, ToolError

    box = Mock()
    box.get.side_effect = LookupIncompleteError("recovery incomplete; retry")
    box.attachment.side_effect = LookupIncompleteError("recovery incomplete; retry")
    monkeypatch.setattr(server, "mailbox", box)
    args = ["id"] if tool == "get_mail" else ["id", "0"]
    with pytest.raises(ResourceError if tool == "attachment_resource" else ToolError, match="incomplete"):
        getattr(server, tool)(*args)


@pytest.mark.parametrize("changes", [{"id_provider": "bad"}, {"lookup_max_messages": 0},
                                    {"lookup_max_messages": 5001}, {"lookup_max_messages": 1.5}])
def test_identity_configuration_is_validated_before_connect(mailbox, changes):
    """Invalid provider choices and scan budgets fail configuration before network access."""
    with pytest.raises(ValueError):
        replace(mailbox.settings, **changes)


def test_identity_configuration_from_environment(monkeypatch):
    """Environment settings preserve explicit provider mode and recovery budget."""
    for name, value in {"IMAP_SERVER": "imap://test", "IMAP_USERNAME": "u", "IMAP_PASSWORD": "p",
                        "IMAP_ID_PROVIDER": "generic", "IMAP_LOOKUP_MAX_MESSAGES": "99"}.items():
        monkeypatch.setenv(name, value)
    settings = model.Settings.from_env()
    assert settings.id_provider == "generic" and settings.lookup_max_messages == 99


def test_missing_emailid_reports_absence_after_native_search(mailbox, protocol):
    """Standard EMAILID recovery can report absence after checking all selectable folders."""
    protocol.caps.add("OBJECTID")
    key = NativeKey(account=mailbox.settings.account, kind="emailid", value="Missing")
    with pytest.raises(KeyError, match="not found"):
        mailbox.get(key.encode())


def test_recovery_spreads_work_across_folders_and_retains_progress(mailbox, protocol):
    """A small global budget can defer a folder without pretending the missing ID is absent."""
    mailbox.settings = replace(mailbox.settings, lookup_max_messages=1)
    protocol.messages["Archive"][9] = (mail("target"), "Target")
    key = NativeKey(account=mailbox.settings.account, kind="proton", value="target")
    with pytest.raises(LookupIncompleteError):
        mailbox.get(key.encode())
    assert mailbox.get(key.encode()).folder == "Archive"


def test_recovery_detects_epoch_changes_before_header_fetch(mailbox, protocol, monkeypatch):
    """An epoch changing between UID discovery and FETCH cannot poison the location cache."""
    protocol.messages.pop("Archive")
    original = protocol.select_folder
    selections = 0

    def select(folder, readonly=False):
        """Change UIDVALIDITY just before the recovery batch is fetched."""
        nonlocal selections
        selections += 1
        if selections == 2:
            protocol.epochs[folder] += 1
        return original(folder, readonly)

    monkeypatch.setattr(protocol, "select_folder", select)
    key = NativeKey(account=mailbox.settings.account, kind="proton", value="native-1")
    with pytest.raises(LookupIncompleteError, match="changed"):
        mailbox.get(key.encode())
    assert protocol.fetches == []


def test_recovery_skips_concurrently_expunged_uids(mailbox, protocol, monkeypatch):
    """A vanished candidate does not prevent locating another matching message."""
    protocol.messages["INBOX"][8] = (mail("deleted"), "Deleted")
    fetch = protocol.fetch

    def expunge(uids, fields):
        """Remove a UID after it was listed but before its metadata is fetched."""
        protocol.messages["INBOX"].pop(8, None)
        return fetch(uids, fields)

    monkeypatch.setattr(protocol, "fetch", expunge)
    key = NativeKey(account=mailbox.settings.account, kind="proton", value="native-1")
    assert mailbox.get(key.encode()).id == key.encode()


def test_native_read_does_not_hide_network_or_size_errors(mailbox, protocol, monkeypatch):
    """Protocol failures and oversized messages remain errors, not stale-cache misses."""
    message = mailbox.search("ALL", "INBOX")[0]
    fetch = protocol.fetch
    with monkeypatch.context() as patch:
        patch.setattr(protocol, "fetch", Mock(side_effect=OSError("offline")))
        with pytest.raises(OSError):
            mailbox.get(message.id)
    assert mailbox.identities.locations(decode_id(message.id))
    mailbox.settings = replace(mailbox.settings, max_message_bytes=1)
    with pytest.raises(ValueError, match="exceeds"):
        mailbox.get(message.id)
    mailbox.settings = replace(mailbox.settings, max_message_bytes=99999)

    def missing_size(uids, fields):
        """Omit mandatory metadata without changing the verified identity."""
        records = fetch(uids, fields)
        for record in records.values():
            record.pop(b"RFC822.SIZE", None)
        return records

    monkeypatch.setattr(protocol, "fetch", missing_size)
    with pytest.raises(RuntimeError, match="size"):
        mailbox.get(message.id)


@pytest.mark.asyncio
async def test_incomplete_lookup_is_actionable_over_mcp(mailbox, protocol, monkeypatch, tmp_path):
    """Actual transport errors retain the incomplete/retry distinction under error masking."""
    from fastmcp import Client
    from fastmcp.exceptions import ToolError
    from mcp.shared.exceptions import McpError

    mailbox.settings = replace(mailbox.settings, lookup_max_messages=1)
    protocol.messages = {"INBOX": {uid: (mail("target" if uid == 1 else f"other-{uid}"), f"M{uid}")
                                    for uid in range(1, 5)}}
    key = NativeKey(account=mailbox.settings.account, kind="proton", value="target")
    monkeypatch.setattr(server.Settings, "from_env", lambda: mailbox.settings)
    monkeypatch.setattr(server, "Mailbox", lambda settings, state_path=None: mailbox)
    monkeypatch.setattr(events.Monitor, "watch", lambda self, folder: self.stop.wait(5))
    monkeypatch.setenv("IMAP_STATE_PATH", str(tmp_path / "events.sqlite3"))
    async with Client(fresh_mcp()) as client:
        with pytest.raises(ToolError, match="incomplete"):
            await client.call_tool("get_mail", {"message_id": key.encode()})
        with pytest.raises(McpError, match="incomplete"):
            await client.read_resource(f"imap://attachments/{key.encode()}/0")
        with pytest.raises(ToolError, match="incomplete"):
            await client.call_tool("get_attachment", {"message_id": key.encode(), "attachment_id": "0"})
        result = await client.call_tool("get_mail", {"message_id": key.encode()})
        assert result.structured_content["identity_kind"] == "proton"
