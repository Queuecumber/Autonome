"""Local mail search, read-only synchronization, hybrid retrieval, and coverage honesty."""

import base64
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import format_datetime
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import OpenAI, RateLimitError
import pytest

from imap_mcp import embedding, index, model, server, sync
from imap_mcp.embedding import Embedder, EmbeddingSettings
from imap_mcp.index import HeaderObservation, MailIndex, MailQuery
from imap_mcp.sync import IndexedMailbox, IndexSettings
from test_imap_identity import FakeIMAP, fresh_mcp, wire

NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


def raw_mail(native, subject="Weekly Update C125068758", body="The preschool calendar is attached."):
    """Build synthetic mail; no fixture includes real addresses or mailbox content."""
    mail = EmailMessage()
    mail["From"] = "School Office <school@example.test>"
    mail["To"] = "Parent <parent@example.test>"
    mail["Cc"] = "Other <other@example.test>"
    mail["Subject"] = subject
    if native is not None:
        mail[model.PROTON_HEADER] = native
    mail.set_content(body)
    return mail.as_bytes()


class IndexedIMAP(FakeIMAP):
    """Synthetic IMAP with real receipt ordering and mutable flags, never a live connection."""

    def __init__(self):
        """Seed a low-UID recent mail, high-UID old mail, and a native duplicate across folders."""
        super().__init__()
        self.messages = {"INBOX": {7: (raw_mail("recent"), "Recent")},
                         "Archive": {434: (raw_mail("recent"), "Recent"),
                                     100_000: (raw_mail("old", "Old note", "A solar inspection was postponed."), "Old")}}
        self.receipts = {("INBOX", 7): NOW, ("Archive", 434): NOW,
                         ("Archive", 100_000): NOW - timedelta(days=365)}
        self.flags = {("INBOX", 7): (b"\\Seen",)}

    def search(self, criteria):
        """Support only UID inventory and receipt-date narrowing, never server full-text search."""
        if isinstance(criteria, list) and criteria[0] == "SINCE":
            self.searches.append((self.selected, criteria))
            return [uid for uid in self.messages[self.selected]
                    if self.receipts.get((self.selected, uid), NOW).date() >= criteria[1]]
        return super().search(criteria)

    def fetch(self, uids, fields):
        """Supply only requested metadata, flags, and PEEK bytes."""
        assert not any(field.startswith("BODY[") for field in fields)
        records = super().fetch(uids, [field for field in fields if field != "FLAGS"])
        for uid, record in records.items():
            if "FLAGS" in fields:
                record[b"FLAGS"] = self.flags.get((self.selected, uid), ())
            if "INTERNALDATE" in fields:
                record[b"INTERNALDATE"] = self.receipts.get((self.selected, uid), NOW)
        return records


@pytest.fixture
def indexed(tmp_path):
    """Wire the real data model to a synthetic server and an isolated on-disk index."""
    protocol = IndexedIMAP()
    mailbox = model.Mailbox(model.Settings("imap://test", "user", "password"), tmp_path / "notifications.sqlite3")
    wire(mailbox, protocol)
    store = MailIndex(tmp_path / "search.sqlite3", mailbox.settings.account)
    embedder = Embedder(EmbeddingSettings())
    service = IndexedMailbox(mailbox, store, embedder, IndexSettings())
    yield service, protocol
    service.shutdown()
    embedder.close()
    store.close()
    mailbox.close()


def ids(page):
    """Extract stable public mail IDs from a typed result page."""
    return [hit.message.id for hit in page.results]


def test_initial_index_is_explicitly_incomplete_and_search_never_connects(indexed):
    """A cold empty database is not presented as proof that the mailbox contains no matches."""
    service, protocol = indexed
    page = service.search(MailQuery(query="calendar", mode="keyword"))
    assert not page.complete and not page.index.available and not page.results
    assert page.warnings and not protocol.searches and not protocol.fetches


def test_keyword_search_filters_dates_flags_and_native_dedup(indexed):
    """Local queries combine sender/recipient/date/folder/flag filters without IMAP I/O."""
    service, protocol = indexed
    service.sync_once()
    before = len(protocol.searches), len(protocol.fetches)
    page = service.search(MailQuery(query="calendar", sender="SCHOOL", recipient="parent@", mode="keyword"))
    assert page.complete and len(page.results) == 1
    hit = page.results[0]
    assert hit.folders == ["Archive", "INBOX"]
    assert hit.message.body is None and hit.message.attachment_metadata is None
    assert "calendar" in hit.snippet
    assert service.search(MailQuery(query="C125068758", mode="keyword")).results
    assert service.search(MailQuery(subject="weekly UPDATE", after=NOW - timedelta(days=1))).results
    assert not service.search(MailQuery(subject="weekly", before=NOW)).results
    assert service.search(MailQuery(folder="INBOX", unread=False)).results
    assert not service.search(MailQuery(folder="INBOX", unread=True)).results
    assert service.search(MailQuery(folder="Archive", unread=True, subject="weekly")).results
    assert not service.search(MailQuery(sender="missing")).results
    assert (len(protocol.searches), len(protocol.fetches)) == before


def test_browse_uses_receipt_date_not_uid_and_paginates(indexed):
    """Recent low UIDs precede older re-imported high UIDs, without folder-copy duplication."""
    service, _ = indexed
    service.sync_once()
    first = service.search(MailQuery(limit=1))
    second = service.search(MailQuery(limit=1, offset=first.next_offset))
    assert first.results[0].message.subject.startswith("Weekly")
    assert second.results[0].message.subject == "Old note"
    assert second.next_offset is None and ids(first) != ids(second)


def test_cached_bodies_avoid_network_and_refresh_is_explicit(indexed):
    """Body reads expose cache freshness and can deliberately bypass the local snapshot."""
    service, protocol = indexed
    service.sync_once()
    identifier = service.search(MailQuery(subject="weekly")).results[0].message.id
    count = len(protocol.fetches)
    message = service.get(identifier)
    assert message.cached and not message.stale and message.body.strip().startswith("The preschool")
    assert len(protocol.fetches) == count
    assert not service.get(identifier, refresh=True).cached
    assert len(protocol.fetches) > count
    service.index.error("TimeoutError")
    assert service.get(identifier).stale


def test_moves_deletes_and_epoch_reset_do_not_reuse_wrong_message(indexed):
    """Native IDs survive moves; deletions and UIDVALIDITY resets cannot return old cache bytes."""
    service, protocol = indexed
    service.sync_once()
    identifier = ids(service.search(MailQuery(subject="weekly")))[0]
    protocol.messages["INBOX"].clear()
    protocol.messages["Archive"][10] = protocol.messages["Archive"].pop(434)
    service.sync_once()
    assert ids(service.search(MailQuery(subject="weekly"))) == [identifier]
    assert service.get(identifier).folder == "Archive"
    protocol.epochs["Archive"] += 1
    protocol.messages["Archive"] = {10: (raw_mail("replacement", "Replacement", "Different bytes"), "Replacement")}
    service.sync_once()
    assert not service.search(MailQuery(subject="weekly")).results
    assert service.index.cached(identifier) is None
    with pytest.raises(KeyError):
        service.get(identifier)
    protocol.messages["Archive"].clear()
    service.sync_once()
    assert not service.search(MailQuery()).results
    assert service.index.db.execute("SELECT count(*) FROM mail").fetchone()[0] == 0
    assert service.index.db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_folder_removal_and_partial_move_keep_body_until_destination_headers(indexed):
    """Renames do not lose cached content while destination headers are still incomplete."""
    service, protocol = indexed
    service.sync_once()
    identifier = ids(service.search(MailQuery(subject="weekly")))[0]
    protocol.messages["Renamed"] = protocol.messages.pop("INBOX")
    protocol.messages["Archive"].pop(434)
    protocol.epochs["Renamed"] = 40
    service.index.folders(["Archive", "Renamed"])
    service.index.reconcile("Renamed", 40, [7], [], {})
    service.index.reconcile("Archive", 30, [100_000], [], {})
    service.index.prune()
    assert service.index.db.execute("SELECT details FROM mail WHERE public_id=?", (identifier,)).fetchone()[0]
    service.sync_once()
    assert service.get(identifier).folder == "Renamed"


def test_backfill_is_bounded_recent_first_and_resumable(indexed):
    """Header/body progress survives reopening and never writes notification checkpoints/outbox."""
    service, protocol = indexed
    service.settings = replace(service.settings, header_batch=1, body_batch=1)
    service.sync_once()
    state = service.status()
    assert sum(folder.headers_pending for folder in state.folders) == 1
    assert service.search(MailQuery(subject="weekly")).results
    assert not service.search(MailQuery(query="inspection", mode="keyword")).complete
    assert any(criteria[0] == "SINCE" for _, criteria in protocol.searches)
    assert not service.index.db.execute("SELECT name FROM sqlite_master WHERE name IN ('outbox','checkpoints')").fetchall()
    path = Path(service.index.db.execute("PRAGMA database_list").fetchone()[2])
    service.index.close()
    service.index = MailIndex(path, service.mailbox.settings.account)
    service.sync_once()
    assert service.search(MailQuery(query="inspection", mode="keyword")).complete
    service.index.close()


def test_failed_refresh_retains_snapshot_and_reports_stale(indexed, monkeypatch):
    """A network/protocol failure cannot replace a populated folder with an empty one."""
    service, protocol = indexed
    service.sync_once()
    original = protocol.search

    def fail(criteria):
        """Fail one selected folder without exposing private server error details."""
        if protocol.selected == "Archive":
            raise RuntimeError("PRIVATE_SERVER_ERROR")
        return original(criteria)

    monkeypatch.setattr(protocol, "search", fail)
    service.sync_once()
    page = service.search(MailQuery(query="inspection", mode="keyword"))
    assert page.results and not page.complete and page.index.stale
    assert page.index.folders[0].error == "RuntimeError"
    assert "PRIVATE_SERVER_ERROR" not in page.model_dump_json()


def test_flags_refresh_is_bounded_and_case_insensitive(indexed):
    """Observed unread filters update without re-fetching immutable headers or bodies."""
    service, protocol = indexed
    service.sync_once()
    protocol.flags["INBOX", 7] = ()
    protocol.flags["Archive", 434] = (b"\\SEEN",)
    service.index.db.execute("UPDATE locations SET flags_at=0")
    service.index.db.commit()
    assert not service.search(MailQuery(unread=False)).complete
    service.sync_once()
    assert service.search(MailQuery(folder="INBOX", unread=True)).results
    assert not service.search(MailQuery(folder="Archive", subject="weekly", unread=True)).results


def test_oversized_bodies_are_not_fetched_or_silently_complete(indexed):
    """Header searches remain usable when a message exceeds the body safety limit."""
    service, protocol = indexed
    service.mailbox.settings = replace(service.mailbox.settings, max_message_bytes=1)
    service.sync_once()
    page = service.search(MailQuery(query="weekly", mode="keyword"))
    assert page.results and not page.complete and page.index.bodies_unavailable == 2
    assert not any("BODY.PEEK[]" in fields for _, _, fields in protocol.fetches)


def test_foreign_ids_and_notification_database_reuse_are_rejected(indexed, tmp_path):
    """Derived cache operations never cross accounts or open the persistent event store as an index."""
    service, _ = indexed
    foreign = model.MessageKey(account="other", folder="INBOX", validity=21, uid=7)
    with pytest.raises(ValueError, match="another account"):
        service.get(foreign.encode())
    with pytest.raises(ValueError, match="another account"):
        service.index.locations(foreign.encode())
    with pytest.raises(ValueError, match="another account"):
        service._warm_location(foreign.encode())
    path = Path(service.index.db.execute("PRAGMA database_list").fetchone()[2])
    with pytest.raises(ValueError, match="another account"):
        MailIndex(path, "other")
    with pytest.raises(ValueError, match="journal"):
        MailIndex(tmp_path / "bad.sqlite3", "a", journal_mode="invalid")
    with pytest.raises(ValueError, match="notification"):
        MailIndex(tmp_path / "notifications.sqlite3", "a")


def test_header_and_body_changes_replace_fts_content_atomically(indexed):
    """Changed headers invalidate old cached bodies/chunks while retaining the native ID."""
    service, protocol = indexed
    service.sync_once()
    identifier = ids(service.search(MailQuery(subject="weekly")))[0]
    location = model.MessageKey(account=service.mailbox.settings.account, folder="INBOX", validity=21, uid=7)
    message = model.parse_message(raw_mail("recent", "Changed title", "New contents"), location,
                                  received_at=NOW, identity=model.NativeKey.decode(identifier))
    observation = HeaderObservation(location=location, message=message.model_copy(update={"body": None}), flags=[], size=100)
    service.index.reconcile("INBOX", 21, [7], [observation], {})
    assert service.index.cached(identifier) is None
    assert not service.search(MailQuery(query="calendar", mode="keyword")).results
    assert service.index.save_body(message, location)
    assert service.search(MailQuery(query="New contents", mode="keyword")).results
    assert not service.index.save_body(message, location.model_copy(update={"validity": 22}))


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"offset": -1}, {"mode": "bad"},
    {"after": "2026-09-01"}, {"after": NOW, "before": NOW}, {"query": "x" * 4097}])
def test_invalid_search_scope(kwargs):
    """Malformed limits and timezone-ambiguous dates are rejected before search."""
    with pytest.raises(ValueError):
        MailQuery(**kwargs)


def test_literal_filters_and_queries_cannot_inject_sql_or_fts(indexed):
    """Quotes/operators/wildcards are data, not executable query syntax."""
    service, _ = indexed
    service.sync_once()
    for value in ("' OR 1=1 --", "%", "_", "\\"):
        assert not service.search(MailQuery(subject=value, mode="keyword")).results
    assert not service.search(MailQuery(query='" OR nonexistent', mode="keyword")).results
    assert not service.search(MailQuery(query="!!!", mode="keyword")).results
    with pytest.raises(ValueError, match="scope"):
        service.search(MailQuery(folder="missing"))


def enable_vectors(service):
    """Reopen the test index in a new explicitly configured vector space."""
    settings = EmbeddingSettings(model="synthetic", base_url="https://embedding.test/v1", dimensions=2)
    path = Path(service.index.db.execute("PRAGMA database_list").fetchone()[2])
    service.index.close()
    service.index = MailIndex(path, service.mailbox.settings.account, settings.profile)
    service.embedder.settings = settings
    service.embedder.encode = Mock(return_value=[[1.0, 0.0]])
    return path


def test_semantic_and_hybrid_search_filter_before_knn_and_deduplicate(indexed):
    """Semantic meaning can retrieve non-keyword mail, while hard scope filters remain binding."""
    service, _ = indexed
    service.sync_once()
    enable_vectors(service)
    jobs = service.index.chunks()
    vectors = [[1.0, 0.0] if "solar" in job.text else [0.0, 1.0] for job in jobs]
    service.index.vectors(jobs, vectors)
    semantic = service.search(MailQuery(query="rescheduling the installation", mode="semantic"))
    assert semantic.complete and semantic.mode == "semantic"
    assert [hit.message.subject for hit in semantic.results] == ["Old note"]
    assert semantic.results[0].semantic_score == pytest.approx(1)
    assert not service.search(MailQuery(query="rescheduling", folder="INBOX", mode="semantic")).results
    assert not service.search(MailQuery(query="rescheduling", after=NOW, mode="semantic")).results
    hybrid = service.search(MailQuery(query="calendar", mode="hybrid"))
    assert {hit.message.subject for hit in hybrid.results} == {"Weekly Update C125068758", "Old note"}
    assert len(ids(hybrid)) == len(set(ids(hybrid)))
    service.index.close()


def test_embedding_model_change_preserves_mail_but_rebuilds_vectors(indexed):
    """Vector profile changes never mix embeddings or force another IMAP/body backfill."""
    service, _ = indexed
    service.sync_once()
    path = enable_vectors(service)
    jobs = service.index.chunks()
    service.index.vectors(jobs, [[1.0, 0.0] for _ in jobs])
    assert service.status().chunks_pending == 0
    service.index.close()
    service.index = MailIndex(path, service.mailbox.settings.account, "different-profile")
    state = service.status()
    assert state.indexed_messages == 2 and state.bodies_pending == 0 and state.chunks_pending == 2
    assert not service.index.db.execute("SELECT 1 FROM sqlite_master WHERE name='vectors'").fetchone()
    service.index.close()


def test_semantic_failure_falls_back_explicitly_without_losing_keyword_search(indexed):
    """Provider failures leave lexical matches usable and cannot claim complete hybrid coverage."""
    service, _ = indexed
    service.sync_once()
    with pytest.raises(ValueError, match="disabled"):
        service.search(MailQuery(query="calendar", mode="semantic"))
    assert service.search(MailQuery(query="calendar")).mode == "keyword"
    enable_vectors(service)
    service.embedder.encode = Mock(side_effect=RuntimeError("PRIVATE_PROVIDER_BODY"))
    page = service.search(MailQuery(query="calendar"))
    assert page.results and page.mode == "keyword" and not page.complete
    assert page.index.embedding_error == "RuntimeError"
    assert "PRIVATE_PROVIDER_BODY" not in page.model_dump_json()
    with pytest.raises(ValueError, match="unavailable"):
        service.search(MailQuery(query="calendar", mode="semantic"))
    service.index.close()


def test_vectors_enforce_dimensions_and_skip_removed_chunks(indexed):
    """Concurrent source replacement cannot attach an old vector to a newly allocated chunk."""
    service, _ = indexed
    service.sync_once()
    enable_vectors(service)
    jobs = service.index.chunks()
    service.index.vectors(jobs, [[1.0, 0.0] for _ in jobs])
    with pytest.raises(ValueError, match="dimensions"):
        service.index.vectors(jobs, [[1.0] for _ in jobs])
    with pytest.raises(ValueError, match="dimensions"):
        service.index.search(MailQuery(query="x"), vector=[1.0], mode="semantic")
    for vectors in ([], [[0.0, 0.0] for _ in jobs], [[float("nan"), 1] for _ in jobs]):
        with pytest.raises(ValueError):
            service.index.vectors(jobs, vectors)
    service.index.db.execute("DELETE FROM chunks")
    service.index.db.commit()
    service.index.vectors(jobs, [[1.0, 0.0] for _ in jobs])
    service.index.close()


def test_truncated_body_coverage_and_candidate_limit_are_visible(indexed, monkeypatch):
    """Safety truncation and bounded ranked retrieval never masquerade as a complete audit."""
    service, _ = indexed
    monkeypatch.setattr(index, "MAX_BODY_CHARS", 12)
    monkeypatch.setattr(index, "CHUNK_CHARS", 4)
    monkeypatch.setattr(index, "MAX_CHUNKS", 1)
    service.sync_once()
    page = service.search(MailQuery(query="weekly", mode="keyword"))
    assert page.index.bodies_truncated == 2 and page.index.semantic_truncated == 2 and not page.complete
    assert service.index.cached(page.results[0].message.id) is None
    monkeypatch.setattr(index, "CANDIDATES", 1)
    assert service.search(MailQuery(query="weekly", mode="keyword")).candidate_limit_reached


def test_embedding_wire_format_and_response_validation():
    """The actual SDK sends NVIDIA passage/query roles and honors returned input indices."""
    requests = []

    def handle(request):
        """Return a deliberately reversed response without contacting any endpoint."""
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"object": "list", "model": "synthetic", "data": [
            {"object": "embedding", "index": i, "embedding": [1.0, float(i)]}
            for i in reversed(range(len(payload["input"])))], "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    settings = EmbeddingSettings(model="synthetic", base_url="https://embedding.test/v1", dimensions=2)
    embedder = Embedder(settings)
    embedder.client.close()
    embedder.client = OpenAI(api_key="test-key", base_url=settings.base_url, max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    try:
        assert embedder.encode(["one", "two"]) == [[1, 0], [1, 1]]
        embedder.encode(["query"], query=True)
        assert requests[0]["input_type"] == "passage" and requests[1]["input_type"] == "query"
        assert requests[0]["truncate"] == "NONE" and requests[0]["dimensions"] == 2
    finally:
        embedder.close()


@pytest.mark.parametrize("change", [{"provider": "bad"}, {"dimensions": -1}, {"dimensions": True},
    {"timeout": 0}, {"timeout": float("nan")}, {"model": "m"},
    {"model": "m", "base_url": "https://user:secret@example.test"}])
def test_invalid_embedding_settings(change):
    """Invalid endpoints, dimensions, and timeouts fail before any email can be transmitted."""
    with pytest.raises(ValueError):
        EmbeddingSettings(**change)


@pytest.mark.parametrize("change", [{"enabled": "true"}, {"journal_mode": "bad"}, {"sync_seconds": 0},
    {"embedding_seconds": float("inf")}, {"body_batch": 0}, {"header_batch": 2001}, {"flag_batch": True}, {"min_score": -1}])
def test_invalid_index_settings(change):
    """Invalid work budgets are rejected before worker startup."""
    with pytest.raises(ValueError):
        IndexSettings(**change)


def test_configuration_and_disabled_embedder(monkeypatch):
    """Explicit opt-in controls embedding transmission, and credentials do not enter repr/profile."""
    monkeypatch.setenv("IMAP_INDEX_ENABLED", "false")
    assert not IndexSettings.from_env().enabled
    monkeypatch.setenv("IMAP_INDEX_ENABLED", "invalid")
    with pytest.raises(ValueError):
        IndexSettings.from_env()
    monkeypatch.setenv("IMAP_EMBEDDING_API_KEY", "PRIVATE_KEY")
    settings = EmbeddingSettings.from_env()
    assert "PRIVATE_KEY" not in repr(settings)
    assert settings.profile == replace(settings, api_key="rotated").profile
    embedder = Embedder(settings)
    with pytest.raises(ValueError):
        embedder.encode(["must not transmit"])
    embedder.close()


def test_index_sync_failure_and_shutdown(indexed, monkeypatch):
    """Network failures retain explicit stale status; shutdown wakes a sleeping worker."""
    service, _ = indexed

    def fail():
        """End the worker after one synthetic network failure."""
        service.shutdown()
        raise RuntimeError("PRIVATE_ERROR")

    monkeypatch.setattr(service, "sync_once", fail)
    service.sync_loop()
    assert service.status().sync_error == "RuntimeError"
    assert service.stop.is_set() and service.wakeup.is_set()


def test_background_embedding_failure_is_deferred(indexed, monkeypatch):
    """Embedding rate limits do not stall IMAP synchronization or drop pending work."""
    service, _ = indexed
    service.sync_once()
    enable_vectors(service)
    error = RateLimitError("PRIVATE_ERROR", response=httpx.Response(429, headers={"Retry-After": "60"},
        request=httpx.Request("POST", "https://embedding.test")), body={"code": "rate_limit_exceeded"})

    def fail(texts):
        """Stop the synthetic loop after recording one rate-limited embedding batch."""
        service.shutdown()
        raise error

    monkeypatch.setattr(service.embedder, "encode", fail)
    service.embedding_loop()
    assert service.status().embedding_error == "RateLimitError"
    assert service.status().chunks_pending == 2
    assert not service.index.chunks()
    service.index.close()


@pytest.mark.asyncio
async def test_typed_index_tools_over_real_mcp(indexed, monkeypatch):
    """The view returns typed search/status objects directly and exposes explicit live search."""
    from fastmcp import Client, FastMCP

    service, _ = indexed
    service.sync_once()
    monkeypatch.setattr(server, "indexed", service)
    monkeypatch.setattr(server, "mailbox", service.mailbox)
    app = FastMCP("indexed-test")
    for handler in (server.search_mail, server.index_status, server.get_mail, server.search_server):
        app.tool(handler)
    async with Client(app) as client:
        result = await client.call_tool("search_mail", {"query": "calendar", "mode": "keyword"})
        assert result.structured_content["complete"]
        identifier = result.structured_content["results"][0]["message"]["id"]
        message = await client.call_tool("get_mail", {"message_id": identifier})
        assert message.structured_content["cached"]
        status = await client.call_tool("index_status", {})
        assert status.structured_content["indexed_messages"] == 2
    with pytest.raises(Exception, match="timezone"):
        server.search_mail(after=datetime(2026, 1, 1))
    monkeypatch.setattr(server, "indexed", None)
    with pytest.raises(Exception, match="disabled"):
        server.index_status()


def test_bulk_index_does_not_expand_notification_identity_tables(indexed):
    """Only a requested live read warms the small legacy location cache."""
    service, protocol = indexed
    service.sync_once()
    assert service.mailbox.identities.db.execute("SELECT count(*) FROM mail_identity_locations").fetchone()[0] == 0
    identifier = ids(service.search(MailQuery(subject="weekly")))[0]
    searches = len(protocol.searches)
    service.get(identifier, refresh=True)
    assert len(protocol.searches) == searches
    assert service.mailbox.identities.locations(model.NativeKey.decode(identifier))


def test_attachment_fetches_use_indexed_locations_without_server_search(indexed):
    """The original attachment URI still resolves bytes through verified live IMAP reads."""
    from test_imap_identity import mail

    service, protocol = indexed
    protocol.messages["INBOX"][7] = (mail("recent"), "Recent")
    protocol.messages["Archive"].pop(434)
    service.sync_once()
    identifier = ids(service.search(MailQuery(subject="Synthetic")))[0]
    searches = len(protocol.searches)
    raw, metadata = service.attachment(identifier, "0")
    assert raw == b"attachment bytes" and identifier in metadata.uri
    assert len(protocol.searches) == searches


def test_generic_server_ids_are_not_promoted_or_cross_folder_deduplicated(indexed):
    """A generic server retains folder/epoch/UID identity rather than inventing content-based IDs."""
    service, protocol = indexed
    protocol.identification = ((b"name", b"Generic IMAP"),)
    service.sync_once()
    page = service.search(MailQuery(subject="weekly"))
    assert len(page.results) == 2 and all(hit.message.identity_kind == "mailbox" for hit in page.results)
    assert not service.get(page.results[0].message.id, refresh=True).cached


def test_reconcile_rollback_does_not_commit_a_partial_header_batch(indexed):
    """Invalid observations roll back preceding headers, FTS changes, and inventory updates."""
    service, _ = indexed
    service.sync_once()
    message = service.search(MailQuery(subject="weekly")).results[0].message
    key = model.MessageKey(account=service.mailbox.settings.account, folder="INBOX", validity=21, uid=7)
    observation = HeaderObservation(location=key, message=message.model_copy(update={"subject": "changed"}), flags=[], size=100)
    wrong = observation.model_copy(update={"location": key.model_copy(update={"validity": 99})})
    with pytest.raises(ValueError, match="epoch"):
        service.index.reconcile("INBOX", 21, [7], [observation, wrong], {})
    assert service.search(MailQuery(subject="weekly")).results
    assert not service.search(MailQuery(subject="changed")).results
    omitted = observation.model_copy(update={"location": key.model_copy(update={"uid": 8})})
    service.index.reconcile("INBOX", 21, [7], [omitted], {})
    assert not service.search(MailQuery(subject="changed")).results


def test_removed_mail_is_purged_from_vector_and_text_indices(indexed):
    """Deleting all observed copies removes both search representations and cached bodies."""
    service, protocol = indexed
    service.sync_once()
    enable_vectors(service)
    jobs = service.index.chunks()
    service.index.vectors(jobs, [[1.0, 0.0] for _ in jobs])
    protocol.messages = {"INBOX": {}, "Archive": {}}
    service.sync_once()
    assert not service.index.db.execute("SELECT rowid FROM vectors").fetchall()
    assert not service.search(MailQuery(query="calendar", mode="keyword")).results
    service.index.close()


@pytest.mark.parametrize("fault", ["missing_record", "missing_header", "missing_size", "missing_flags", "epoch"])
def test_partial_metadata_and_epoch_races_do_not_publish_false_coverage(indexed, monkeypatch, fault):
    """Incomplete FETCH responses and changed epochs cannot create a complete but wrong snapshot."""
    service, protocol = indexed
    original = protocol.fetch

    def fetch(uids, fields):
        """Alter the selected folder's initial header response only."""
        rows = original(uids, fields)
        if "BODY.PEEK[HEADER]" in fields and protocol.selected == "INBOX":
            if fault == "missing_record":
                return {}
            if fault == "epoch":
                protocol.epochs["INBOX"] += 1
            else:
                key = {"missing_header": b"BODY[HEADER]", "missing_size": b"RFC822.SIZE", "missing_flags": b"FLAGS"}[fault]
                for row in rows.values():
                    row.pop(key)
        return rows

    monkeypatch.setattr(protocol, "fetch", fetch)
    service.sync_once()
    assert not service.search(MailQuery(query="calendar", mode="keyword")).complete


@pytest.mark.parametrize("when", ["before", "after"])
def test_epoch_change_during_body_fetch_is_not_cached(indexed, monkeypatch, when):
    """A body must retain the indexed epoch through its entire fetch operation."""
    service, protocol = indexed
    original_jobs = service.index.body_jobs
    original_read = service.mailbox._read_selected

    def jobs(limit):
        """Switch epochs immediately before body retrieval when requested."""
        result = original_jobs(limit)
        if when == "before":
            for folder in protocol.epochs:
                protocol.epochs[folder] += 1
        return result

    def read(client, location, identity):
        """Switch epochs immediately after the server returns a body when requested."""
        result = original_read(client, location, identity)
        if when == "after":
            protocol.epochs[location.folder] += 1
        return result

    monkeypatch.setattr(service.index, "body_jobs", jobs)
    monkeypatch.setattr(service.mailbox, "_read_selected", read)
    service.sync_once()
    assert service.status().bodies_pending == 2
    assert not service.index.db.execute("SELECT id FROM mail WHERE details IS NOT NULL").fetchall()


def test_body_bytes_are_bounded_per_sweep(indexed):
    """A batch count does not permit downloading many maximum-sized MIME messages at once."""
    service, protocol = indexed
    service.mailbox.settings = replace(service.mailbox.settings, max_message_bytes=500)
    service.sync_once()
    assert sum("BODY.PEEK[]" in fields for _, _, fields in protocol.fetches) == 1
    assert service.status().bodies_pending == 1


@pytest.mark.parametrize("stop_at", ["folders", "headers", "flags", "bodies"])
def test_sync_shutdown_is_cooperative(indexed, monkeypatch, stop_at):
    """Shutdown prevents committing partial observations or starting further body work."""
    service, protocol = indexed
    if stop_at == "folders":
        service.shutdown()
    elif stop_at == "headers":
        original = protocol.fetch

        def fetch(uids, fields):
            """Signal shutdown as the header request returns."""
            result = original(uids, fields)
            service.shutdown()
            return result

        monkeypatch.setattr(protocol, "fetch", fetch)
    elif stop_at == "flags":
        service.sync_once()
        service.index.db.execute("UPDATE locations SET flags_at=0")
        service.index.db.commit()
        original = protocol.fetch

        def fetch(uids, fields):
            """Signal shutdown as flag refresh returns."""
            result = original(uids, fields)
            service.shutdown()
            return result

        monkeypatch.setattr(protocol, "fetch", fetch)
    else:
        original = service.index.body_jobs

        def jobs(limit):
            """Signal shutdown just before any body is fetched."""
            result = original(limit)
            service.shutdown()
            return result

        monkeypatch.setattr(service.index, "body_jobs", jobs)
    service.sync_once()
    assert service.stop.is_set()


def test_successful_background_loops_and_idle_hint(indexed, monkeypatch):
    """Workers make bounded progress, and monitoring wakes indexing without creating a notification."""
    from imap_mcp.events import EventStore, Monitor
    import tempfile

    service, protocol = indexed
    service.settings = replace(service.settings, sync_seconds=0.001)
    monkeypatch.setattr(service.wakeup, "wait", lambda seconds: service.shutdown())
    monkeypatch.setattr(service.stop, "wait", lambda seconds: False)
    service.sync_loop()
    assert service.status().indexed_messages == 2
    service.stop.clear()
    enable_vectors(service)
    service.embedder.encode = lambda texts: [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(service.stop, "wait", lambda seconds: service.shutdown())
    service.embedding_loop()
    assert service.status().chunks_pending == 0
    service.index.close()
    with tempfile.TemporaryDirectory() as directory:
        events = EventStore(Path(directory) / "events.sqlite3")
        hint = Mock()
        monitor = Monitor(service.mailbox, events, on_change=hint)
        monkeypatch.setattr(monitor, "wait_for_change", lambda client, idle: monitor.stop.set())
        monitor.watch("INBOX")
        hint.assert_called_once_with()
        assert not events.pending(service.mailbox.settings.account)
        events.close()


@pytest.mark.parametrize("headers, expected", [
    ({}, 30), ({"retry-after-ms": "45000"}, 45), ({"retry-after": "90"}, 90),
    ({"retry-after": "bad"}, 30), ({"retry-after": "inf"}, 30),
    ({"retry-after": format_datetime(NOW + timedelta(seconds=60))}, 60),
])
def test_embedding_backoff_honors_server_hints(monkeypatch, headers, expected):
    """Missing or malformed hints use a conservative retry delay instead of a busy loop."""
    monkeypatch.setattr(sync, "time", SimpleNamespace(time=lambda: NOW.timestamp()))
    assert sync._retry_delay(SimpleNamespace(response=SimpleNamespace(headers=headers))) == expected


@pytest.mark.parametrize("rows", [[], [(1, [1.0, 0.0])], [(0, [])], [(0, [0.0, 0.0])],
                                   [(0, [float("nan"), 1.0])], [(0, [1.0])]])
def test_invalid_embedding_results_cannot_enter_vector_index(rows):
    """Missing indices, wrong dimensions, and non-finite/zero outputs are rejected."""
    embedder = Embedder(EmbeddingSettings(model="m", base_url="https://embedding.test", dimensions=2, provider="openai"))
    embedder.client.close()
    embedder.client = Mock()
    embedder.client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=vector) for i, vector in rows])
    with pytest.raises(ValueError):
        embedder.encode(["text"])
    embedder.close()


@pytest.mark.asyncio
async def test_enabled_index_lifespan_starts_and_joins_workers(indexed, monkeypatch, tmp_path):
    """The actual MCP lifespan owns a separate index and stops workers before closing stores."""
    from fastmcp import Client
    from imap_mcp.events import Monitor
    import asyncio

    service, _ = indexed
    monkeypatch.setattr(server.Settings, "from_env", lambda: service.mailbox.settings)
    monkeypatch.setattr(server, "Mailbox", lambda settings, state_path=None: service.mailbox)
    monkeypatch.setenv("IMAP_INDEX_ENABLED", "true")
    monkeypatch.setenv("IMAP_STATE_PATH", str(tmp_path / "events.sqlite3"))
    monkeypatch.setenv("IMAP_INDEX_DIR", str(tmp_path / "derived"))
    monkeypatch.setattr(Monitor, "watch", lambda self, folder: self.stop.wait(5))
    async with Client(fresh_mcp()) as client:
        async with asyncio.timeout(5):
            while True:
                state = await client.call_tool("index_status", {})
                if state.structured_content["indexed_messages"] == 2:
                    break
                await asyncio.sleep(0.01)
        result = await client.call_tool("search_mail", {"subject": "weekly"})
        assert len(result.structured_content["results"]) == 1
    assert server.indexed is None and server.mailbox is None


def test_index_permissions_and_storage_error_status(indexed):
    """Mail text is private on disk and failed diagnostic writes cannot hide storage failure."""
    service, _ = indexed
    service.sync_once()
    path = Path(service.index.db.execute("PRAGMA database_list").fetchone()[2])
    assert path.stat().st_mode & 0o777 == 0o600
    service.index.db.execute("CREATE TRIGGER full_metadata BEFORE INSERT ON meta BEGIN SELECT RAISE(ABORT,'full'); END")
    service.index.db.commit()
    service.index.error("OperationalError")
    service.index.embedding_error([], "OperationalError", 30)
    state = service.status()
    assert state.stale and state.sync_error == "OperationalError" and state.embedding_error == "OperationalError"
    service.index.db.execute("DROP TRIGGER full_metadata")
    service.index.db.commit()
    service.sync_once()
    assert service.status().sync_error is None


def test_wal_index_and_profile_reopening_are_supported_on_local_storage(tmp_path):
    """Explicit local-storage WAL mode preserves derived rows when reopening the same profile."""
    path = tmp_path / "private" / "index.sqlite3"
    store = MailIndex(path, "account", journal_mode="WAL")
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    store.folders([])
    store.close()
    reopened = MailIndex(path, "account", journal_mode="WAL")
    assert reopened.status().available
    reopened.close()
