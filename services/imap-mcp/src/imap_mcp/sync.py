"""Read-only IMAP synchronization and indexed query orchestration."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
import logging
import math
import os
import threading
import time

from imap_mcp.embedding import Embedder
from imap_mcp.identity import NativeKey, decode_id
from imap_mcp.index import CachedMessage, HeaderObservation, IndexStatus, MailIndex, MailQuery, SearchPage
from imap_mcp.model import AttachmentMetadata, EmailAddress, Mailbox, MessageKey, parse_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexSettings:
    """Bounded background synchronization policy, independent of mail notification cutoffs.

    Args:
        enabled: Enable indexed search and background synchronization.
        sync_seconds: Minimum delay between bounded indexing sweeps.
        header_batch: New headers per folder per sweep.
        body_batch: Full bodies per sweep, across all folders.
        flag_batch: Existing flag observations refreshed per folder per sweep.
        embedding_seconds: Minimum delay between embedding batches.
        min_score: Minimum cosine similarity admitted to semantic search results.
        journal_mode: SQLite journal policy for the separately stored index.
    """

    enabled: bool = True
    sync_seconds: float = 15
    header_batch: int = 100
    body_batch: int = 10
    flag_batch: int = 1000
    embedding_seconds: float = 5
    min_score: float = 0.5
    journal_mode: str = "DELETE"

    def __post_init__(self):
        """Reject unbounded or malformed policies before workers start."""
        if type(self.enabled) is not bool or self.journal_mode not in {"DELETE", "WAL"}:
            raise ValueError("Invalid mail index enablement/journal mode")
        for value in (self.sync_seconds, self.embedding_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Index intervals must be finite and positive")
        for value in (self.header_batch, self.body_batch, self.flag_batch):
            if type(value) is not int or not 1 <= value <= 2000:
                raise ValueError("Index batches must be integers from 1 to 2000")
        if not math.isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ValueError("Embedding minimum score must be between 0 and 1")

    @classmethod
    def from_env(cls) -> "IndexSettings":
        """Read IMAP_INDEX_* limits; malformed environment settings raise ValueError."""
        enabled = os.getenv("IMAP_INDEX_ENABLED", "true").lower()
        if enabled not in {"true", "false"}:
            raise ValueError("IMAP_INDEX_ENABLED must be true or false")
        return cls(enabled=enabled == "true", sync_seconds=float(os.getenv("IMAP_INDEX_SYNC_SECONDS", "15")),
                   header_batch=int(os.getenv("IMAP_INDEX_HEADER_BATCH", "100")),
                   body_batch=int(os.getenv("IMAP_INDEX_BODY_BATCH", "10")),
                   flag_batch=int(os.getenv("IMAP_INDEX_FLAG_BATCH", "1000")),
                   embedding_seconds=float(os.getenv("IMAP_INDEX_EMBEDDING_SECONDS", "5")),
                   min_score=float(os.getenv("IMAP_EMBEDDING_MIN_SCORE", "0.5")),
                   journal_mode=os.getenv("IMAP_INDEX_JOURNAL_MODE", "DELETE").upper())


def _flags(record: dict) -> list[str]:
    """Decode a required FLAGS response; omission must not be mistaken for unread mail."""
    if b"FLAGS" not in record:
        raise RuntimeError("IMAP omitted requested flags")
    return [value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
            for value in record[b"FLAGS"]]


def _retry_delay(error: Exception) -> float:
    """Choose a background retry delay, honoring positive Retry-After hints when available."""
    headers = getattr(getattr(error, "response", None), "headers", {})
    for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(name)
        try:
            delay = float(value) / divisor
        except (TypeError, ValueError):
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time() if name == "retry-after" and value else 0
            except (TypeError, ValueError, OverflowError):
                delay = 0
        if math.isfinite(delay) and delay > 0:
            return max(30, delay)
    return 30


class IndexedMailbox:
    """Coordinate a local read model without changing IMAP or notification checkpoints.

    Args:
        mailbox: Existing read-only backend and native identity resolver.
        index: Separate rebuildable search database.
        embedder: Explicitly configured optional embedding client.
        settings: Bounded background work policy.
    """

    def __init__(self, mailbox: Mailbox, index: MailIndex, embedder: Embedder, settings: IndexSettings):
        """Initialize worker signals without network calls or historical notifications."""
        self.mailbox, self.index, self.embedder, self.settings = mailbox, index, embedder, settings
        self.stop = threading.Event()
        self.wakeup = threading.Event()

    def status(self) -> IndexStatus:
        """Return local indexing and embedding coverage without querying IMAP."""
        return self.index.status(self.embedder.settings.model)

    def search(self, query: MailQuery) -> SearchPage:
        """Search local data; semantic failures explicitly fall back to keyword results.

        Args:
            query: Validated text and structured search scope.

        Returns:
            Typed results, actual retrieval mode, warnings, and coverage.

        Raises:
            ValueError: Explicit semantic mode is unavailable or filters are invalid.
        """
        vector, warnings, mode = None, [], "keyword"
        if query.query.strip() and query.mode != "keyword":
            if not self.embedder.settings.model:
                if query.mode == "semantic":
                    raise ValueError("Semantic search is disabled; configure IMAP_EMBEDDING_MODEL and endpoint")
                warnings.append("Semantic search is disabled; returned keyword matches")
            else:
                try:
                    vector = self.embedder.encode([query.query], query=True)[0]
                    mode = query.mode
                except Exception as error:
                    self.index.embedding_error([], type(error).__name__, _retry_delay(error))
                    if query.mode == "semantic":
                        raise ValueError("Semantic query encoding is unavailable; keyword search remains usable") from None
                    warnings.append("Semantic query encoding failed; returned keyword matches")
        return self.index.search(query, vector=vector, mode=mode, model=self.embedder.settings.model,
                                 warnings=warnings, min_score=self.settings.min_score)

    def get(self, message_id: str, refresh: bool = False) -> CachedMessage:
        """Return cached details with freshness, or explicitly fetch the current server version.

        Args:
            message_id: Existing native/folder-scoped public mail ID.
            refresh: Bypass local body cache, including a stale but usable snapshot.

        Raises backend lookup/size errors on a live read; attachments remain live resources.
        """
        if not refresh and (cached := self.index.cached(message_id)) is not None:
            return cached
        self._warm_location(message_id)
        raw, location, receipt = self.mailbox.raw_message(message_id)
        identity = decode_id(message_id)
        message = parse_message(raw, location, received_at=receipt,
                                identity=identity if isinstance(identity, NativeKey) else None)
        self.index.save_body(message, location)
        return CachedMessage.live(message)

    def _warm_location(self, message_id: str) -> None:
        """Populate only a requested native ID's live lookup hints from the bulk index."""
        identity = decode_id(message_id)
        if identity.account != self.mailbox.settings.account:
            raise ValueError("Message ID belongs to another account")
        if isinstance(identity, NativeKey):
            for location in self.index.locations(message_id):
                self.mailbox.identities.remember(location, identity.kind, identity.value)

    def attachment(self, message_id: str, attachment_id: str) -> tuple[bytes, AttachmentMetadata]:
        """Fetch original attachment bytes using indexed location hints, validated by the backend."""
        self._warm_location(message_id)
        return self.mailbox.attachment(message_id, attachment_id)

    def _sync_folder(self, client, name: str, capabilities) -> None:
        """Refresh one complete UID inventory and bounded header/flag batches, checking its epoch twice."""
        selected = client.select_folder(name, readonly=True)
        validity = int(selected[b"UIDVALIDITY"])
        uids = client.search(["ALL"])
        uid_set = set(uids)
        known = self.index.known(name, validity)
        missing = uid_set - known
        recent = set()
        if len(missing) > self.settings.header_batch:
            recent = set(client.search(["SINCE", (datetime.now(timezone.utc) - timedelta(days=30)).date()]))
        ordered = sorted(missing, key=lambda uid: (uid in recent, uid), reverse=True)[:self.settings.header_batch]
        observations = []
        if ordered:
            records = client.fetch(ordered, ["BODY.PEEK[HEADER]", "INTERNALDATE", "RFC822.SIZE", "FLAGS", *capabilities.fetch_fields])
            for uid in ordered:
                if self.stop.is_set():
                    return
                record = records.get(uid)
                if record is None:
                    continue
                if b"BODY[HEADER]" not in record or b"RFC822.SIZE" not in record:
                    raise RuntimeError("IMAP omitted requested header/size metadata")
                key = MessageKey(account=self.mailbox.settings.account, folder=name, validity=validity, uid=uid)
                message = self.mailbox.summarize(key, record, capabilities, cache=False)
                headers = BytesParser(policy=policy.default).parsebytes(record[b"BODY[HEADER]"], headersonly=True)
                for field in ("to", "cc"):
                    setattr(message, field, [EmailAddress(name=n, address=a) for n, a in
                        getaddresses([str(v) for v in headers.get_all(field, [])])])
                observations.append(HeaderObservation(location=key, message=message, flags=_flags(record), size=int(record[b"RFC822.SIZE"])))
        flag_uids = [uid for uid in self.index.flag_jobs(name, validity, self.settings.flag_batch) if uid in uid_set]
        flags = {uid: _flags(record) for uid, record in client.fetch(flag_uids, ["FLAGS"]).items()} if flag_uids else {}
        if self.stop.is_set():
            return
        if int(client.select_folder(name, readonly=True)[b"UIDVALIDITY"]) != validity:
            raise RuntimeError("Mailbox epoch changed during indexing")
        self.index.reconcile(name, validity, uids, observations, flags)

    def sync_once(self) -> None:
        """Perform one bounded, resumable sweep without emitting mail_received events.

        Folder failures retain their last valid snapshot and do not stop other
        folders. Bodies are prioritized by observed receipt date and read with PEEK.
        """
        with self.mailbox.connect() as client:
            names = [name for flags, _, name in client.list_folders() if b"\\Noselect" not in flags]
            self.index.folders(names)
            capabilities = self.mailbox.identity_capabilities(client)
            for name in names:
                if self.stop.is_set():
                    return
                try:
                    self._sync_folder(client, name, capabilities)
                except Exception as error:
                    self.index.error(type(error).__name__, name)
                    logger.warning("Mail index folder refresh failed: %s", type(error).__name__)
            self.index.prune()
            downloaded = 0
            for job in self.index.body_jobs(self.settings.body_batch):
                if self.stop.is_set():
                    return
                if downloaded and downloaded + job.size > self.mailbox.settings.max_message_bytes:
                    break
                try:
                    selected = client.select_folder(job.location.folder, readonly=True)
                    if int(selected[b"UIDVALIDITY"]) != job.location.validity:
                        raise RuntimeError("Mailbox epoch changed before body indexing")
                    identity = decode_id(job.message_id)
                    expected = identity if isinstance(identity, NativeKey) else None
                    raw, location, receipt = self.mailbox._read_selected(client, job.location, expected)
                    downloaded += len(raw)
                    if int(client.select_folder(location.folder, readonly=True)[b"UIDVALIDITY"]) != location.validity:
                        raise RuntimeError("Mailbox epoch changed during body indexing")
                    message = parse_message(raw, location, received_at=receipt, identity=expected)
                    self.index.save_body(message, location)
                except Exception as error:
                    oversized = isinstance(error, ValueError) and str(error) == "Message exceeds IMAP_MAX_MESSAGE_BYTES"
                    self.index.body_error(job.message_id, unavailable=oversized)
                    logger.warning("Mail index body unavailable: %s", "size_limit" if oversized else type(error).__name__)

    def sync_loop(self) -> None:
        """Synchronize until stopped, waking early for IDLE hints without busy-spinning."""
        while not self.stop.is_set():
            self.wakeup.clear()
            try:
                self.sync_once()
            except Exception as error:
                self.index.error(type(error).__name__)
                logger.warning("Mail index refresh failed: %s", type(error).__name__)
            if self.stop.wait(1):
                return
            self.wakeup.wait(self.settings.sync_seconds)

    def embedding_loop(self) -> None:
        """Process bounded embedding batches independently of IMAP sync and notifications."""
        while not self.stop.is_set() and self.embedder.settings.model:
            delay = self.settings.embedding_seconds
            jobs = []
            try:
                jobs = self.index.chunks()
                if jobs:
                    self.index.vectors(jobs, self.embedder.encode([job.text for job in jobs]))
            except Exception as error:
                delay = _retry_delay(error)
                self.index.embedding_error(jobs, type(error).__name__, delay)
                logger.warning("Mail embeddings deferred: %s", type(error).__name__)
            self.stop.wait(delay)

    def shutdown(self) -> None:
        """Wake both workers so lifecycle cleanup can join them before closing storage."""
        self.stop.set()
        self.wakeup.set()
