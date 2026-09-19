"""IMAP IDLE notifications with persistent checkpoints and an HTTP delivery outbox."""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time

import httpx
from imapclient import IMAPClient

from imap_mcp.model import IdentityCapabilities, Mailbox, MessageKey, notification_date, received_time

logger = logging.getLogger(__name__)


class EventStore:
    """Thread-safe SQLite checkpoints and pending events, committed together.

    Args:
        path: Persistent database file. Parent directories are created.

    Raises:
        sqlite3.Error: If the store cannot be opened or updated.
    """
    def __init__(self, path: Path):
        """Open the database and initialize its tables without discarding existing state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                account TEXT NOT NULL, folder TEXT NOT NULL, validity INTEGER NOT NULL,
                uid INTEGER NOT NULL, PRIMARY KEY(account, folder));
            CREATE TABLE IF NOT EXISTS outbox (
                id TEXT PRIMARY KEY, account TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS notification_floors (
                account TEXT NOT NULL, folder TEXT NOT NULL, since TEXT NOT NULL,
                PRIMARY KEY(account, folder));
        """)

    def notification_floor(self, account: str, folder: str) -> datetime:
        """Return a persisted first-use cutoff without resetting it on restart or UID changes."""
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO notification_floors VALUES (?, ?, ?)",
                            (account, folder, datetime.now(timezone.utc).isoformat()))
            value = self.db.execute("SELECT since FROM notification_floors WHERE account=? AND folder=?",
                                    (account, folder)).fetchone()[0]
            return datetime.fromisoformat(value)

    def checkpoint(self, account: str, folder: str, validity: int, baseline: int) -> int:
        """Return the cursor, baselining first use or a UIDVALIDITY reset without replay.

        Args:
            account: Opaque account identity.
            folder: Exact selected folder.
            validity: Current server UIDVALIDITY.
            baseline: UIDNEXT minus one from the initial folder selection.

        Returns:
            Highest UID already staged, or the initial baseline for a new epoch.
            Existing undelivered events remain queued across epoch changes.
        """
        with self.lock, self.db:
            row = self.db.execute("SELECT validity, uid FROM checkpoints WHERE account=? AND folder=?",
                                  (account, folder)).fetchone()
            if row is not None and row[0] == validity:
                return row[1]
            self.db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?)",
                            (account, folder, validity, baseline))
            if row is not None:
                logger.warning("Mailbox UIDVALIDITY changed; existing contents were not replayed")
            return baseline

    def stage(self, key: MessageKey, event: dict | None) -> None:
        """Atomically enqueue an event and advance its cursor; None records an expunged UID.

        Args:
            key: Message identity in the currently selected epoch.
            event: Canonical session event, or None for mail removed before fetch.

        Raises:
            RuntimeError: If the checkpoint belongs to a different mailbox epoch.
        """
        payload = json.dumps(event) if event is not None else None
        with self.lock, self.db:
            row = self.db.execute("SELECT validity, uid FROM checkpoints WHERE account=? AND folder=?",
                                  (key.account, key.folder)).fetchone()
            if row is None or row[0] != key.validity:
                raise RuntimeError("Mailbox checkpoint changed during staging")
            if key.uid <= row[1]:
                return
            if payload is not None:
                self.db.execute("INSERT INTO outbox VALUES (?, ?, ?)", (key.encode(), key.account, payload))
            self.db.execute("UPDATE checkpoints SET uid=? WHERE account=? AND folder=?",
                            (key.uid, key.account, key.folder))

    def pending(self, account: str, limit: int = 100) -> list[tuple[str, dict]]:
        """Return a bounded FIFO page for one account; other accounts remain untouched."""
        with self.lock:
            return [(identifier, json.loads(payload)) for identifier, payload in self.db.execute(
                "SELECT id, payload FROM outbox WHERE account=? ORDER BY rowid LIMIT ?", (account, limit))]

    def acknowledge(self, identifier: str) -> None:
        """Remove an event after HTTP acceptance; repeated acknowledgements are harmless."""
        with self.lock, self.db:
            self.db.execute("DELETE FROM outbox WHERE id=?", (identifier,))

    def close(self) -> None:
        """Close the store after all monitoring and delivery workers have stopped."""
        with self.lock:
            self.db.close()


class Monitor:
    """Watch selected folders and push new-mail summaries without agent polling.

    Args:
        mailbox: Account used by read tools, with separate watcher connections.
        store: Durable outbox and checkpoints.
        session_id: Optional explicit event route; blank uses the platform default.
        poll_seconds: Fallback interval only for servers without IDLE.
        retry_seconds: Delay after connection/authentication/scan failures.
        energy: Passive by default; active preempts the agent's current turn.

    Raises:
        ValueError: For invalid energy or intervals.
    """
    def __init__(self, mailbox: Mailbox, store: EventStore, session_id: str = "",
                 poll_seconds: float = 60, retry_seconds: float = 5, energy: str = "passive"):
        """Configure workers without opening IMAP or HTTP connections."""
        if energy not in {"passive", "active"} or min(poll_seconds, retry_seconds) <= 0:
            raise ValueError("Invalid IMAP event energy or intervals")
        self.mailbox, self.store = mailbox, store
        self.session_id, self.energy = session_id, energy
        self.poll_seconds, self.retry_seconds = poll_seconds, retry_seconds
        self.stop = threading.Event()
        for folder in mailbox.settings.folders:
            self.notification_floor(folder)

    def notification_floor(self, folder: str) -> datetime | None:
        """Return the configured receipt-date floor for one folder, or None in all mode."""
        configured = self.mailbox.settings.notify_since
        if configured == "startup":
            return self.store.notification_floor(self.mailbox.settings.account, folder)
        return notification_date(configured)

    def scan(self, client, folder: str, validity: int, baseline: int,
             capabilities: IdentityCapabilities | None = None) -> None:
        """Stage unseen UIDs in order; duplicates and messages expunged before fetch are skipped.

        Args:
            client: Authenticated, read-only selected client, not currently in IDLE.
            folder: Selected folder name.
            validity: Selected folder epoch.
            baseline: UIDNEXT minus one at initial selection.
            capabilities: Connection's negotiated identity support, or None to discover it.

        Raises:
            Exception: IMAP, parsing, or storage failure; cursor stops at the last committed UID.
        """
        account = self.mailbox.settings.account
        capabilities = capabilities or self.mailbox.identity_capabilities(client)
        last_uid = self.store.checkpoint(account, folder, validity, baseline)
        floor = self.notification_floor(folder)
        # UID n:* can return the current maximum even when it is smaller than n.
        uids = sorted(uid for uid in client.search(["UID", f"{last_uid + 1}:*"]) if uid > last_uid)
        for uid in uids:
            if self.stop.is_set():
                return
            key = MessageKey(account=account, folder=folder, validity=validity, uid=uid)
            record = client.fetch([uid], ["BODY.PEEK[HEADER]", "INTERNALDATE", *capabilities.fetch_fields]).get(uid)
            event = None
            received = received_time(record.get(b"INTERNALDATE")) if record is not None else None
            if record is not None and (floor is None or (received is not None and received >= floor)):
                message = self.mailbox.summarize(key, record, capabilities)
                summary = message.model_dump(mode="json", exclude_none=True)
                summary["subject"] = summary["subject"][:500]
                summary["from_"] = {name: value[:500] for name, value in summary["from_"].items()}
                event = {"source": "imap", "event_type": "mail_received", "energy": self.energy,
                         "text": json.dumps(summary), "metadata": {
                             "event_id": key.encode(), "message_id": message.id,
                             "account": account, "folder": folder,
                             "uidvalidity": validity, "uid": uid,
                             "received_at": received.isoformat() if received else None}}
                if self.session_id:
                    event["session_id"] = self.session_id
            self.store.stage(key, event)

    def wait_for_change(self, client, idle: bool) -> None:
        """Wait for IDLE responses, refreshing within 25 minutes, or use the fallback interval."""
        if not idle:
            self.stop.wait(self.poll_seconds)
            if not self.stop.is_set():
                client.noop()
            return
        client.idle()
        try:
            deadline = time.monotonic() + 25 * 60
            while not self.stop.is_set() and time.monotonic() < deadline:
                if client.idle_check(timeout=5):
                    return
        finally:
            client.idle_done()

    def watch(self, folder: str) -> None:
        """Watch one folder until stop is set, reconnecting and catching up after failures."""
        while not self.stop.is_set():
            try:
                with self.mailbox.connect() as client:
                    selected = client.select_folder(folder, readonly=True)
                    validity = int(selected[b"UIDVALIDITY"])
                    baseline = int(selected[b"UIDNEXT"]) - 1
                    capabilities = self.mailbox.identity_capabilities(client)
                    idle = client.has_capability("IDLE")
                    if not idle:
                        logger.warning("IMAP IDLE unavailable; using adapter-side polling")
                    while not self.stop.is_set():
                        self.scan(client, folder, validity, baseline, capabilities)
                        self.wait_for_change(client, idle)
            except Exception as error:
                logger.warning("IMAP watcher reconnecting after %s", type(error).__name__)
                self.stop.wait(self.retry_seconds)

    async def deliver_once(self, http: httpx.AsyncClient, url: str) -> int:
        """POST one pending page and acknowledge only successful HTTP responses.

        Args:
            http: HTTP client with finite timeouts.
            url: Session manager base URL.

        Returns:
            Number processed, including notifications suppressed by the current
            cutoff. HTTP acceptance is not confirmation of agent processing.

        Raises:
            httpx.HTTPError: Failed/ambiguous requests stay queued for retry. A lost
                response can cause repeat delivery with the same metadata.event_id.
        """
        pending = self.store.pending(self.mailbox.settings.account)
        for identifier, event in pending:
            floor = self.notification_floor(event["metadata"]["folder"])
            if floor is not None:
                received = received_time(event["metadata"].get("received_at"))
                if received is None:
                    try:
                        received = await asyncio.to_thread(self.mailbox.received_at, identifier)
                    except ValueError:
                        received = None
                if received is None or received < floor:
                    self.store.acknowledge(identifier)
                    continue
            response = await http.post(f"{url.rstrip('/')}/event", json=event)
            response.raise_for_status()
            self.store.acknowledge(identifier)
        return len(pending)

    async def deliver(self, http: httpx.AsyncClient, url: str) -> None:
        """Drain the outbox until stopped, retrying failed HTTP deliveries without data loss."""
        while not self.stop.is_set():
            try:
                count = await self.deliver_once(http, url)
            except (httpx.HTTPError, sqlite3.Error, OSError, IMAPClient.Error) as error:
                logger.warning("IMAP event delivery retry after %s", type(error).__name__)
                count = 0
            if not count:
                await asyncio.to_thread(self.stop.wait, self.retry_seconds)
