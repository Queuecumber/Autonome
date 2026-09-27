"""Typed local mail search over a rebuildable SQLite full-text/vector index."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import sqlite_vec

from imap_mcp.model import Message, MessageKey, decode_id

MAX_BODY_CHARS = 500_000
MAX_CHUNKS = 256
CHUNK_CHARS = 1600
CANDIDATES = 1000


class MailQuery(BaseModel):
    """Literal text plus structured filters; dates are inclusive-after/exclusive-before UTC instants."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(default="", max_length=4096)
    folder: str | None = None
    sender: str | None = Field(default=None, max_length=500)
    recipient: str | None = Field(default=None, max_length=500)
    subject: str | None = Field(default=None, max_length=1000)
    after: datetime | None = None
    before: datetime | None = None
    unread: bool | None = None
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=1_000_000)

    @field_validator("after", "before")
    @classmethod
    def aware_date(cls, value):
        """Reject ambiguous local dates and normalize aware values to UTC."""
        if value is not None and value.utcoffset() is None:
            raise ValueError("Search dates must include a timezone")
        return value.astimezone(timezone.utc) if value else None

    @model_validator(mode="after")
    def date_order(self):
        """Reject an empty or reversed date window before evaluating a query."""
        if self.after and self.before and self.after >= self.before:
            raise ValueError("after must precede before")
        return self


class FolderStatus(BaseModel):
    """Observed folder inventory, header/flag coverage, and safe refresh diagnostics."""

    folder: str
    messages: int
    headers_pending: int
    flags_stale: int
    as_of: datetime | None
    error: str | None = None


class IndexStatus(BaseModel):
    """Index coverage is observational; incomplete results never prove server-side absence."""

    available: bool
    stale: bool
    indexed_messages: int
    bodies_pending: int
    bodies_unavailable: int
    bodies_truncated: int
    chunks_pending: int
    semantic_truncated: int
    embedding_model: str | None = None
    embedding_error: str | None = None
    sync_error: str | None = None
    folders: list[FolderStatus]


class SearchHit(BaseModel):
    """A bounded summary, current observed folders, and optional ranked text excerpt."""

    message: Message
    folders: list[str]
    snippet: str | None = None
    score: float | None = None
    semantic_score: float | None = None


class SearchPage(BaseModel):
    """One search page with explicit fallback, coverage, and candidate-window limits."""

    results: list[SearchHit]
    next_offset: int | None = None
    mode: Literal["keyword", "hybrid", "semantic"]
    complete: bool
    candidate_limit_reached: bool = False
    warnings: list[str] = Field(default_factory=list)
    index: IndexStatus


class CachedMessage(Message):
    """Mail details read from the local snapshot, with explicit freshness and cache provenance."""

    cached: bool = False
    as_of: datetime | None = None
    stale: bool = False

    @classmethod
    def live(cls, message: Message) -> "CachedMessage":
        """Wrap a freshly fetched message without labeling it as a cached snapshot."""
        return cls(**message.model_dump(), as_of=datetime.now(timezone.utc))


class HeaderObservation(BaseModel):
    """A parsed message header at one validated IMAP location."""

    location: MessageKey
    message: Message
    flags: list[str]
    size: int


class BodyJob(BaseModel):
    """A current location at which an uncached body can be read."""

    message_id: str
    location: MessageKey
    size: int


class ChunkJob(BaseModel):
    """One pending document chunk, whose ID is internal and never exposed as a mail ID."""

    id: int
    text: str


def _like(value: str) -> str:
    """Escape a literal substring for a bound SQLite LIKE parameter."""
    return "%" + value.casefold().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class MailIndex:
    """Store a single account's derived mail data separately from notification state.

    Args:
        path: Rebuildable database file on storage suitable for SQLite locking.
        account: Opaque account identity; opening another account's file is rejected.
        embedding_profile: Vector-space fingerprint, or empty when disabled.
        journal_mode: DELETE by default; WAL requires local/block-backed storage.

    Raises:
        ValueError: Unsafe database reuse or an unsupported journal mode.
        sqlite3.Error: Storage/extension failure; writes use atomic transactions.
    """

    def __init__(self, path: Path, account: str, embedding_profile: str = "", journal_mode: str = "DELETE"):
        """Open or initialize derived tables without touching notification checkpoints."""
        if journal_mode not in {"DELETE", "WAL"}:
            raise ValueError("Index journal mode must be DELETE or WAL")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.account, self.profile = account, embedding_profile
        self.runtime_error = None
        self.runtime_embedding_error = None
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        try:
            os.chmod(path, 0o600)
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE name IN ('checkpoints','outbox','mail_identity_locations')").fetchone():
                raise ValueError("The search index must not share the notification/identity database")
            self.db.enable_load_extension(True)
            try:
                sqlite_vec.load(self.db)
            finally:
                self.db.enable_load_extension(False)
            self.db.execute(f"PRAGMA journal_mode={journal_mode}")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS folders (
                    name TEXT PRIMARY KEY, validity INTEGER NOT NULL DEFAULT 0,
                    as_of REAL, error TEXT);
                CREATE TABLE IF NOT EXISTS mail (
                    id INTEGER PRIMARY KEY, public_id TEXT UNIQUE NOT NULL,
                    summary TEXT NOT NULL, header_hash TEXT NOT NULL,
                    subject TEXT NOT NULL, sender TEXT NOT NULL, recipients TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '', details TEXT,
                    body_state TEXT NOT NULL DEFAULT 'pending', body_retry REAL NOT NULL DEFAULT 0,
                    body_truncated INTEGER NOT NULL DEFAULT 0, semantic_truncated INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS locations (
                    folder TEXT NOT NULL REFERENCES folders(name) ON DELETE CASCADE,
                    validity INTEGER NOT NULL, uid INTEGER NOT NULL,
                    message INTEGER REFERENCES mail(id), received REAL, size INTEGER,
                    flags TEXT, unread INTEGER, flags_at REAL,
                    PRIMARY KEY(folder,validity,uid));
                CREATE INDEX IF NOT EXISTS locations_message ON locations(message);
                CREATE INDEX IF NOT EXISTS locations_date ON locations(received DESC);
                CREATE INDEX IF NOT EXISTS locations_flags ON locations(flags_at);
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, message INTEGER NOT NULL REFERENCES mail(id) ON DELETE CASCADE,
                    text TEXT NOT NULL, embedded INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS chunks_pending ON chunks(embedded,retry_at);
                CREATE INDEX IF NOT EXISTS chunks_message ON chunks(message);
                CREATE VIRTUAL TABLE IF NOT EXISTS mail_fts USING fts5(
                    subject,sender,recipients,body,content=mail,content_rowid=id);
                CREATE TRIGGER IF NOT EXISTS mail_insert AFTER INSERT ON mail BEGIN
                    INSERT INTO mail_fts(rowid,subject,sender,recipients,body)
                    VALUES(new.id,new.subject,new.sender,new.recipients,new.body);
                END;
                CREATE TRIGGER IF NOT EXISTS mail_delete AFTER DELETE ON mail BEGIN
                    INSERT INTO mail_fts(mail_fts,rowid,subject,sender,recipients,body)
                    VALUES('delete',old.id,old.subject,old.sender,old.recipients,old.body);
                END;
                CREATE TRIGGER IF NOT EXISTS mail_update AFTER UPDATE OF subject,sender,recipients,body ON mail BEGIN
                    INSERT INTO mail_fts(mail_fts,rowid,subject,sender,recipients,body)
                    VALUES('delete',old.id,old.subject,old.sender,old.recipients,old.body);
                    INSERT INTO mail_fts(rowid,subject,sender,recipients,body)
                    VALUES(new.id,new.subject,new.sender,new.recipients,new.body);
                END;
                CREATE TEMP TABLE IF NOT EXISTS observed_uids (uid INTEGER PRIMARY KEY);
            """)
            with self.db:
                stored = self._meta("account")
                if stored and stored != account:
                    raise ValueError("Search index belongs to another account")
                self._set_meta("account", account)
                if self._meta("embedding_profile") != embedding_profile:
                    self.db.execute("DROP TABLE IF EXISTS vectors")
                    self.db.execute("UPDATE chunks SET embedded=0,retry_at=0")
                    self._set_meta("embedding_profile", embedding_profile)
                    self._set_meta("dimension", "0")
                    self._set_meta("embedding_error", "")
        except BaseException:
            self.db.close()
            raise

    def _meta(self, key: str) -> str:
        """Read one internal metadata value while the caller holds the store lock."""
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else ""

    def _set_meta(self, key: str, value: str) -> None:
        """Update an internal metadata value within the caller's transaction."""
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    @contextmanager
    def transaction(self):
        """Serialize a storage transaction and roll back all its writes on failure."""
        with self.lock, self.db:
            yield

    def close(self) -> None:
        """Close after all readers and sync/embedding workers have stopped."""
        with self.lock:
            self.db.close()

    def folders(self, names: list[str]) -> None:
        """Record a successfully enumerated folder set; absent folders lose their locations."""
        with self.transaction():
            known = {row[0] for row in self.db.execute("SELECT name FROM folders")}
            self.db.executemany("INSERT OR IGNORE INTO folders(name) VALUES (?)", [(name,) for name in names])
            self.db.executemany("DELETE FROM folders WHERE name=?", [(name,) for name in known - set(names)])
            self._set_meta("enumerated", str(time.time()))
            self._set_meta("sync_error", "")
        self.runtime_error = None

    def error(self, code: str, folder: str | None = None) -> None:
        """Record a safe exception class/category without erasing the last valid snapshot."""
        self.runtime_error = code
        try:
            with self.transaction():
                if folder is None:
                    self._set_meta("sync_error", code)
                else:
                    self.db.execute("UPDATE folders SET error=? WHERE name=?", (code, folder))
        except sqlite3.Error:
            pass

    def known(self, folder: str, validity: int) -> set[int]:
        """Return UIDs whose headers were indexed in this exact mailbox epoch."""
        with self.lock:
            return {row[0] for row in self.db.execute(
                "SELECT uid FROM locations WHERE folder=? AND validity=? AND message IS NOT NULL", (folder, validity))}

    def locations(self, message_id: str) -> list[MessageKey]:
        """Return indexed candidate locations for a native ID; live reads must still validate them."""
        if decode_id(message_id).account != self.account:
            raise ValueError("Message ID belongs to another account")
        with self.lock:
            return [MessageKey(account=self.account, folder=row[0], validity=row[1], uid=row[2])
                    for row in self.db.execute("SELECT l.folder,l.validity,l.uid FROM locations l "
                        "JOIN mail m ON m.id=l.message WHERE m.public_id=? ORDER BY l.received DESC,l.folder", (message_id,))]

    def flag_jobs(self, folder: str, validity: int, limit: int) -> list[int]:
        """Return a bounded, oldest-observation-first flag refresh batch."""
        with self.lock:
            return [row[0] for row in self.db.execute(
                "SELECT uid FROM locations WHERE folder=? AND validity=? AND message IS NOT NULL "
                "AND coalesce(flags_at,0)<? ORDER BY coalesce(flags_at,0),uid LIMIT ?",
                (folder, validity, time.time() - 300, limit))]

    def _drop_chunks(self, message: int) -> None:
        """Delete a message's derived vectors/chunks within the caller's transaction."""
        if int(self._meta("dimension") or 0):
            self.db.execute("DELETE FROM vectors WHERE rowid IN (SELECT id FROM chunks WHERE message=?)", (message,))
        self.db.execute("DELETE FROM chunks WHERE message=?", (message,))

    def reconcile(self, folder: str, validity: int, uids: list[int],
                  headers: list[HeaderObservation], flags: dict[int, list[str]]) -> None:
        """Commit a verified UID inventory, header batch, and flag observations atomically.

        Args:
            folder: Successfully selected folder.
            validity: Epoch checked before and after all IMAP fetches.
            uids: Complete observed UID inventory, including headers not yet fetched.
            headers: Newly fetched typed headers at locations in this inventory.
            flags: Successfully fetched current flags, never guessed after errors.

        Removes stale locations/epochs, not other copies of a native message.
        """
        now = time.time()
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO folders(name) VALUES (?)", (folder,))
            self.db.execute("DELETE FROM observed_uids")
            self.db.executemany("INSERT OR IGNORE INTO observed_uids VALUES (?)", [(uid,) for uid in uids])
            self.db.execute("DELETE FROM locations WHERE folder=? AND (validity<>? OR uid NOT IN (SELECT uid FROM observed_uids))",
                            (folder, validity))
            self.db.executemany("INSERT OR IGNORE INTO locations(folder,validity,uid) VALUES (?,?,?)",
                                [(folder, validity, uid) for uid in uids])
            for observation in headers:
                key, message = observation.location, observation.message
                if (key.account != self.account or key.folder != folder or key.validity != validity
                        or decode_id(message.id).account != self.account):
                    raise ValueError("Header observation does not belong to the selected account/epoch")
                if not self.db.execute("SELECT 1 FROM observed_uids WHERE uid=?", (key.uid,)).fetchone():
                    continue
                payload = message.model_dump(mode="json")
                digest = hashlib.sha256(json.dumps({k: payload[k] for k in (
                    "subject", "from_", "to", "cc", "date_time")}, sort_keys=True).encode()).hexdigest()
                old = self.db.execute("SELECT id,header_hash FROM mail WHERE public_id=?", (message.id,)).fetchone()
                sender = f"{message.from_.name} {message.from_.address}".casefold()
                recipients = " ".join(f"{item.name} {item.address}" for item in [*(message.to or []), *(message.cc or [])]).casefold()
                self.db.execute("INSERT INTO mail(public_id,summary,header_hash,subject,sender,recipients) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(public_id) DO UPDATE SET summary=excluded.summary,header_hash=excluded.header_hash,"
                    "subject=excluded.subject,sender=excluded.sender,recipients=excluded.recipients",
                    (message.id, message.model_dump_json(), digest, message.subject.casefold(), sender, recipients))
                rowid = self.db.execute("SELECT id FROM mail WHERE public_id=?", (message.id,)).fetchone()[0]
                if old and old["header_hash"] != digest:
                    self._drop_chunks(rowid)
                    self.db.execute("UPDATE mail SET details=NULL,body='',body_state='pending',body_retry=0,"
                                    "body_truncated=0,semantic_truncated=0 WHERE id=?", (rowid,))
                receipt = message.received_at.timestamp() if message.received_at else None
                values = observation.flags
                self.db.execute("UPDATE locations SET message=?,received=?,size=?,flags=?,unread=?,flags_at=? "
                    "WHERE folder=? AND validity=? AND uid=?",
                    (rowid, receipt, observation.size, json.dumps(values), int("\\seen" not in {v.casefold() for v in values}), now,
                     folder, validity, key.uid))
            for uid, values in flags.items():
                self.db.execute("UPDATE locations SET flags=?,unread=?,flags_at=? WHERE folder=? AND validity=? AND uid=?",
                                (json.dumps(values), int("\\seen" not in {v.casefold() for v in values}), now, folder, validity, uid))
            self.db.execute("UPDATE folders SET validity=?,as_of=?,error=NULL WHERE name=?", (validity, now, folder))

    def prune(self) -> None:
        """Purge unreferenced messages only after all folder/header inventories are usable.

        Incomplete moves/renames must not discard bodies while their destination
        headers are still being indexed. Removed mail is excluded from reads
        immediately, even before this conservative garbage collection runs.
        """
        with self.transaction():
            if (self._meta("sync_error") or not self._meta("enumerated")
                    or self.db.execute("SELECT 1 FROM folders WHERE as_of IS NULL OR error IS NOT NULL").fetchone()
                    or self.db.execute("SELECT 1 FROM locations WHERE message IS NULL").fetchone()):
                return
            for row in self.db.execute("SELECT id FROM mail WHERE NOT EXISTS(SELECT 1 FROM locations WHERE message=mail.id)").fetchall():
                self._drop_chunks(row[0])
                self.db.execute("DELETE FROM mail WHERE id=?", (row[0],))

    def body_jobs(self, limit: int) -> list[BodyJob]:
        """Select pending bodies by real receipt date, never by UID allocation order."""
        with self.lock:
            rows = self.db.execute("SELECT m.public_id,l.folder,l.validity,l.uid,l.size FROM mail m JOIN locations l ON l.message=m.id "
                "WHERE m.body_state='pending' AND m.body_retry<=? GROUP BY m.id "
                "ORDER BY max(l.received) DESC,m.id LIMIT ?", (time.time(), limit)).fetchall()
            return [BodyJob(message_id=row["public_id"], size=row["size"], location=MessageKey(account=self.account,
                folder=row["folder"], validity=row["validity"], uid=row["uid"])) for row in rows]

    def body_error(self, message_id: str, *, unavailable: bool = False) -> None:
        """Back off a failed body, or mark an oversized body as explicitly unavailable."""
        with self.transaction():
            self.db.execute("UPDATE mail SET body_state=?,body_retry=? WHERE public_id=?",
                            ("unavailable" if unavailable else "pending", time.time() + 300, message_id))

    def save_body(self, message: Message, location: MessageKey) -> bool:
        """Cache parsed details only if the location still identifies this message.

        Returns False for a stale observation. Chunk/vector replacement and text
        indexing are atomic; the public ID and attachment URIs are unchanged.
        """
        with self.transaction():
            row = self.db.execute("SELECT m.id FROM mail m JOIN locations l ON l.message=m.id "
                "WHERE m.public_id=? AND l.folder=? AND l.validity=? AND l.uid=?",
                (message.id, location.folder, location.validity, location.uid)).fetchone()
            if row is None or location.account != self.account:
                return False
            body = (message.body or "")[:MAX_BODY_CHARS]
            truncated = len(message.body or "") > MAX_BODY_CHARS
            metadata = message.model_dump(mode="json", exclude={"body"})
            prefix = f"{message.subject}\n{message.from_.address}"[:400]
            passages = [prefix + "\n" + body[start:start + CHUNK_CHARS]
                        for start in range(0, len(body), CHUNK_CHARS)] or [prefix]
            passages = [part for part in passages if part.strip()]
            self._drop_chunks(row[0])
            self.db.execute("UPDATE mail SET body=?,details=?,body_state='ready',body_retry=0,body_truncated=?,"
                            "semantic_truncated=? WHERE id=?",
                            (body, json.dumps(metadata), int(truncated), int(truncated or len(passages) > MAX_CHUNKS), row[0]))
            self.db.executemany("INSERT INTO chunks(message,text) VALUES (?,?)", [(row[0], part) for part in passages[:MAX_CHUNKS]])
            return True

    def chunks(self, limit: int = 16) -> list[ChunkJob]:
        """Return a bounded embedding batch, prioritizing recently received mail."""
        with self.lock:
            return [ChunkJob(id=row[0], text=row[1]) for row in self.db.execute(
                "SELECT c.id,c.text FROM chunks c JOIN locations l ON l.message=c.message "
                "WHERE c.embedded=0 AND c.retry_at<=? GROUP BY c.id "
                "ORDER BY max(l.received) DESC,c.id LIMIT ?", (time.time(), limit))]

    def vectors(self, jobs: list[ChunkJob], vectors: list[list[float]]) -> None:
        """Store a validated embedding batch without mixing vector spaces or dimensions."""
        if not self.profile or len(jobs) != len(vectors) or not vectors:
            raise ValueError("Invalid embedding batch")
        dimension = len(vectors[0])
        if not 1 <= dimension <= 16384 or any(len(vector) != dimension or not all(math.isfinite(v) for v in vector)
                                             or not any(vector) for vector in vectors):
            raise ValueError("Invalid embedding dimensions")
        with self.transaction():
            known = int(self._meta("dimension") or 0)
            if known and known != dimension:
                raise ValueError("Embedding dimensions changed; change the configured embedding profile")
            if not known:
                self.db.execute(f"CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{dimension}] distance_metric=cosine)")
                self._set_meta("dimension", str(dimension))
            for job, vector in zip(jobs, vectors):
                current = self.db.execute("SELECT text FROM chunks WHERE id=?", (job.id,)).fetchone()
                if current is None or current[0] != job.text:
                    continue
                self.db.execute("DELETE FROM vectors WHERE rowid=?", (job.id,))
                self.db.execute("INSERT INTO vectors(rowid,embedding) VALUES (?,?)", (job.id, sqlite_vec.serialize_float32(vector)))
                self.db.execute("UPDATE chunks SET embedded=1,retry_at=0 WHERE id=?", (job.id,))
            self._set_meta("embedding_error", "")
        self.runtime_embedding_error = None

    def embedding_error(self, jobs: list[ChunkJob], code: str, delay: float) -> None:
        """Retain failed embedding work for a later bounded retry, without blocking keyword reads."""
        self.runtime_embedding_error = code
        try:
            with self.transaction():
                self.db.executemany("UPDATE chunks SET retry_at=? WHERE id=? AND embedded=0",
                                    [(time.time() + delay, job.id) for job in jobs])
                self._set_meta("embedding_error", code)
        except sqlite3.Error:
            pass

    def status(self, model: str = "", stale_seconds: float = 600) -> IndexStatus:
        """Return typed coverage/freshness without exposing mail content, paths, or credentials."""
        with self.lock:
            now = time.time()
            folders = [FolderStatus(folder=row["name"], messages=row["n"], headers_pending=row["pending"],
                flags_stale=row["old_flags"], as_of=datetime.fromtimestamp(row["as_of"], timezone.utc) if row["as_of"] else None,
                error=row["error"]) for row in self.db.execute(
                "SELECT f.*,count(l.uid) n,sum(CASE WHEN l.uid IS NOT NULL AND l.message IS NULL THEN 1 ELSE 0 END) pending,"
                "sum(CASE WHEN l.uid IS NOT NULL AND coalesce(l.flags_at,0)<? THEN 1 ELSE 0 END) old_flags "
                "FROM folders f LEFT JOIN locations l ON l.folder=f.name GROUP BY f.name ORDER BY f.name", (now - stale_seconds,))]
            counts = self.db.execute("SELECT count(*) n,sum(body_state='pending') pending,sum(body_state='unavailable') unavailable,"
                "sum(body_truncated) truncated,sum(semantic_truncated) semantic_truncated FROM mail "
                "WHERE EXISTS(SELECT 1 FROM locations WHERE message=mail.id)").fetchone()
            pending = self.db.execute("SELECT count(*) FROM chunks WHERE embedded=0 "
                "AND EXISTS(SELECT 1 FROM locations WHERE message=chunks.message)").fetchone()[0]
            enumerated = float(self._meta("enumerated") or 0)
            sync_error = self.runtime_error or self._meta("sync_error") or None
            stale = bool(not enumerated or now - enumerated > stale_seconds or sync_error or
                         any(f.as_of is None or f.error or now - f.as_of.timestamp() > stale_seconds for f in folders))
            return IndexStatus(available=bool(enumerated), stale=stale, indexed_messages=counts["n"],
                bodies_pending=counts["pending"] or 0, bodies_unavailable=counts["unavailable"] or 0,
                bodies_truncated=counts["truncated"] or 0, semantic_truncated=counts["semantic_truncated"] or 0,
                chunks_pending=pending if model else 0, embedding_model=model or None,
                embedding_error=self.runtime_embedding_error or self._meta("embedding_error") or None, sync_error=sync_error, folders=folders)

    def cached(self, message_id: str, stale_seconds: float = 600) -> CachedMessage | None:
        """Read cached details only while an observed location remains; unknown IDs return None."""
        key = decode_id(message_id)
        if key.account != self.account:
            raise ValueError("Message ID belongs to another account")
        with self.lock:
            row = self.db.execute("SELECT m.details,m.body,l.folder,l.received,f.as_of,f.error FROM mail m "
                "JOIN locations l ON l.message=m.id JOIN folders f ON f.name=l.folder "
                "WHERE m.public_id=? AND m.details IS NOT NULL AND m.body_truncated=0 "
                "ORDER BY l.received DESC,l.folder LIMIT 1", (message_id,)).fetchone()
            if row is None:
                return None
            value = json.loads(row["details"])
            value.update(body=row["body"], folder=row["folder"], cached=True,
                         received_at=datetime.fromtimestamp(row["received"], timezone.utc) if row["received"] is not None else None,
                         as_of=datetime.fromtimestamp(row["as_of"], timezone.utc) if row["as_of"] else None,
                         stale=bool(self.runtime_error or self._meta("sync_error") or row["error"] or not row["as_of"] or time.time() - row["as_of"] > stale_seconds))
            return CachedMessage.model_validate(value)

    def _eligible(self, query: MailQuery) -> tuple[str, list]:
        """Build parameterized scope filters; user input never becomes SQL syntax."""
        where, parameters = [], []
        for field, column in ((query.folder, "l.folder"),):
            if field is not None:
                where.append(f"{column}=?")
                parameters.append(field)
        for value, column in ((query.sender, "m.sender"), (query.recipient, "m.recipients"), (query.subject, "m.subject")):
            if value is not None:
                where.append(f"lower({column}) LIKE ? ESCAPE '\\'")
                parameters.append(_like(value))
        for value, operator in ((query.after, ">="), (query.before, "<")):
            if value:
                where.append(f"l.received{operator}?")
                parameters.append(value.timestamp())
        if query.unread is not None:
            where.append("l.unread=?")
            parameters.append(int(query.unread))
        clause = " AND ".join(where) or "1"
        return ("SELECT m.id,max(l.received) received FROM mail m JOIN locations l ON l.message=m.id "
                f"WHERE {clause} GROUP BY m.id"), parameters

    def search(self, query: MailQuery, *, vector: list[float] | None = None,
               mode: Literal["keyword", "hybrid", "semantic"] = "keyword", model: str = "",
               warnings: list[str] | None = None, min_score: float = 0.5) -> SearchPage:
        """Search a local snapshot and fuse lexical/vector ranks, deduplicating native mail IDs.

        Semantic filters are applied before KNN selection. Text queries use a
        bounded candidate window; empty queries page through all indexed headers.
        The caller supplies a query vector from the configured embedding space.
        """
        warnings = list(warnings or [])
        with self.lock:
            if query.folder is not None and not self.db.execute("SELECT 1 FROM folders WHERE name=?", (query.folder,)).fetchone():
                raise ValueError("Folder is not in the observed index scope; inspect index_status")
            scope, args = self._eligible(query)
            common = "WITH eligible AS (" + scope + ") "
            scores, semantic, excerpts = {}, {}, {}
            limited = False
            if not query.query.strip():
                rows = self.db.execute(common + "SELECT id FROM eligible ORDER BY received DESC,id LIMIT ? OFFSET ?",
                                       (*args, query.limit + 1, query.offset)).fetchall()
                ids = [row[0] for row in rows]
                next_offset = query.offset + query.limit if len(ids) > query.limit else None
                ids = ids[:query.limit]
            else:
                terms = re.findall(r"\w+", query.query, flags=re.UNICODE)
                literal = " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)
                if mode != "semantic" and literal:
                    rows = self.db.execute(common + "SELECT mail_fts.rowid,bm25(mail_fts,4,2,2,1) rank,"
                        "snippet(mail_fts,-1,'','',' ... ',32) excerpt FROM mail_fts JOIN eligible e ON e.id=mail_fts.rowid "
                        "WHERE mail_fts MATCH ? ORDER BY rank,e.received DESC,e.id LIMIT ?", (*args, literal, CANDIDATES)).fetchall()
                    limited = len(rows) == CANDIDATES
                    for rank, row in enumerate(rows, 1):
                        scores[row[0]] = 1 / (60 + rank)
                        excerpts[row[0]] = row["excerpt"][:500]
                if vector is not None and int(self._meta("dimension") or 0):
                    if len(vector) != int(self._meta("dimension")):
                        raise ValueError("Query vector dimensions do not match the index")
                    rows = self.db.execute(common + "SELECT v.rowid,v.distance,c.message,c.text FROM "
                        "(SELECT rowid,distance FROM vectors WHERE embedding MATCH ? AND k=? "
                        "AND rowid IN (SELECT c.id FROM chunks c JOIN eligible e ON e.id=c.message)) v "
                        "JOIN chunks c ON c.id=v.rowid ORDER BY v.distance",
                        (*args, sqlite_vec.serialize_float32(vector), CANDIDATES)).fetchall()
                    limited |= len(rows) == CANDIDATES
                    for row in rows:
                        identifier, similarity = row["message"], 1 - row["distance"]
                        if similarity < min_score or identifier in semantic:
                            continue
                        semantic[identifier] = similarity
                        scores[identifier] = scores.get(identifier, 0) + 1 / (60 + len(semantic))
                        excerpts.setdefault(identifier, row["text"][:500])
                ids = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))
                next_offset = query.offset + query.limit if len(ids) > query.offset + query.limit else None
                ids = ids[query.offset:query.offset + query.limit]
            hits = []
            for identifier in ids:
                summary = Message.model_validate_json(self.db.execute("SELECT summary FROM mail WHERE id=?", (identifier,)).fetchone()[0])
                locations = self.db.execute("SELECT folder,received,unread FROM locations WHERE message=? ORDER BY received DESC,folder", (identifier,)).fetchall()
                selected = next(row for row in locations
                    if (query.folder is None or row["folder"] == query.folder)
                    and (query.unread is None or row["unread"] == int(query.unread))
                    and (query.after is None or (row["received"] is not None and row["received"] >= query.after.timestamp()))
                    and (query.before is None or (row["received"] is not None and row["received"] < query.before.timestamp())))
                summary.folder = selected["folder"]
                summary.received_at = datetime.fromtimestamp(selected["received"], timezone.utc) if selected["received"] is not None else None
                hits.append(SearchHit(message=summary, folders=sorted({row["folder"] for row in locations}),
                                      snippet=excerpts.get(identifier), score=scores.get(identifier), semantic_score=semantic.get(identifier)))
            status = self.status(model)
            complete = status.available and not status.stale and not any(f.headers_pending for f in status.folders)
            if query.query.strip():
                complete &= not (status.bodies_pending or status.bodies_unavailable or status.bodies_truncated)
            if query.unread is not None:
                complete &= not any(f.flags_stale for f in status.folders)
            if query.mode != "keyword" and query.query.strip() and model:
                complete &= not (status.chunks_pending or status.embedding_error or status.semantic_truncated or vector is None)
            if not complete:
                warnings.append("Index coverage is incomplete or stale; missing results do not prove mail is absent")
            if limited:
                warnings.append("Ranked candidate window reached; narrow the query or filters for an exhaustive search")
            return SearchPage(results=hits, next_offset=next_offset, mode=mode, complete=bool(complete),
                              candidate_limit_reached=limited, warnings=warnings, index=status)
