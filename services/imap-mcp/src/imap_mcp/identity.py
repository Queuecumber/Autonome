"""Native mail identities and a durable cache of their current IMAP locations."""

import base64
import binascii
from pathlib import Path
import re
import sqlite3
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IdentityKind = Literal["emailid", "proton"]


def _decode(value: str) -> bytes:
    """Decode a bounded opaque ID; malformed encodings raise ValueError."""
    if not value or len(value) > 4096:
        raise ValueError("Invalid message ID")
    try:
        return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Invalid message ID") from error


class MessageKey(BaseModel):
    """Account, folder, UIDVALIDITY, and UID identifying one native mailbox item."""

    account: str
    folder: str = Field(min_length=1)
    validity: int = Field(gt=0)
    uid: int = Field(gt=0)

    def encode(self) -> str:
        """Return the existing URL-safe mailbox ID, unchanged for older callers."""
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")

    @classmethod
    def decode(cls, value: str) -> "MessageKey":
        """Read a legacy mailbox ID; malformed or incomplete IDs raise ValueError."""
        try:
            return cls.model_validate_json(_decode(value))
        except ValueError as error:
            raise ValueError("Invalid message ID") from error


class NativeKey(BaseModel):
    """An account-scoped server identifier, independent of folder and mailbox UID.

    kind distinguishes standard EMAILID from a recognized provider's identifier.
    value is the native identifier, not a content hash or locally allocated UUID.
    """

    model_config = ConfigDict(extra="forbid")
    account: str = Field(min_length=1, max_length=128)
    kind: IdentityKind
    value: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_native_value(self) -> "NativeKey":
        """Enforce safe native-ID alphabets; malformed IDs raise ValueError."""
        pattern = r"[A-Za-z0-9_-]{1,255}" if self.kind == "emailid" else r"[A-Za-z0-9_+/=-]{1,1024}"
        if re.fullmatch(pattern, self.value) is None:
            raise ValueError("Invalid native message ID")
        return self

    def encode(self) -> str:
        """Return a versioned URL-safe ID retaining its native value and account scope."""
        payload = base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")
        return "v2." + payload

    @classmethod
    def decode(cls, value: str) -> "NativeKey":
        """Read a v2 native ID; malformed, unversioned, or unknown-kind IDs raise ValueError."""
        if not value.startswith("v2."):
            raise ValueError("Invalid native message ID")
        try:
            return cls.model_validate_json(_decode(value[3:]))
        except ValueError as error:
            raise ValueError("Invalid native message ID") from error


def decode_id(value: str) -> MessageKey | NativeKey:
    """Decode a public mail ID without reinterpreting old IDs when server capabilities change."""
    return NativeKey.decode(value) if value.startswith("v2.") else MessageKey.decode(value)


class LookupIncompleteError(RuntimeError):
    """A bounded recovery has not proved absence; another attempt can continue cached progress."""


class IdentityStore:
    """Thread-safe location observations, not an allocator of message identities.

    Args:
        path: Existing adapter SQLite file, or None for an isolated in-memory cache.

    Raises:
        sqlite3.Error: If the cache cannot be opened or updated.
    """

    def __init__(self, path: Path | None = None):
        """Initialize only identity tables; existing checkpoints and outbox data are untouched."""
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path) if path is not None else ":memory:", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS mail_identity_locations (
                account TEXT NOT NULL, folder TEXT NOT NULL, validity INTEGER NOT NULL,
                uid INTEGER NOT NULL, kind TEXT NOT NULL, value TEXT,
                PRIMARY KEY (account, folder, validity, uid, kind));
            CREATE INDEX IF NOT EXISTS mail_identity_lookup
                ON mail_identity_locations (account, kind, value);
        """)

    def remember(self, location: MessageKey, kind: IdentityKind, value: str | None) -> None:
        """Record a native ID or its absence at a location; later observations replace stale values.

        Args:
            location: Verified mailbox epoch and UID.
            kind: Identifier field that was fetched.
            value: Validated native value, or None when that field is absent.

        Returns:
            None after the observation is committed.
        """
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO mail_identity_locations VALUES (?, ?, ?, ?, ?, ?)",
                            (location.account, location.folder, location.validity, location.uid, kind, value))

    def locations(self, identity: NativeKey) -> list[MessageKey]:
        """Return known locations for an account-scoped ID; callers must validate before use."""
        with self.lock:
            rows = self.db.execute(
                "SELECT folder, validity, uid FROM mail_identity_locations "
                "WHERE account=? AND kind=? AND value=? ORDER BY rowid DESC",
                (identity.account, identity.kind, identity.value)).fetchall()
        return [MessageKey(account=identity.account, folder=folder, validity=validity, uid=uid)
                for folder, validity, uid in rows]

    def unseen(self, account: str, folder: str, validity: int, kind: IdentityKind,
               uids: list[int]) -> list[int]:
        """Return unobserved UIDs newest-allocation first, without using UID as a message date."""
        with self.lock:
            seen = {row[0] for row in self.db.execute(
                "SELECT uid FROM mail_identity_locations WHERE account=? AND folder=? AND validity=? AND kind=?",
                (account, folder, validity, kind))}
        return sorted(set(uids) - seen, reverse=True)

    def forget(self, location: MessageKey, kind: IdentityKind) -> None:
        """Remove a missing or mismatched cached location; unrelated copies remain usable."""
        with self.lock, self.db:
            self.db.execute(
                "DELETE FROM mail_identity_locations WHERE account=? AND folder=? AND validity=? AND uid=? AND kind=?",
                (location.account, location.folder, location.validity, location.uid, kind))

    def prune_epoch(self, account: str, folder: str, validity: int) -> None:
        """Discard observations from older epochs for this account and folder only."""
        with self.lock, self.db:
            self.db.execute("DELETE FROM mail_identity_locations WHERE account=? AND folder=? AND validity<>?",
                            (account, folder, validity))

    def close(self) -> None:
        """Close the cache after tools and monitoring workers have stopped."""
        with self.lock:
            self.db.close()
