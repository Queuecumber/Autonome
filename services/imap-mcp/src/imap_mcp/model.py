"""Read-only IMAP access and MIME parsing, adapted from aibs/imap/model.py."""

import base64
import binascii
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
import hashlib
import heapq
import json
import os
import ssl
from urllib.parse import urlparse

from imapclient import IMAPClient
from markdownify import markdownify
from pydantic import BaseModel, Field

DATE_FETCH_BATCH_SIZE = 200


def notification_date(value: str) -> datetime | None:
    """Parse a notification floor; startup/all are modes, dates mean midnight UTC.

    Args:
        value: startup, all, an ISO date, or a timezone-aware ISO timestamp.

    Returns:
        A UTC timestamp for an explicit floor, or None for a mode.

    Raises:
        ValueError: Invalid value or a timestamp without a timezone.
    """
    if value in {"startup", "all"}:
        return None
    result = datetime.fromisoformat(value)
    if len(value) == 10:
        result = result.replace(tzinfo=timezone.utc)
    if result.utcoffset() is None:
        raise ValueError("IMAP_NOTIFY_SINCE requires a date or timezone-aware timestamp")
    return result.astimezone(timezone.utc)


def received_time(value) -> datetime | None:
    """Normalize an aware INTERNALDATE or saved ISO receipt time; unknown values return None."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if isinstance(value, datetime) and value.utcoffset() is not None:
        return value.astimezone(timezone.utc)
    return None


@dataclass(frozen=True)
class Settings:
    """One mail account; TLS is required except trusted plaintext relays. Passwords are excluded from repr.

    Args:
        server: imaps:// (or tls://) host, starttls:// host, or imap:// for a trusted plaintext relay.
        username: Login name.
        password: Login password or app password.
        folders: Case-sensitive folders to watch, not a restriction on searches.
        timeout: Socket timeout in seconds.
        max_message_bytes: Maximum full message size accepted by detail tools.
        notify_since: startup persists the first watcher time; all disables the
            receipt-date filter; an ISO date/timestamp sets an explicit floor.

    Raises:
        ValueError: If the account, URL, folders, or limits are invalid.
    """
    server: str
    username: str
    password: str = field(repr=False)
    folders: tuple[str, ...] = ("INBOX",)
    timeout: float = 20
    max_message_bytes: int = 25 * 1024 * 1024
    notify_since: str = "startup"

    def __post_init__(self):
        """Reject invalid configuration before opening a network connection."""
        url = urlparse(self.server)
        if (url.scheme not in {"imaps", "tls", "starttls", "imap"} or not url.hostname
                or url.username or url.password or url.path not in {"", "/"}
                or url.query or url.fragment):
            raise ValueError("IMAP_SERVER must be an imaps://, starttls://, or imap:// host URL")
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError("Invalid IMAP port")
        if not self.username or not self.password:
            raise ValueError("IMAP_USERNAME and IMAP_PASSWORD are required")
        if (not self.folders or len(set(self.folders)) != len(self.folders)
                or any(not folder or any(c in folder for c in "\r\n\x00") for folder in self.folders)):
            raise ValueError("IMAP_FOLDERS must contain unique, nonempty folder names")
        if self.timeout <= 0 or self.max_message_bytes <= 0:
            raise ValueError("IMAP limits must be positive")
        notification_date(self.notify_since)

    @property
    def account(self) -> str:
        """Return an opaque identity scoped to server, port, and login, not password."""
        url = urlparse(self.server)
        identity = [url.hostname, url.port or (143 if url.scheme in {"starttls", "imap"} else 993), self.username]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]

    @classmethod
    def from_env(cls) -> "Settings":
        """Read IMAP_* settings; raises ValueError for malformed configuration."""
        folders = json.loads(os.environ.get("IMAP_FOLDERS", '["INBOX"]'))
        if not isinstance(folders, list) or not all(isinstance(folder, str) for folder in folders):
            raise ValueError("IMAP_FOLDERS must be a JSON list of names")
        return cls(os.environ.get("IMAP_SERVER", ""), os.environ.get("IMAP_USERNAME", ""),
                   os.environ.get("IMAP_PASSWORD", ""), tuple(folders),
                   float(os.environ.get("IMAP_TIMEOUT_SECONDS", "20")),
                   int(os.environ.get("IMAP_MAX_MESSAGE_BYTES", str(25 * 1024 * 1024))),
                   os.environ.get("IMAP_NOTIFY_SINCE", "startup"))


class MessageKey(BaseModel):
    """An account, folder, UIDVALIDITY, and UID that identify exactly one mailbox item."""
    account: str
    folder: str = Field(min_length=1)
    validity: int = Field(gt=0)
    uid: int = Field(gt=0)

    def encode(self) -> str:
        """Return a URL-safe opaque message ID for tools and resource URIs."""
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")

    @classmethod
    def decode(cls, value: str) -> "MessageKey":
        """Parse a complete message ID; raises ValueError for malformed IDs."""
        if not value or len(value) > 4096:
            raise ValueError("Invalid message ID")
        try:
            return cls.model_validate_json(base64.b64decode(
                value + "=" * (-len(value) % 4), altchars=b"-_", validate=True))
        except (ValueError, binascii.Error) as error:
            raise ValueError("Invalid message ID") from error


class EmailAddress(BaseModel):
    """A parsed display name and address; missing headers produce empty strings."""
    name: str
    address: str


class AttachmentMetadata(BaseModel):
    """Attachment index, display filename, MIME type, decoded byte count, and resource URI."""
    id: str
    name: str
    content_type: str
    size: int
    uri: str


class Message(BaseModel):
    """Mail details or a header-only summary; None means a field was not fetched."""
    id: str
    subject: str
    from_: EmailAddress
    folder: str = ""
    date_time: datetime | None = None
    received_at: datetime | None = None
    to: list[EmailAddress] | None = None
    cc: list[EmailAddress] | None = None
    body: str | None = None
    has_attachments: bool | None = None
    attachment_metadata: list[AttachmentMetadata] | None = None


def attachment_bytes(part: EmailMessage) -> bytes:
    """Decode an attachment, including an attached RFC822 message, into bytes."""
    payload = part.get_payload(decode=True)
    if payload is not None:
        return payload
    nested = part.get_payload()
    if isinstance(nested, list):
        return b"\r\n".join(item.as_bytes() for item in nested)
    return str(nested or "").encode()


def parse_message(raw: bytes, key: MessageKey, summary: bool = False,
                  received_at: datetime | None = None) -> Message:
    """Parse MIME without changing mailbox state.

    Args:
        raw: RFC822 message or headers.
        key: Stable mailbox identity to attach to results.
        summary: Omit body, recipients, and attachment claims for header-only data.
        received_at: Server INTERNALDATE, distinct from the sender's Date header.

    Returns:
        A message; malformed or absent dates are represented as None.
    """
    mail = BytesParser(policy=policy.default).parsebytes(raw)
    sent = None
    try:
        sent = parsedate_to_datetime(str(mail.get("Date", "")))
    except (ValueError, TypeError, OverflowError):
        pass
    name, address = parseaddr(str(mail.get("From", "")))
    result = Message(id=key.encode(), subject=str(mail.get("Subject", "")),
                     from_=EmailAddress(name=name, address=address), folder=key.folder,
                     date_time=sent, received_at=received_time(received_at))
    if summary:
        return result
    for header in ["to", "cc"]:
        setattr(result, header, [EmailAddress(name=name, address=address)
            for name, address in getaddresses([str(value) for value in mail.get_all(header, [])])])
    body = mail.get_body(preferencelist=("plain", "html"))
    result.body = ""
    if body is not None:
        content = body.get_content()
        result.body = markdownify(content) if body.get_content_type() == "text/html" else content
    result.attachment_metadata = [AttachmentMetadata(
        id=str(index), name=part.get_filename() or f"attachment-{index}",
        content_type=part.get_content_type(), size=len(attachment_bytes(part)),
        uri=f"imap://attachments/{key.encode()}/{index}",
    ) for index, part in enumerate(mail.iter_attachments())]
    result.has_attachments = bool(result.attachment_metadata)
    return result


class Mailbox:
    """Read-only tools with a separate connection per call, isolated from IDLE workers.

    Args:
        settings: Validated account settings.
    """
    def __init__(self, settings: Settings):
        """Retain settings without connecting to the mail server."""
        self.settings = settings

    @contextmanager
    def connect(self):
        """Yield an authenticated connection; TLS unless imap://. Errors propagate."""
        url = urlparse(self.settings.server)
        direct_tls = url.scheme in {"imaps", "tls"}
        with IMAPClient(url.hostname, port=url.port or (993 if direct_tls else 143),
                        ssl=direct_tls,
                        ssl_context=None if url.scheme == "imap" else ssl.create_default_context(),
                        timeout=self.settings.timeout, use_uid=True) as client:
            if url.scheme == "starttls":
                client.starttls(ssl_context=ssl.create_default_context())
            client.login(self.settings.username, self.settings.password)
            client.normalise_times = False
            yield client

    def received_at(self, message_id: str) -> datetime | None:
        """Read a message's authoritative receipt time for legacy queued notifications.

        Args:
            message_id: Complete account-scoped ID.

        Returns:
            UTC INTERNALDATE, or None if removed, reset, or missing a usable date.

        Raises:
            ValueError: Invalid/foreign ID.
            Exception: Connection or protocol failure; callers must retain queued events.
        """
        key = MessageKey.decode(message_id)
        if key.account != self.settings.account:
            raise ValueError("Message ID belongs to another account")
        with self.connect() as client:
            selected = client.select_folder(key.folder, readonly=True)
            if int(selected[b"UIDVALIDITY"]) != key.validity:
                return None
            record = client.fetch([key.uid], ["INTERNALDATE"]).get(key.uid, {})
            return received_time(record.get(b"INTERNALDATE"))

    def folders(self) -> list[str]:
        """Return selectable folder names with case and hierarchy preserved."""
        with self.connect() as client:
            return [name for flags, _, name in client.list_folders()
                    if b"\\Noselect" not in flags]

    def raw_message(self, message_id: str) -> tuple[bytes, MessageKey, datetime | None]:
        """Fetch full mail using BODY.PEEK; reject stale IDs or oversized messages.

        Args:
            message_id: ID from a search or incoming-mail event.

        Returns:
            Message bytes, validated key, and server receipt time (None if unavailable).

        Raises:
            ValueError: For an invalid/foreign ID, UIDVALIDITY change, or size limit.
            KeyError: If the mail has been removed.
        """
        key = MessageKey.decode(message_id)
        if key.account != self.settings.account:
            raise ValueError("Message ID belongs to another account")
        with self.connect() as client:
            selected = client.select_folder(key.folder, readonly=True)
            if int(selected[b"UIDVALIDITY"]) != key.validity:
                raise ValueError("Mailbox UIDVALIDITY changed; search again for a current ID")
            size = client.fetch([key.uid], ["RFC822.SIZE"]).get(key.uid)
            if size is None:
                raise KeyError("Message no longer exists")
            if size[b"RFC822.SIZE"] > self.settings.max_message_bytes:
                raise ValueError("Message exceeds IMAP_MAX_MESSAGE_BYTES")
            record = client.fetch([key.uid], ["BODY.PEEK[]", "INTERNALDATE"]).get(key.uid, {})
            data = record.get(b"BODY[]")
            if data is None:
                raise KeyError("Message no longer exists")
            if len(data) > self.settings.max_message_bytes:
                raise ValueError("Message exceeds IMAP_MAX_MESSAGE_BYTES")
            return data, key, received_time(record.get(b"INTERNALDATE"))

    def get(self, message_id: str) -> Message:
        """Return full details for an ID; propagates missing, stale, and oversized-mail errors."""
        raw, key, received = self.raw_message(message_id)
        return parse_message(raw, key, received_at=received)

    def attachment(self, message_id: str, attachment_id: str) -> tuple[bytes, AttachmentMetadata]:
        """Return attachment bytes and metadata; raises KeyError for an absent attachment."""
        raw, key, _ = self.raw_message(message_id)
        mail = BytesParser(policy=policy.default).parsebytes(raw)
        for index, part in enumerate(mail.iter_attachments()):
            if str(index) == attachment_id:
                data = attachment_bytes(part)
                return data, AttachmentMetadata(id=attachment_id, name=part.get_filename() or f"attachment-{index}",
                    content_type=part.get_content_type(), size=len(data),
                    uri=f"imap://attachments/{key.encode()}/{index}")
        raise KeyError("Attachment not found")

    def search(self, search: str, folder: str | None = None, limit: int = 10) -> list[Message]:
        """Search all selectable folders by default, ranking by server receipt date.

        Args:
            search: IMAP search criteria, not a Graph/KQL query.
            folder: Exact folder name, or None (default) for all selectable folders.
            limit: Total cap from 1 to 100, applied after ranking across all folders.

        Returns:
            Header summaries ordered by descending INTERNALDATE. Missing/invalid
            receipt dates sort last, with folder and UID as deterministic tie-breaks.
            date_time remains the sender's Date header; received_at is the sort date.
            Copies in different folders remain separate. Concurrent expunges may
            shorten the result. Use get for bodies and attachment details.

        Raises:
            ValueError: For invalid limits or command control characters.
            RuntimeError: A selected folder changes UIDVALIDITY during the search.
            Exception: Protocol failures propagate instead of returning partial results.
        """
        if not 1 <= limit <= 100 or not search.strip() or any(c in search for c in "\r\n\x00"):
            raise ValueError("Use nonempty IMAP criteria and a limit from 1 to 100")
        candidates = []
        minimum = datetime.min.replace(tzinfo=timezone.utc)
        with self.connect() as client:
            folders = [folder] if folder is not None else [name for flags, _, name in client.list_folders()
                                                          if b"\\Noselect" not in flags]
            for name in folders:
                selected = client.select_folder(name, readonly=True)
                validity = int(selected[b"UIDVALIDITY"])
                uids = client.search(search)
                for offset in range(0, len(uids), DATE_FETCH_BATCH_SIZE):
                    batch = uids[offset:offset + DATE_FETCH_BATCH_SIZE]
                    records = client.fetch(batch, ["INTERNALDATE"])
                    for uid in batch:
                        if uid not in records:
                            continue
                        received = received_time(records[uid].get(b"INTERNALDATE"))
                        candidate = (received is not None, received or minimum, name, uid, validity)
                        if len(candidates) < limit:
                            heapq.heappush(candidates, candidate)
                        else:
                            heapq.heappushpop(candidates, candidate)
            ranked = sorted(candidates, reverse=True)
            grouped = {}
            for candidate in ranked:
                grouped.setdefault(candidate[2], []).append(candidate)
            results = {}
            for name, entries in grouped.items():
                selected = client.select_folder(name, readonly=True)
                if int(selected[b"UIDVALIDITY"]) != entries[0][4]:
                    raise RuntimeError("Mailbox changed during search; retry for current message IDs")
                records = client.fetch([entry[3] for entry in entries], ["BODY.PEEK[HEADER]"])
                for known_date, received, _, uid, validity in entries:
                    raw = records.get(uid, {}).get(b"BODY[HEADER]")
                    if raw is None:
                        continue
                    key = MessageKey(account=self.settings.account, folder=name, validity=validity, uid=uid)
                    results[name, uid] = parse_message(raw, key, summary=True,
                                                       received_at=received if known_date else None)
        return [results[name, uid] for _, _, name, uid, _ in ranked if (name, uid) in results]
