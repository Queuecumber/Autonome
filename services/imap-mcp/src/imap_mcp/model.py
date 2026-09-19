"""Read-only IMAP access and MIME parsing, adapted from aibs/imap/model.py."""

from contextlib import contextmanager
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
import hashlib
import heapq
import json
import logging
import os
from pathlib import Path
import ssl
from typing import Literal
from urllib.parse import unquote, urlparse

from imapclient import IMAPClient
from imapclient.exceptions import IMAPClientError
from markdownify import MarkdownConverter
from pydantic import BaseModel

from imap_mcp.identity import IdentityStore, LookupIncompleteError, MessageKey, NativeKey, decode_id

DATE_FETCH_BATCH_SIZE = 200
IDENTITY_FETCH_BATCH_SIZE = 50
PROTON_HEADER = "X-Pm-Internal-Id"
PROTON_FETCH = "BODY.PEEK[HEADER.FIELDS (X-PM-INTERNAL-ID)]"
logger = logging.getLogger(__name__)
RawMessage = tuple[bytes, MessageKey, datetime | None]


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
        id_provider: auto detects known servers via IMAP ID; generic disables
            provider-specific IDs; proton explicitly trusts Proton Bridge headers.
            Standard OBJECTID support is preferred in every mode.
        lookup_max_messages: Maximum header records read during native-ID recovery.

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
    id_provider: Literal["auto", "generic", "proton"] = "auto"
    lookup_max_messages: int = 200

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
        if self.id_provider not in {"auto", "generic", "proton"}:
            raise ValueError("IMAP_ID_PROVIDER must be auto, generic, or proton")
        if type(self.lookup_max_messages) is not int or not 1 <= self.lookup_max_messages <= 5000:
            raise ValueError("IMAP_LOOKUP_MAX_MESSAGES must be between 1 and 5000")
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
                   os.environ.get("IMAP_NOTIFY_SINCE", "startup"),
                   os.environ.get("IMAP_ID_PROVIDER", "auto"),
                   int(os.environ.get("IMAP_LOOKUP_MAX_MESSAGES", "200")))


@dataclass(frozen=True)
class IdentityCapabilities:
    """Native identifier fields supported by this authenticated server connection."""
    emailid: bool = False
    proton: bool = False

    @property
    def fetch_fields(self) -> list[str]:
        """Return only supported native fields; provider IDs are supplied by fetched headers."""
        return ["EMAILID"] if self.emailid else []


def native_value(record: dict, kind: Literal["emailid", "proton"]) -> str | None:
    """Extract and validate one native identifier from an IMAP fetch response.

    Args:
        record: A metadata or full-message FETCH record.
        kind: Field to inspect; selection of trusted providers happens separately.

    Returns:
        A case-sensitive native ID, or None for absent, duplicated, or malformed data.
    """
    if kind == "emailid":
        value = record.get(b"EMAILID")
        if isinstance(value, (tuple, list)) and len(value) == 1:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")
        elif type(value) is int:
            value = str(value)
    else:
        raw = record.get(b"BODY[]", record.get(b"BODY[HEADER]"))
        if raw is None:
            raw = next((value for name, value in record.items()
                        if isinstance(name, bytes) and name.upper().startswith(b"BODY[HEADER.FIELDS")), b"")
        headers = BytesParser(policy=policy.default).parsebytes(raw, headersonly=True)
        values = headers.get_all(PROTON_HEADER, [])
        value = str(values[0]).strip() if len(values) == 1 else None
    if not isinstance(value, str):
        return None
    try:
        return NativeKey(account="validation", kind=kind, value=value).value
    except ValueError:
        return None


class EmailAddress(BaseModel):
    """A parsed display name and address; missing headers produce empty strings."""
    name: str
    address: str


class AttachmentMetadata(BaseModel):
    """A fetchable MIME part, including inline media and its optional Content-ID."""
    id: str
    name: str
    content_type: str
    size: int
    uri: str
    inline: bool = False
    content_id: str | None = None


class Message(BaseModel):
    """Mail details or a header-only summary; None means a field was not fetched."""
    id: str
    identity_kind: Literal["emailid", "proton", "mailbox"] = "mailbox"
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


def _content_id(part: EmailMessage) -> str | None:
    """Return one unbracketed Content-ID; absent, blank, or duplicate headers return None."""
    values = part.get_all("Content-ID", [])
    if len(values) != 1:
        return None
    value = str(values[0]).strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    return value or None


def attachment_parts(mail: EmailMessage) -> list[tuple[str, EmailMessage]]:
    """Discover attachments and nested inline MIME media without entering attached emails.

    Args:
        mail: Parsed RFC822 message.

    Returns:
        Attachment IDs and MIME parts. Existing top-level attachments retain
        numeric IDs. Additional parts use deterministic part-0.0-style MIME
        paths, so discovering inline images never renumbers old attachments.
        Body alternatives are not presented as attachments. Explicit attached
        MIME containers and message/rfc822 parts remain single downloadable items.
    """
    result = [(str(index), part) for index, part in enumerate(mail.iter_attachments())]
    known = {id(part) for _, part in result}
    pending = [("0", mail)]
    while pending:
        path, part = pending.pop()
        attached = part.is_attachment() or part.get_content_type() == "message/rfc822"
        media = part.get_content_maintype() == "image" or (
            part.get_content_maintype() not in {"text", "multipart"}
            and (part.get_content_disposition() == "inline" or _content_id(part) is not None))
        if (attached or media) and id(part) not in known:
            result.append((f"part-{path}", part))
            known.add(id(part))
        if not attached and part.get_content_maintype() == "multipart":
            children = list(part.iter_parts())
            pending.extend((f"{path}.{index}", child)
                           for index, child in reversed(list(enumerate(children))))
    return result


def attachment_content(part: EmailMessage, attachment_id: str,
                       message_id: str) -> tuple[bytes, AttachmentMetadata]:
    """Decode a MIME part into bytes and its public attachment metadata.

    Args:
        part: The attachment or inline MIME media to decode.
        attachment_id: ID assigned by attachment_parts.
        message_id: Complete account-scoped public message ID.

    Returns:
        Decoded bytes and typed metadata with the part's media type and resource URI.
        A filename is descriptive only; it is never used as a local path.
    """
    data = attachment_bytes(part)
    cid = _content_id(part)
    inline = not part.is_attachment() and (
        part.get_content_disposition() == "inline" or cid is not None
        or part.get_content_maintype() == "image")
    return data, AttachmentMetadata(
        id=attachment_id, name=part.get_filename() or f"attachment-{attachment_id}",
        content_type=part.get_content_type(), size=len(data),
        uri=f"imap://attachments/{message_id}/{attachment_id}", inline=inline, content_id=cid,
    )


class MailMarkdown(MarkdownConverter):
    """Render mail HTML with unique Content-ID references linked to MCP attachments.

    Args:
        attachments: Fetchable MIME parts belonging to the message.

    Duplicate Content-IDs are not guessed. Remote and browser-local URLs are not
    fetched, and no HTML scripts or remote resources are executed.
    """

    def __init__(self, attachments: list[AttachmentMetadata]):
        """Build an unambiguous CID lookup without embedding any attachment bytes in text."""
        super().__init__()
        counts = Counter(item.content_id for item in attachments if item.content_id)
        self.cid_uris = {item.content_id: item.uri for item in attachments
                         if item.content_id and counts[item.content_id] == 1}

    def _link(self, element, attribute: str) -> bool:
        """Replace a uniquely resolved cid attribute with its resource URI; return whether changed."""
        value = element.get(attribute, "")
        if isinstance(value, str) and value[:4].lower() == "cid:":
            cid = unquote(value[4:]).strip("<>")
            if uri := self.cid_uris.get(cid):
                element[attribute] = uri
                return True
        return False

    def convert_img(self, el, text, parent_tags):
        """Render an image reference, retaining resolved embedded images even inside table cells."""
        if self._link(el, "src"):
            parent_tags = parent_tags - {"_inline"}
        return super().convert_img(el, text, parent_tags)

    def convert_a(self, el, text, parent_tags):
        """Render links to CID-backed files using the same attachment resource mapping."""
        self._link(el, "href")
        return super().convert_a(el, text, parent_tags)


def parse_message(raw: bytes, key: MessageKey, summary: bool = False,
                  received_at: datetime | None = None, identity: NativeKey | None = None) -> Message:
    """Parse MIME without changing mailbox state.

    Args:
        raw: RFC822 message or headers.
        key: Stable mailbox identity to attach to results.
        summary: Omit body, recipients, and attachment claims for header-only data.
        received_at: Server INTERNALDATE, distinct from the sender's Date header.
        identity: Validated account-scoped native ID, or None for a mailbox-scoped ID.

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
    message_id = (identity or key).encode()
    result = Message(id=message_id, identity_kind=identity.kind if identity else "mailbox",
                     subject=str(mail.get("Subject", "")),
                     from_=EmailAddress(name=name, address=address), folder=key.folder,
                     date_time=sent, received_at=received_time(received_at))
    if summary:
        return result
    for header in ["to", "cc"]:
        setattr(result, header, [EmailAddress(name=name, address=address)
            for name, address in getaddresses([str(value) for value in mail.get_all(header, [])])])
    result.attachment_metadata = [attachment_content(part, identifier, message_id)[1]
                                  for identifier, part in attachment_parts(mail)]
    result.has_attachments = bool(result.attachment_metadata)
    body = mail.get_body(preferencelist=("plain", "html"))
    result.body = ""
    if body is not None:
        content = body.get_content()
        result.body = MailMarkdown(result.attachment_metadata).convert(content) if body.get_content_type() == "text/html" else content
    return result


class Mailbox:
    """Read-only tools with a separate connection per call, isolated from IDLE workers.

    Args:
        settings: Validated account settings.
        state_path: Persistent adapter SQLite file, or None for a temporary location cache.
    """
    def __init__(self, settings: Settings, state_path: Path | None = None):
        """Retain settings and open a location cache; None uses memory only, without network access."""
        self.settings = settings
        self.identities = IdentityStore(state_path)

    def close(self) -> None:
        """Close the owned location cache after all tools and workers have stopped."""
        self.identities.close()

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

    def identity_capabilities(self, client) -> IdentityCapabilities:
        """Negotiate standard IDs and recognize providers from authenticated server metadata.

        Args:
            client: Authenticated IMAP connection, not currently in IDLE.

        Returns:
            Supported native identity fields. Unknown or unsupported ID responses
            do not enable provider headers; a configured proton override does.

        Raises:
            Exception: Network and capability-query failures propagate.
        """
        emailid = bool(client.has_capability("OBJECTID"))
        proton = self.settings.id_provider == "proton"
        if self.settings.id_provider == "auto" and client.has_capability("ID"):
            try:
                response = client.id_()
            except IMAPClientError:
                logger.warning("IMAP ID unavailable; provider-specific identifiers disabled")
            else:
                if isinstance(response, dict):
                    fields = response.items()
                elif (isinstance(response, (tuple, list)) and len(response) == 1
                      and isinstance(response[0], (tuple, list)) and len(response[0]) % 2 == 0):
                    fields = zip(response[0][::2], response[0][1::2])
                else:
                    fields = []
                for key, value in fields:
                    key = key.decode(errors="replace") if isinstance(key, bytes) else key
                    value = value.decode(errors="replace") if isinstance(value, bytes) else value
                    if key == "name" and isinstance(value, str):
                        proton = value.casefold() in {"proton mail bridge", "protonmail bridge", "protonmail-bridge"}
        return IdentityCapabilities(emailid=emailid, proton=proton)

    def observe(self, location: MessageKey, record: dict,
                capabilities: IdentityCapabilities) -> NativeKey | None:
        """Cache fetched native fields and return the highest-priority available identity.

        Args:
            location: The mailbox epoch and UID associated with the fetch.
            record: A header or full-message FETCH record, including requested EMAILID.
            capabilities: Known supported identifier mechanisms for this connection.

        Returns:
            EMAILID, otherwise a recognized Proton ID, otherwise None. Both native
            forms are cached when available so previously issued IDs keep working.
        """
        preferred = None
        for kind, enabled in [("emailid", capabilities.emailid), ("proton", capabilities.proton)]:
            if not enabled:
                continue
            value = native_value(record, kind)
            self.identities.remember(location, kind, value)
            if value is not None and preferred is None:
                preferred = NativeKey(account=location.account, kind=kind, value=value)
        return preferred

    def summarize(self, location: MessageKey, record: dict,
                  capabilities: IdentityCapabilities) -> Message:
        """Return a typed header summary and remember its native identity/location mapping.

        Args:
            location: Selected mailbox epoch and fetched UID.
            record: Header FETCH record with receipt time and supported native fields.
            capabilities: Identifier fields requested for this connection.

        Returns:
            A header-only Message whose ID is independent of location when supported.
        """
        return parse_message(record[b"BODY[HEADER]"], location, summary=True,
                             received_at=received_time(record.get(b"INTERNALDATE")),
                             identity=self.observe(location, record, capabilities))

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
        """Fetch mail read-only, resolving native IDs across folders when necessary.

        Args:
            message_id: ID from a search or incoming-mail event.

        Returns:
            Message bytes, validated key, and server receipt time (None if unavailable).

        Raises:
            ValueError: For an invalid/foreign ID, UIDVALIDITY change, or size limit.
            KeyError: If the mail has been removed.
            LookupIncompleteError: Recovery reached its metadata budget; retry
                continues cached progress, without claiming the mail is absent.
        """
        key = decode_id(message_id)
        if key.account != self.settings.account:
            raise ValueError("Message ID belongs to another account")
        with self.connect() as client:
            if isinstance(key, NativeKey):
                return self._native_message(client, key)
            selected = client.select_folder(key.folder, readonly=True)
            if int(selected[b"UIDVALIDITY"]) != key.validity:
                raise ValueError("Mailbox UIDVALIDITY changed; search again for a current ID")
            return self._read_selected(client, key)

    def _read_selected(self, client, location: MessageKey,
                       identity: NativeKey | None = None) -> tuple[bytes, MessageKey, datetime | None]:
        """Read a selected UID with size checks and optional native-identity verification.

        Args:
            client: Connection with location.folder selected read-only.
            location: Confirmed epoch and UID.
            identity: Expected native identifier, checked before and after body fetch.

        Returns:
            Full message bytes, location, and normalized server receipt date.

        Raises:
            KeyError: The message disappeared or no longer matches the native ID.
            ValueError: The message exceeds the configured size limit.
            RuntimeError: The server omitted required size metadata.
        """
        native_fields = [] if identity is None else (["EMAILID"] if identity.kind == "emailid" else [PROTON_FETCH])
        size = client.fetch([location.uid], ["RFC822.SIZE", *native_fields]).get(location.uid)
        if size is None or (identity is not None and native_value(size, identity.kind) != identity.value):
            raise KeyError("Message no longer matches this location")
        if b"RFC822.SIZE" not in size:
            raise RuntimeError("Server omitted message size")
        if size[b"RFC822.SIZE"] > self.settings.max_message_bytes:
            raise ValueError("Message exceeds IMAP_MAX_MESSAGE_BYTES")
        extra = ["EMAILID"] if identity is not None and identity.kind == "emailid" else []
        record = client.fetch([location.uid], ["BODY.PEEK[]", "INTERNALDATE", *extra]).get(location.uid, {})
        data = record.get(b"BODY[]")
        if data is None:
            raise KeyError("Message no longer exists")
        if len(data) > self.settings.max_message_bytes:
            raise ValueError("Message exceeds IMAP_MAX_MESSAGE_BYTES")
        if identity is not None and native_value(record, identity.kind) != identity.value:
            raise KeyError("Message changed during retrieval")
        return data, location, received_time(record.get(b"INTERNALDATE"))

    def _try_location(self, client, location: MessageKey, identity: NativeKey) -> RawMessage | None:
        """Try a cached location; stale epochs/identities return None and protocol failures propagate."""
        selected = client.select_folder(location.folder, readonly=True)
        validity = int(selected[b"UIDVALIDITY"])
        self.identities.prune_epoch(location.account, location.folder, validity)
        if validity != location.validity:
            return None
        try:
            return self._read_selected(client, location, identity)
        except KeyError:
            self.identities.forget(location, identity.kind)
            return None

    def _native_message(self, client, identity: NativeKey) -> RawMessage:
        """Resolve a native ID using verified cached locations, then bounded protocol recovery.

        Args:
            client: Authenticated read-only tool connection.
            identity: Already account-validated native message identity.

        Returns:
            Message bytes, its current location, and receipt date.

        Raises:
            ValueError: The native mechanism is unavailable or a matching message is too large.
            KeyError: All available locations were checked and no matching message remains.
            LookupIncompleteError: More metadata needs to be examined on another attempt.
            Exception: Protocol failures propagate without being treated as absence.
        """
        capabilities = self.identity_capabilities(client)
        if ((identity.kind == "emailid" and not capabilities.emailid)
                or (identity.kind == "proton" and not capabilities.proton)):
            raise ValueError("This server does not support the ID's native identity mechanism")
        folders = [name for flags, _, name in client.list_folders() if b"\\Noselect" not in flags]
        for location in self.identities.locations(identity):
            if location.folder not in folders:
                self.identities.forget(location, identity.kind)
                continue
            if result := self._try_location(client, location, identity):
                return result
        if identity.kind == "emailid":
            for folder in folders:
                selected = client.select_folder(folder, readonly=True)
                validity = int(selected[b"UIDVALIDITY"])
                for uid in client.search(["EMAILID", identity.value]):
                    location = MessageKey(account=identity.account, folder=folder, validity=validity, uid=uid)
                    if result := self._try_location(client, location, identity):
                        self.identities.remember(location, identity.kind, identity.value)
                        return result
        else:
            return self._recover_provider_id(client, identity, folders, capabilities)
        raise KeyError("Message not found in selectable folders")

    def _recover_provider_id(self, client, identity: NativeKey, folders: list[str],
                             capabilities: IdentityCapabilities) -> RawMessage:
        """Scan bounded native-header batches, caching progress without whole-mailbox HEADER SEARCH.

        Args:
            client: Authenticated connection used only for read-only operations.
            identity: Account-scoped provider ID to recover.
            folders: Selectable folders to examine, including unwatched folders.
            capabilities: Supported native fields to cache alongside the provider ID.

        Returns:
            Full message bytes, a verified location, and receipt date on success.

        Raises:
            LookupIncompleteError: The header budget is exhausted or a mailbox changes epochs.
            KeyError: No match remains after all listed UIDs were examined.
            Exception: Connection or protocol errors propagate; progress remains cached.
        """
        remaining = self.settings.lookup_max_messages
        batch_size = max(1, min(IDENTITY_FETCH_BATCH_SIZE, remaining // max(1, len(folders))))
        pending = []
        for folder in folders:
            selected = client.select_folder(folder, readonly=True)
            validity = int(selected[b"UIDVALIDITY"])
            self.identities.prune_epoch(identity.account, folder, validity)
            uids = self.identities.unseen(identity.account, folder, validity, identity.kind,
                                          client.search(["ALL"]))
            if uids:
                pending.append((folder, validity, uids))
        while pending and remaining:
            next_round = []
            for folder, validity, uids in pending:
                if remaining == 0:
                    next_round.append((folder, validity, uids))
                    continue
                selected = client.select_folder(folder, readonly=True)
                if int(selected[b"UIDVALIDITY"]) != validity:
                    raise LookupIncompleteError("Mailbox changed during ID recovery; retry get_mail")
                batch = uids[:min(batch_size, remaining)]
                records = client.fetch(batch, [PROTON_FETCH, *capabilities.fetch_fields])
                remaining -= len(batch)
                for uid in batch:
                    record = records.get(uid)
                    if record is None:
                        continue
                    location = MessageKey(account=identity.account, folder=folder, validity=validity, uid=uid)
                    self.observe(location, record, capabilities)
                    if native_value(record, identity.kind) == identity.value:
                        if result := self._try_location(client, location, identity):
                            return result
                if len(uids) > len(batch):
                    next_round.append((folder, validity, uids[len(batch):]))
            pending = next_round
        if pending:
            raise LookupIncompleteError(
                "Message location recovery is incomplete, not absent; retry get_mail to continue cached progress")
        raise KeyError("Message not found in selectable folders")

    def get(self, message_id: str) -> Message:
        """Return full details for an ID; propagates missing, stale, and oversized-mail errors."""
        raw, key, received = self.raw_message(message_id)
        identity = decode_id(message_id)
        return parse_message(raw, key, received_at=received,
                             identity=identity if isinstance(identity, NativeKey) else None)

    def attachment(self, message_id: str, attachment_id: str) -> tuple[bytes, AttachmentMetadata]:
        """Return attachment bytes and metadata; raises KeyError for an absent attachment."""
        raw, key, _ = self.raw_message(message_id)
        mail = BytesParser(policy=policy.default).parsebytes(raw)
        for identifier, part in attachment_parts(mail):
            if identifier == attachment_id:
                return attachment_content(part, identifier, decode_id(message_id).encode())
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
            capabilities = self.identity_capabilities(client)
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
                self.identities.prune_epoch(self.settings.account, name, entries[0][4])
                records = client.fetch([entry[3] for entry in entries], ["BODY.PEEK[HEADER]", *capabilities.fetch_fields])
                for known_date, received, _, uid, validity in entries:
                    raw = records.get(uid, {}).get(b"BODY[HEADER]")
                    if raw is None:
                        continue
                    key = MessageKey(account=self.settings.account, folder=name, validity=validity, uid=uid)
                    record = {**records[uid], b"INTERNALDATE": received if known_date else None}
                    results[name, uid] = self.summarize(key, record, capabilities)
        return [results[name, uid] for _, _, name, uid, _ in ranked if (name, uid) in results]
