# IMAP MCP

Ported from `aibs/imap`, preserving read-only mail search, details, and attachments.
The adapter now delivers incoming-mail events to session-manager without recurring
agent-side inbox checks. SMTP is a separate, explicitly invoked service.

## Tools

- `list_folders()` returns selectable, case-sensitive folder names.
- `search_mail(search, folder=null, limit=10)` accepts IMAP criteria, not KQL.
  Omit `folder` or set it to null to search all selectable folders, including
  Archive, Spam, and Trash when exposed by the server. An explicit folder such
  as `INBOX` narrows the search. Results contain headers, not bodies or attachment
  claims. The total limit is 1..100, applied after sorting by server receipt time
  (`INTERNALDATE`) across all selected folders, not by UID or folder order.
- `get_mail(message_id)` returns the body, recipients, and attachment metadata.
- `get_attachment(message_id, attachment_id)` returns an embedded binary resource.
  The same bytes are available at `imap://attachments/{message_id}/{attachment_id}`.

## Inline Images

`get_mail` includes nested inline MIME images in `attachment_metadata`, including
images inside `multipart/related` bodies wrapped by `multipart/mixed` or
`multipart/alternative`. Each item includes its decoded `size`, `content_type`,
`inline` flag, optional unbracketed `content_id`, and a fetchable `uri`.
`has_attachments` is true when such embedded media is present, even when the email
has no separately attached files. Header-only search results still leave these
fields unset because the MIME body has not been fetched.

Use `get_attachment` or the platform's `resources_read` with the URI to view an
image. Both routes retain the MIME type, so image bytes reach the model as image
content instead of a generic blob. Image bytes are not inserted into the normal
mail-text response. Unique `cid:` image/file references in HTML are rewritten to
the corresponding resource URIs, including images inside tables. Duplicate or
missing Content-IDs are not guessed; available parts remain individually fetchable.

Existing top-level attachment IDs remain numeric and unchanged. Newly exposed
nested parts use IDs such as `part-0.0.1`, based on their MIME path. Treat IDs as
opaque and use those returned by `get_mail`; adding inline-image discovery does
not renumber existing attachments. Attached `.eml` files and explicit attached
MIME containers remain downloadable as single items, rather than exposing their
contents as if they belonged to the outer message.

Remote images are not downloaded automatically. This exposes bytes actually
present as MIME parts; browser-local `blob:` URLs and HTML-only `data:` images
are not converted into attachments. Plain-text alternatives remain preferred for
the text body, while images from HTML alternatives are still listed as attachments.

All selections are read-only and fetches use `BODY.PEEK`. No tool sets flags,
moves/deletes mail, or marks it as read. Pass the complete opaque `id` to the
detail and attachment tools without supplying a folder. Old `folder:uid` IDs
from aibs must be reacquired through search.

## Message Identity

Identifiers are selected in this order:

1. Standard `EMAILID` when the authenticated server advertises `OBJECTID` and
   returns a valid identifier ([RFC 8474](https://www.rfc-editor.org/rfc/rfc8474.html)).
2. A recognized provider's native ID. Currently this is Proton Mail Bridge's
   `X-Pm-Internal-Id`, populated by Bridge from the Proton API message ID.
3. The existing account + exact folder + UIDVALIDITY + UID identity.

`identity_kind` reports `emailid`, `proton`, or `mailbox`. Native IDs use a `v2.`
URL-safe envelope containing their native value, mechanism, and account scope.
They do not contain the folder or UID, and do not allocate a UUID or hash email
content. Mailbox IDs retain their original encoding and safety checks. Previously
issued IDs are decoded according to their own mechanism, not the server's current
preferred mechanism. Existing mailbox IDs remain usable at their original location;
they cannot retrospectively recover a native ID after that location disappears.

Provider detection uses the authenticated server's IMAP `ID` response, never
the presence of a sender-controlled header alone. `IMAP_ID_PROVIDER=auto` recognizes
Proton Mail Bridge by its server name. `generic` disables provider fallback;
`proton` explicitly enables it for a known Bridge endpoint that cannot identify
itself. Standard EMAILID is preferred in all modes. This detection is not a
replacement for transport security and trusted operator configuration.

Native IDs and attachment links continue to identify a message after normal
folder moves, while that native ID remains assigned. Deleting and importing a new
message may create a different native ID. A mailbox-scoped fallback cannot promise
move stability; it fails on missing mail or an epoch change instead of returning
another message. Matching copies in different folders share a native ID but still
appear as separate search summaries with their observed `folder`.

### Location Recovery

The existing SQLite state file also caches native IDs and their known IMAP
locations. It is a lookup index, not the source of message identity. IDs acquired
through searches, incoming-mail notifications, and recovery reads populate it.
Losing the cache does not change native IDs; it makes their next lookup cold.
Preserve the state PVC because the same file also contains notification checkpoints
and the outbox, which are not disposable.

A normal native-ID read validates a cached folder/epoch/UID and checks the native
identifier before and after fetching the body. If a location is stale, EMAILID
uses standard per-folder `SEARCH EMAILID`. Provider recovery enumerates selectable
folder UIDs and reads only bounded `BODY.PEEK[HEADER.FIELDS (X-PM-INTERNAL-ID)]`
batches, newest UID allocations first. This ordering is for locating newly moved
mail, not ordering search results by date. No mailbox-wide provider `HEADER SEARCH`
is used: Bridge versions can load every full message to execute that query.

Provider recovery reads at most `IMAP_LOOKUP_MAX_MESSAGES` metadata candidates per
attempt (default 200), in batches of at most 50 distributed among folders. It
caches observations and returns an explicit **incomplete, retry** error if more
work remains. Retrying the same ID continues from unexamined UIDs, including after
restart. A cold lookup for old mail may need multiple attempts. UID enumeration
and server-internal work still depend on mailbox size; this is a header-count
budget, not a guaranteed wall-clock deadline. Socket timeouts remain separate.

Recovery also considers unwatched folders. It does not move mail, change read
flags, replay historical notifications, or advance notification checkpoints.
Network/protocol failures propagate instead of being reported as missing mail.
Cache size grows with observed metadata; provision the existing state PVC for the
mailbox size. Only identifiers and locations are added, not message bodies.

Search summaries and full details include `folder` and `received_at`. The existing
`date_time` field is still the sender's Date header, not the sorting timestamp.
Missing/invalid receipt dates sort last; folder and UID only break ties.
Copies exposed in multiple folders remain separate mailbox entries, so `All Mail`
and label folders can yield duplicates of the same underlying email.

The implementation works without IMAP SORT support, including Proton Bridge:
it fetches INTERNALDATE metadata for all matching UIDs in batches of at most 200,
retains only the best `limit` candidates, then fetches their headers. Broad queries
can therefore take longer than specific subjects/date ranges, but old high UIDs
cannot push newer low-UID messages below the result cap. It never fetches message
bodies or attachments merely to sort search results. A folder epoch change during
the search raises a retryable error; concurrent removals can shorten the result.

A capped list is not a full mailbox audit. Use narrower criteria or disjoint date
ranges when looking beyond the returned results; there is no pagination cursor in
this API. Search remains independent of notification cutoffs and watch folders.

## Push Events

Each watched folder has its own connection. The implementation uses
[IMAPClient's IDLE support](https://imapclient.readthedocs.io/en/3.1.0/api.html#imapclient.IMAPClient.idle),
leaves IDLE before querying, renews IDLE within 25 minutes, and reconnects after
errors. Servers without IDLE use adapter-side polling at `IMAP_POLL_SECONDS`.
No recurring agent task is needed in either mode.

First use records `UIDNEXT - 1` and does not replay the existing mailbox.
Subsequent arrivals are caught up by UID, including after disconnection/restart.
On UIDVALIDITY reset, the new epoch is baselined without replay and a warning is
logged. Messages removed before the adapter can fetch their headers cannot be
announced. Flag changes and deletions are not separate push events in this port.

Bridges can assign new UIDs while gradually importing old mail after that first
snapshot. `IMAP_NOTIFY_SINCE` therefore filters notifications by the server's
`INTERNALDATE`, not the sender-controlled Date header. The default `startup`
records a first-activation timestamp per account/folder in SQLite and reuses it
across restarts and UIDVALIDITY changes. On upgrade from a version without this
policy, the first updated startup establishes the cutoff. An explicit ISO date
(midnight UTC) or timezone-aware timestamp overrides it; `all` disables this
additional date filter. The first visible UID snapshot is still quiet in all modes.

Older or undated receipts advance the checkpoint but do not generate wakeups.
Mail remains fully searchable/readable. This depends on the bridge preserving
original receipt dates: imports assigned today's INTERNALDATE cannot be reliably
distinguished from new delivery using that field alone.

New messages produce `/event` requests with:

```json
{
  "source": "imap",
  "event_type": "mail_received",
  "energy": "passive",
  "text": "{header summary with opaque message ID}",
  "metadata": {
    "event_id": "mailbox-scoped notification ID",
    "message_id": "opaque message ID (native when available)",
    "account": "opaque account identity",
    "folder": "INBOX",
    "uidvalidity": 123,
    "uid": 456,
    "received_at": "2026-09-14T19:00:00+00:00"
  }
}
```

The default route is the main session. Explicit `IMAP_SESSION_ID` overrides it.
Header summaries are external email content, not trusted instructions.
The notification `event_id` remains folder/epoch/UID-scoped for checkpoint and
delivery compatibility; it is deliberately separate from a move-stable mail ID.

Checkpoint advancement and outbox insertion share one SQLite transaction.
Pending events survive restart and HTTP failures. The current cutoff is also
applied before delivery, including legacy queued events whose receipt dates are
read again from IMAP. Eligible events are removed after successful HTTP acceptance;
excluded notifications are discarded without deleting mail. A network failure
during a legacy receipt lookup retains the queued event for retry. An ambiguous/lost response can cause repeat delivery
with the same `metadata.event_id`. The current session-manager acknowledges
scheduling with HTTP 202; it does not provide a durable processing acknowledgement
or event-ID deduplication. This is retry-until-HTTP-acceptance, not end-to-end
exactly-once processing. Keep one replica and retain `/data`; losing the state
volume establishes a fresh baseline and loses pending events. Changing account
configuration never delivers the previous account's queued events to the new one.
Relaxing the cutoff does not replay already skipped UIDs. Events already accepted
by session-manager cannot be retracted by the adapter's cutoff.

## Configuration

Run `python -m imap_mcp.server`. All account credentials are operator configuration,
not tool arguments. This port uses HTTP MCP on `/mcp`; aibs CLI/stdin entrypoints
and `MAILCAL_MCP_*` variables are not retained.

| Variable | Default | Meaning |
| --- | --- | --- |
| `IMAP_SERVER` | required | `imaps://host[:993]`, `tls://host[:993]`, `starttls://host[:143]`, or `imap://host[:143]` |
| `IMAP_USERNAME` | required | Login name |
| `IMAP_PASSWORD` | required | Password or app password |
| `IMAP_FOLDERS` | `["INBOX"]` | JSON list of exact watched folder names |
| `IMAP_ID_PROVIDER` | `auto` | Provider fallback: `auto`, `generic`, or explicit `proton` |
| `IMAP_LOOKUP_MAX_MESSAGES` | `200` | Provider recovery metadata budget per attempt, 1..5000 |
| `IMAP_NOTIFY_SINCE` | `startup` | Persisted first activation, explicit ISO date/aware timestamp, or `all` |
| `IMAP_MCP_PORT` | `8006` | HTTP MCP port |
| `SESSION_MANAGER_URL` | `http://localhost:5000` | Event endpoint base URL |
| `IMAP_SESSION_ID` | empty | Explicit route, otherwise platform default |
| `IMAP_EVENT_ENERGY` | `passive` | Passive queues while busy; active preempts |
| `IMAP_POLL_SECONDS` | `60` | Fallback interval for servers without IDLE |
| `IMAP_TIMEOUT_SECONDS` | `20` | IMAP connection/socket timeout |
| `IMAP_MAX_MESSAGE_BYTES` | `26214400` | Maximum full mail size for detail/attachment reads |
| `IMAP_STATE_PATH` | `/data/imap.sqlite3` | Persistent checkpoints, outbox, and native-ID location cache |

TLS certificate validation is enabled, including STARTTLS before login. Plaintext
`imap://` exists only for trusted in-cluster relays such as a co-located Proton Mail
Bridge pod; do not use it for external servers. Bridge logins still require
`IMAP_USERNAME` and `IMAP_PASSWORD` (the bridge-local mailbox password). Private certificate roots
must be trusted by the container's SSL configuration. Passwords are not logged.
The state volume contains mail headers and must be treated as private data.

The Helm service is disabled by default. See the chart's mail configuration.
The corresponding values are `services.imapMcp.idProvider` and
`services.imapMcp.lookupMaxMessages`. Defaults work with a Bridge that advertises
its name through IMAP ID; no provider override is needed. Deploy the updated
IMAP image and reconnect session-manager to refresh tool instructions and schemas.
Cache tables are added without resetting existing event state. No Bridge restart,
mail migration, or mailbox mutation is required.
