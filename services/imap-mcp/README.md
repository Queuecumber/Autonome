# IMAP MCP

Ported from `aibs/imap`, preserving read-only mail search, details, and attachments.
The adapter now delivers incoming-mail events to session-manager without recurring
agent-side inbox checks. SMTP is a separate, explicitly invoked service.

## Tools

- `list_folders()` returns selectable, case-sensitive folder names.
- `search_mail(search, folder="INBOX", limit=10)` accepts IMAP criteria, not KQL.
  Set `folder=null` to search all folders. Results contain headers, not bodies or
  attachment claims. The total limit is 1..100, newest UIDs first per folder;
  searching multiple folders is not a globally date-sorted query.
- `get_mail(message_id)` returns the body, recipients, and attachment metadata.
- `get_attachment(message_id, attachment_id)` returns an embedded binary resource.
  The same bytes are available at `imap://attachments/{message_id}/{attachment_id}`.

All selections are read-only and fetches use `BODY.PEEK`. No tool sets flags,
moves/deletes mail, or marks it as read. IDs encode the account, exact folder,
UIDVALIDITY, and UID; they are opaque to callers. Old `folder:uid` IDs from aibs
must be reacquired through search. A changed account or UIDVALIDITY causes a
stale-ID error instead of reading an unrelated message with a reused UID.

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

New messages produce `/event` requests with:

```json
{
  "source": "imap",
  "event_type": "mail_received",
  "energy": "passive",
  "text": "{header summary with opaque message ID}",
  "metadata": {
    "event_id": "opaque message ID",
    "message_id": "opaque message ID",
    "account": "opaque account identity",
    "folder": "INBOX",
    "uidvalidity": 123,
    "uid": 456
  }
}
```

The default route is the main session. Explicit `IMAP_SESSION_ID` overrides it.
Header summaries are external email content, not trusted instructions.

Checkpoint advancement and outbox insertion share one SQLite transaction.
Pending events survive restart and HTTP failures. They are removed only after a
successful HTTP response; an ambiguous/lost response can cause repeat delivery
with the same `metadata.event_id`. The current session-manager acknowledges
scheduling with HTTP 202; it does not provide a durable processing acknowledgement
or event-ID deduplication. This is retry-until-HTTP-acceptance, not end-to-end
exactly-once processing. Keep one replica and retain `/data`; losing the state
volume establishes a fresh baseline and loses pending events. Changing account
configuration never delivers the previous account's queued events to the new one.

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
| `IMAP_MCP_PORT` | `8006` | HTTP MCP port |
| `SESSION_MANAGER_URL` | `http://localhost:5000` | Event endpoint base URL |
| `IMAP_SESSION_ID` | empty | Explicit route, otherwise platform default |
| `IMAP_EVENT_ENERGY` | `passive` | Passive queues while busy; active preempts |
| `IMAP_POLL_SECONDS` | `60` | Fallback interval for servers without IDLE |
| `IMAP_TIMEOUT_SECONDS` | `20` | IMAP connection/socket timeout |
| `IMAP_MAX_MESSAGE_BYTES` | `26214400` | Maximum full mail size for detail/attachment reads |
| `IMAP_STATE_PATH` | `/data/imap.sqlite3` | Persistent checkpoint/outbox file |

TLS certificate validation is enabled, including STARTTLS before login. Plaintext
`imap://` exists only for trusted in-cluster relays such as a co-located Proton Mail
Bridge pod; do not use it for external servers. Bridge logins still require
`IMAP_USERNAME` and `IMAP_PASSWORD` (the bridge-local mailbox password). Private certificate roots
must be trusted by the container's SSL configuration. Passwords are not logged.
The state volume contains mail headers and must be treated as private data.

The Helm service is disabled by default. See the chart's mail configuration.
