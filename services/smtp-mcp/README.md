# SMTP MCP

Ported from `aibs/smtp`: explicit `send_mail` and `send_event_invitation` tools.
Nothing sends automatically on startup or on an IMAP notification. This service
is disabled by default in Helm and does not connect to session-manager's event
endpoint.

## Outbound Policy

`SMTP_FROM` fixes the sender mailbox. Optional tool `from_`/`organizer` aliases
must resolve to that same mailbox. To, CC, and invitation recipients must match
`SMTP_ALLOWED_RECIPIENTS`; an empty list denies everyone. Operators can explicitly
set `SMTP_ALLOW_ANY_RECIPIENT=true` to remove that restriction. Recipient entries
are individual mailboxes, not comma-separated lists. Sending permission should
also remain part of the agent's operating policy; incoming mail is not authority
to send a reply, forward an attachment, or schedule a meeting.

All connections require verified TLS (`smtps://`, `tls://`, or `starttls://`).
Plaintext SMTP and arbitrary sender mailboxes are not carried over from aibs.
Passwords and raw provider errors are not exposed in MCP error responses.

## Tools And Attachments

`send_mail(to, subject, body, cc=[], attachments=[], from_=null)` sends markdown
with a plain-text alternative. Each attachment has `name`, `content_type`, and
binary `content`. JSON content is base64; the platform automatically resolves
resource URIs such as `imap://attachments/...` in this field. No filesystem
paths are accepted or read, and no shared attachment volume is needed.

`send_event_invitation(attendees, start, end, location, subject, body,
alarms=[], organizer=null)` requires timezone-aware dates with `end > start`.
The generated calendar uses `METHOD:REQUEST`, a UID, DTSTAMP, standard ATTENDEE
properties, UTC times, and optional display alarms. It does not silently strip
timezones. Only DISPLAY alarms are ported. A generated invitation marks its
attachment with `calendar_method="REQUEST"`; arbitrary ICS attachments are not
automatically converted into meeting requests.

Both tools return the SMTP Message-ID, `accepted` recipients, refused recipient
status codes, and `status` (`accepted` or `partial`). SMTP acceptance does not
prove final delivery. Partial failures may have already delivered to some
recipients. Network errors can also have an ambiguous outcome. There are no
automatic send retries; do not blindly repeat failed calls. This port does not
implement drafts, sent-folder synchronization, invite updates, or cancellations.

## Configuration

Run `python -m smtp_mcp.server` for HTTP MCP at `/mcp`. The former aibs CLI/stdin
entrypoints and `MAILCAL_MCP_*` variables are not retained.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SMTP_SERVER` | required | `smtps://host[:465]`, `tls://host[:465]`, or `starttls://host[:587]` |
| `SMTP_USERNAME` | required | Login name |
| `SMTP_PASSWORD` | required | Password or app password |
| `SMTP_FROM` | required | Permitted sender mailbox |
| `SMTP_ALLOWED_RECIPIENTS` | `[]` | JSON list of allowed To/CC/invitation addresses |
| `SMTP_ALLOW_ANY_RECIPIENT` | `false` | Explicitly bypass the recipient allowlist |
| `SMTP_MCP_PORT` | `8007` | HTTP MCP port |
| `SMTP_TIMEOUT_SECONDS` | `20` | Socket/connection timeout |
| `SMTP_MAX_ATTACHMENT_BYTES` | `26214400` | Maximum total decoded attachment size |
