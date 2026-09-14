# iCal MCP

Read-only calendar-feed tools ported from `aibs/ical`, with source-change events
sent to session-manager. Outlook is intentionally not ported; the existing aibs
Outlook implementation is being superseded by the NVIDIA-provided integration.

## Tools

- `list_calendars()` lists configured names, availability, `as_of`, last check,
  sanitized refresh errors, and `stale`. It does not expose private feed URLs.
- `range_events(start_date_time, end_date_time, calendar=null, limit=50)` returns
  `{events, truncated, calendars}`. Dates must include a timezone; the half-open
  interval is at most 366 days. Results overlap the interval, are start-ordered
  across calendars, and are capped at 100. Narrow the range when truncated.
- `get_event(event_id)` reads full source-event or occurrence details, including
  organizer, attendees, description, status, recurrence rule, and freshness.

Opaque IDs scope provider UIDs to their feed. Expanded repetitions have distinct
IDs including their occurrence start. A removed or rescheduled occurrence can
make a previous ID stale; query the new range. A source ID from a change event
addresses the underlying series or exception instead of expanding every repeat.
Bare UIDs from the old aibs tool must be reacquired using the new tools.

All-day events retain DATE values and exclusive end dates. Floating times use the
feed's `X-WR-TIMEZONE`, falling back to `ICAL_TIMEZONE`. Explicit event timezone
information is retained. `calendar_timezone` identifies the floating/all-day
interpretation. Timed zero-duration events are supported; malformed or mixed-type
intervals are rejected. Recurrence rules, EXDATE/RDATE, and moved/cancelled
exceptions use [recurring-ical-events](https://recurring-ical-events.readthedocs.io/en/v3.8.0/reference/api.html).
Queries expand only bounded day-sized windows, never an entire infinite series.
Result and feed-size limits are not a hard CPU budget for pathological rules.

## Updates And Reliability

An iCal feed URL is fetched periodically by the adapter; it is not a provider
webhook. Conditional GET uses stored ETag and Last-Modified validators. The
default refresh period is five minutes. Initial successful loading establishes a
quiet baseline rather than announcing historical entries.

`ICAL_NOTIFY_SINCE` additionally filters history that appears in later snapshots.
The default `startup` persists the first updated adapter activation time for each
feed; restarts do not move it forward. Existing state from older versions gets
its cutoff on the first startup with this policy. An ISO date (midnight UTC) or
timezone-aware timestamp sets an explicit floor; `all` disables the date filter.
The initial complete snapshot is quiet in every mode.

Notification eligibility is based on event intervals at/after the cutoff, not
only the original start of a recurring master. The recurrence engine checks for
a qualifying occurrence, so an ongoing anniversary beginning in 1982 is not
mistaken for a finished historical event. Expired series and past individual
exceptions are suppressed. Both previous and new source dates are considered:
cancelling or moving an upcoming event into the past still generates a change.
`notification_context` on each change provides the qualifying interval; this is
not necessarily the next occurrence relative to today's wall clock.

All source snapshots and historical reads remain intact. `list_calendars`
reports the effective `notify_since`. Suppression only affects notifications.

Source additions, updates, explicit cancellations, and removals produce passive
`calendar_changed` events at `/event`, in batches of at most 20 changes. Recurrence
expansion and the passage of time alone do not produce notifications. Metadata-only
DTSTAMP/LAST-MODIFIED/SEQUENCE churn is ignored when the event data is unchanged.
`removed_from_feed` means exactly that: rolling exports can omit old events without
cancelling them. Calendar descriptions remain external content, not instructions.

```json
{
  "source": "ical",
  "event_type": "calendar_changed",
  "energy": "passive",
  "text": "{calendar name and bounded source changes}",
  "metadata": {
    "event_id": "stable delivery UUID",
    "calendar": "Personal",
    "observed_at": "UTC timestamp"
  }
}
```

Feed bytes, source snapshots, validators, and pending events persist in SQLite.
Replacing a snapshot and queuing its changes share one transaction. Failed HTTP
downloads, invalid feeds, and storage failures do not advance the source snapshot.
Reads can use the previous valid snapshot but report `stale`; a feed that has
never loaded fails explicitly instead of returning an empty or partial calendar.
Freshness is observational, not proof that a provider export reflects its latest
underlying state. Concurrent reads across feeds are not one atomic snapshot.

Pending events are rechecked against the current cutoff before delivery. Legacy
batches are filtered using their saved dates and available current source data;
recurrences are re-evaluated instead of judging them solely by an old master date.
Excluded changes are removed from the notification outbox, not from the calendar.
Eligible pending events retry until successful HTTP acceptance. A lost response may cause
repeat delivery with the same `metadata.event_id`. Session-manager's HTTP 202
acknowledges scheduling, not durable processing; end-to-end exactly-once processing
and event-ID deduplication are not provided. Retain the state volume and run one
replica. Replacing a feed URL, even for token rotation, establishes a new feed
identity and quiet baseline; old pending events remain stored but are not delivered
under the new configuration. Removing the volume loses pending notifications.
Changing the cutoff to an earlier date does not replay already processed source
changes. The adapter cannot retract events already accepted by session-manager.

These are calendar change notifications, not appointment reminders. Use the time
service for scheduled reminders. This adapter does not create, update, cancel, or
RSVP to events; SMTP invitations remain separate explicit actions.

## Configuration

Run `python -m ical_mcp.server` for HTTP MCP at `/mcp`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ICAL_URLS` | required | JSON object mapping calendar names to private HTTPS feed URLs |
| `ICAL_TIMEZONE` | `UTC` | Floating/all-day fallback timezone |
| `ICAL_MCP_PORT` | `8008` | HTTP MCP port |
| `ICAL_REFRESH_SECONDS` | `300` | Adapter-side refresh period |
| `ICAL_NOTIFY_SINCE` | `startup` | Persisted first activation, explicit ISO date/aware timestamp, or `all` |
| `ICAL_TIMEOUT_SECONDS` | `20` | Overall refresh and HTTP timeout |
| `ICAL_MAX_FEED_BYTES` | `5242880` | Maximum decoded feed size |
| `ICAL_STATE_PATH` | `/data/ical.sqlite3` | Persistent state database |
| `ICAL_EVENT_ENERGY` | `passive` | Passive queues while busy; active preempts |
| `ICAL_SESSION_ID` | empty | Explicit session route, otherwise platform default |
| `SESSION_MANAGER_URL` | `http://localhost:5000` | Event endpoint base URL |

Only operator-configured HTTPS feeds are fetched. TLS validation remains enabled;
redirects are not followed, so configure the final feed URL. URL credentials in
the path/query are supported, but URL userinfo and fragments are rejected. URLs
are excluded from settings repr, tool status, and request logs. Cache volumes
contain private calendar content and require the same access protection as memory.
No embedded aibs example URLs or credentials are copied into this port.

The old `MAILCAL_MCP_*` variables and CLI/stdin entrypoints are not retained.
The chart is opt-in and reads `ICAL_URLS` from a referenced Kubernetes Secret.
