# Local Mail Index

The IMAP MCP maintains a read-only, rebuildable local copy of searchable message
content. IMAP remains authoritative. The model returns typed query results and
status objects; the MCP server only exposes those operations and resources.

## Search

```json
{"query": "preschool calendar", "sender": "school@example.test", "mode": "hybrid", "limit": 10}
```

`search_mail` accepts literal keyword/semantic text plus optional exact `folder`,
literal substring `sender`, `recipient` (To/Cc), and `subject` filters. `after` is
an inclusive receipt timestamp and `before` is exclusive; both require timezones.
`unread` uses observed IMAP flags. These are not raw IMAP search expressions.
Use `search_server` explicitly when native IMAP criteria or a live query is needed.

- `keyword`: local FTS5 term search; words are ANDed, not interpreted as FTS or SQL
  operators. Results use BM25 ranking with higher weights for subject/addresses.
- `semantic`: embed the query and find similar stored message chunks. Disabled or
  failed query encoding is an explicit error, never an empty successful search.
- `hybrid`: combine keyword and semantic rankings using reciprocal-rank fusion.
  If embeddings are disabled/unavailable, return keyword results with a warning.
- Empty queries page through headers by descending server `INTERNALDATE`, not UID.

Folder/sender/recipient/date/unread filters apply before vector candidate selection.
The vector backend is pinned `sqlite-vec` 0.1.9 with exact cosine KNN, not an ANN
service. Query cost grows with the number of eligible vectors; no scale-independent
latency promise is made. Native-ID copies across folders appear once per message,
with observed folder membership listed on each hit. Generic folder/epoch/UID IDs
cannot safely be deduplicated across folders and remain separate.

Responses contain header summaries and bounded excerpts, not full mail bodies.
`next_offset` can be passed to the next call. Pagination reflects the current index,
not a frozen search session; concurrent synchronization can change ordering.
Text searches consider at most 1000 lexical candidates and 1000 vector chunks.
`candidate_limit_reached` explicitly reports this boundary; narrow the scope rather
than treating the last page as a complete mailbox audit. Empty-query browsing has
no ranked-candidate cap, and accepts offsets up to one million.

## Coverage And Reads

`index_status` and every search page report folder inventories, missing headers,
old flag observations, pending/unavailable/truncated bodies, pending embeddings,
and sanitized refresh errors. A cold index is explicitly incomplete. `complete`
describes indexed coverage of the observed snapshot, not proof that Bridge has
finished synchronizing with its provider. It is conservative across all indexed
folders, even when a query selects just one. A reached candidate cap is reported
separately from coverage.

`get_mail` uses cached parsed bodies when available and returns `cached`, `as_of`,
and `stale`. `refresh=true` bypasses the local copy. These fields are observational,
not a guarantee that a provider has not changed since the last successful scan.
Original attachment bytes, including inline images, are still fetched on demand.
Attachments and PDF contents are not included in body full-text search.

Indexing retains at most 500,000 body characters per message, with a visible
truncation count; a truncated body is not served as a complete cached `get_mail`
response. Semantic indexing uses up to 256 chunks of 1600 body characters with a
bounded subject/sender prefix. Chunk-limit truncation is also reported. Embedding
requests use `truncate=NONE` on NVIDIA endpoints so an incompatible model/context
limit fails visibly rather than silently dropping the tail. Chunks preserve quoted
history; this version does not attempt destructive thread/quotation stripping.

## Synchronization

Indexing enumerates selectable folders, performs complete UID inventory checks,
then fetches bounded batches of headers and flags. Unknown UIDs from the last
30 receipt-date days are prioritized with IMAP `SINCE` when backfill is large.
Within that discovery window UID ordering is only a work-order hint, never the
date ordering shown in search results. Bodies are prioritized by observed receipt
dates and fetched with `BODY.PEEK` without marking mail as read.

Each sweep processes at most the configured body count and normally no more than
`IMAP_MAX_MESSAGE_BYTES` of body traffic in aggregate, based on observed sizes.
Each individual read retains the backend's size checks. Metadata batches are
bounded per folder, but enumerating folder UID inventories still depends on
mailbox size; these are work-count budgets, not a hard sweep wall-clock deadline.
Unknown-size messages are not fetched as if they were small.

Existing flags rotate through bounded refresh batches. Old flags make unread-filter
coverage incomplete; IDLE is a wakeup hint, not a reliable deletion/flag change log.
Unwatched folders are reconciled periodically, without adding more IDLE connections.
Epochs are checked before and after fetches. A failed scan retains the last valid
folder snapshot and reports stale state, rather than inventing an empty mailbox.

Removed locations immediately stop qualifying for search and cached reads. Bodies,
text-index entries, and vectors with no remaining location are purged after all
folder/header inventories are usable. This avoids throwing away bodies during a
move while the destination headers are still being indexed. New header observations
at a native ID invalidate old body/chunk content if their header fingerprint changes.

The existing EMAILID / recognized-provider-ID / folder+UIDVALIDITY+UID precedence
is unchanged. Internal row IDs and hashes are never new public email identifiers.
Bulk indexing stores locations in its own database and warms the old live lookup
cache only for requested reads. Every original-byte read still validates its
location and identity through the IMAP backend.

Index workers never write notification checkpoints or emit historical events.
Existing IDLE notifications and their persisted startup floor continue independently.
Rebuilding the derived database does not replay notifications. Retain the original
`/data/imap.sqlite3` volume: it is not a disposable search cache.

## Embeddings

An empty model disables all embedding requests. Enabling a model explicitly sends
indexed email text and search queries to the configured endpoint; treat that as a
separate privacy decision from IMAP access. It does not depend on Graphiti or insert
mail into the agent's memory graph. The API/provider/dimension configuration follows
the same pattern as graph-memory embeddings but is independently configured.

Document chunks are embedded in background batches of at most 16. Failures retain
pending work and back off at least 30 seconds, honoring longer Retry-After hints.
Keyword search and IMAP synchronization remain available. A failed explicit semantic
query is reported as unavailable; hybrid falls back to lexical results visibly.

Changing model, endpoint, provider, or requested dimensions rebuilds the vector
index from cached text without resetting bodies or mail notifications. Credential
rotation does not invalidate vectors. Silent upstream changes behind the same
model alias require an operator-triggered rebuild. Dimensions, returned indices,
and finite/nonzero vectors are validated before storage.

## Storage And Helm

The index uses a separate account-scoped SQLite file. New directories are private
and database files are mode 0600; storage still needs appropriate host/PVC access
control because it contains decrypted mail text. Neither SQLite nor this adapter
adds encryption at rest.

The chart defaults `/index` to a 5 GiB pod-local `emptyDir`, rather than assuming
the existing notification PVC is safe or large enough. Container restarts retain
that volume, but pod replacement loses it and requires backfill again. Standalone
execution defaults to `/tmp/imap-index`; set `IMAP_INDEX_DIR` explicitly for persistence.

For durable indexing, provision a **local/block-backed** index claim:

```yaml
services:
  imapMcp:
    index:
      enabled: true
      storage:
        persistent: true
        storageClass: your-block-storage-class
        size: 5Gi
      embedding:
        model: nvidia/nvidia/nemotron-3-embed-1b
        baseUrl: https://your-nvidia-gateway.example/v1
        provider: nvidia
        dim: 2048
        apiKeySecretRef:
          name: model-gateway-credentials
          key: OPENAI_API_KEY
```

Alternatively set `index.storage.existingClaim`. Creating a new persistent claim
requires an explicit storage class; it does not inherit an NFS-backed global class.
The example model ID is gateway-specific; use the exact NVIDIA-hosted ID your
endpoint exposes. An empty secret name reuses the chart's main Secret and the
configured key. No embedding secret is mounted when embeddings are disabled.

SQLite WAL is opt-in with `index.journalMode: WAL` and must not be used on NFS.
DELETE mode does not make arbitrary network filesystem locking safe either;
local/block storage is recommended for both modes. One IMAP MCP replica owns the
index. Size the claim for text, FTS, vectors, and SQLite overhead, not attachment
sizes. Data is not automatically evicted to fit the volume.

Set `index.enabled: false` to disable all index workers; explicit `search_server`
and live body/attachment reads remain available. Upgrade the IMAP MCP image/chart
and reconnect session-manager to refresh the changed search schema/instructions.
No Bridge restart or mailbox mutation is required.
