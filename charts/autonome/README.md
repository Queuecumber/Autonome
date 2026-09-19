# autonome chart

Deploys one Autonome agent — session-manager + matrix-adapter + MCP servers — into a Kubernetes namespace. One Helm release per agent.

## Install

```bash
helm install <release> ./charts/autonome \
  --namespace <release> --create-namespace \
  --set image.tag=unstable \
  --set storage.storageClass=<your-sc> \
  --set matrix.homeserver=https://matrix.example.com \
  --set matrix.userId=@agent:matrix.example.com \
  --set openai.apiKey=… \
  --set matrix.password=… \
  --set-file agent.config=./agent.yaml \
  --set-file agent.personality=./PERSONALITY.md
```

`--set-file` injects the file contents into the chart's ConfigMap. Session-manager mounts them at `/app/agent.yaml` and `/app/PERSONALITY.md`.

## Secrets

`secrets.create: true` (default) writes a Secret from the inline values (`openai.apiKey`, `matrix.password`, `system.searchApiKey`). Plain-text in values, so don't commit them — fine in private files.

`secrets.create: false` references an existing Secret (`<release>-secrets`, or override via `secrets.existingName`). Use this with sealed-secrets, SOPS, vault, etc.

Keys: `OPENAI_API_KEY` (required), `MATRIX_PASSWORD` (required), `SEARCH_API_KEY` and `EMBEDDING_API_KEY` (optional).

## Memory Deployment

Graph memory replaces the legacy markdown memory MCP in this chart. The chart no
longer deploys the `memory-mcp` Deployment or Service, and release builds no longer
publish that service's image. Old `services.memoryMcp` values left by
`--reuse-values` are ignored; they do not restore the retired deployment.

Replace any legacy local memory endpoint in your supplied `agent.config` with:

```yaml
mcp_servers:
  graph: http://graphiti-mcp:8005/mcp
```

Keep your other MCP registrations. The chart does not rewrite operator-supplied
agent configuration, including custom URLs or authentication headers. Restart
session-manager after updating its mounted config so it reconnects to the graph
service rather than retrying a removed memory endpoint.

The existing `<release>-memory` PVC remains as an unmounted archive, with
`helm.sh/resource-policy: keep`, so this upgrade does not delete markdown history.
The legacy Python package remains available for manual archival access. No automatic
markdown-to-graph import or source-data deletion is performed. On a fresh install,
set `storage.memory: null` to avoid creating that archive claim. For an existing
release, first apply this upgrade with the archive retained; only then omit it
after verifying backups. Kept PVCs require deliberate manual cleanup later.

## FalkorDB Storage

The graph PVC is mounted at `/var/lib/falkordb/data`, matching the deployed
FalkorDB image's persistence directory, rather than `/data`. The existing
`<release>-graph` claim name is unchanged.

Before upgrading a deployment that still mounts only `/data`, check the running
server's actual directory with `redis-cli CONFIG GET dir`. If it has been writing
outside the PVC, take a verified backup and migrate the persistence files onto
the graph PVC **before replacing the pod**. A mount-path change alone does not
copy data and can otherwise hide the old container's persistence files behind an
empty mount. This chart does not run an automatic migration or change the Redis
RDB/AOF policy. Deployments already corrected to `/var/lib/falkordb/data` need no
path migration; keep their existing claim.

## Graph Memory Terminology

Stories are the memories; entities and directed relationships help locate them.
Descriptions may record observations, feelings, interpretations, or uncertainty.
An evidence label describes how something was recorded, not whether it is
objectively verified. `story` is the narrative; `source` is its attribution.

`save_memory(relationships, story, source)` records them together. Each
relationship supplies `subject`, `relation`, `object`, and `description`, with
optional entity types, dates, and reasoning. Reuse `story_id` to attach more
relationships to an existing story instead of duplicating its narrative.

| Previous API | Current API |
|---|---|
| `save_facts(facts=...)` | `save_memory(relationships=...)` |
| `search_facts(query)` | `search(query, kind="relationships")` |
| `list_facts(...)` | `list_relationships(...)` |
| `get_neighborhood(...)` | `explore(...)` |
| `explain_fact(fact_id)` | `explain_relationship(relationship_id)` |
| `supersede_fact(fact_id)` | `supersede(relationship_id)` |
| `set_fact_valid_at(fact_id, ...)` | `set_valid_at(relationship_id, ...)` |

Public fields use `relationship_id`, `description`, `relationships`, `story_id`,
`relationship_count`, `entities`, `explored_relationships`, and `missing_story_ids`
instead of `fact_id`, `fact`, `facts`, `episode_id`, `fact_count`, `nodes`,
`explored_facts`, and `missing_episode_ids`. `get_story` takes `story_id`;
entity IDs identify entities, never relationships. The relationship input model
is named `Relationship`. The predicate field remains `relation`.

`search(query)` now finds entities by name or meaning, returning up to five
relationship previews per entity by default. `relationship_limit` can be set
from 1 to 20; `relationships_truncated` identifies incomplete previews. Search
relationship descriptions explicitly with `kind="relationships"`. Both modes
combine optional embeddings with keyword retrieval. `include_superseded` controls
relationship history, not whether an entity exists.

`list_entities(limit, cursor)` lists actual entities, including isolated ones;
`list_relationships(limit, cursor)` inventories relationships, including history.
Both are bounded to 100 results per page and return `next_cursor`. They order by
ID rather than date and are not snapshots of concurrent changes. `get_entity`
returns an entity's `entity_id` and all its relationships, including history.

These are intentional MCP API renames; the old tool names are not advertised.
Refresh session-manager's tool discovery and update any scripted callers after
deploying the graphiti-mcp image. Stored Graphiti property names, UUIDs, vectors,
stories, and date-correction history are unchanged. No FalkorDB migration,
re-embedding, deletion, or restart is required.

## Graph Model And Resources

The graph MCP separates persistence, the public data model, and the MCP view:

- `store.py` handles Graphiti/FalkorDB persistence; `traversal.py` handles bounded
  graph algorithms and `embed.py` handles the embedding endpoint.
- `model.py` owns backend coercion and memory data flow. Its operations return
  Pydantic objects such as `MemorySaved`, `RelationshipRecord`, `EntityResult`,
  `Story`, typed inventory pages, paths, and explanations. Nested corrections and
  source previews are models too, and timestamps are native Python `datetime`s.
- `server.py` contains prompting, tool registration, and resource handlers. Tools
  expose model operations directly, so FastMCP derives output schemas and handles
  JSON serialization. Resource handlers put typed objects into `ResourceContent`,
  without converting them to dictionaries or writing custom JSON serialization.

The existing tool names and identifier fields remain available. JSON responses
also expose `uri` links for entities, relationships, and stories, and `story_uri`
on saved batches and relationships with provenance:

| Resource | Content |
|---|---|
| `memory:///entities/{entity_id}` | Entity details and its relationships, including history. |
| `memory:///relationships/{relationship_id}` | One relationship, with dates and correction history. |
| `memory:///stories/{story_id}` | Full narrative, attribution, and recording/source dates. |

All resources use `application/json` and enforce the configured memory group.
Unknown resource IDs raise errors rather than returning invented records.
Name-based tool lookups retain `found=false` for absent entities; resource reads
are by stable UUID. Story reuse is also restricted to the configured group.

Output schemas now explicitly describe optional fields and empty collections,
including empty `valid_at_corrections` lists. Dates remain ISO-8601 on the wire;
the standard serializer uses `Z` for UTC. Consumers should parse timestamps
rather than rely on the former `+00:00` spelling. No stored-data migration is
required. After updating the graph MCP image, reconnect session-manager so it
discovers the output schemas and the `memory` resource scheme.

## Graph Exploration

The graph MCP exposes structural exploration tools alongside semantic and keyword
search. These tools use recorded relationships and do not call an embedding or
language model.

| Tool | Purpose |
|---|---|
| `explore(name, max_hops=2, limit=50, direction="both")` | Explore nearby relationships and entities, including hop distances. Direction can be `both`, `outgoing`, or `incoming`. |
| `find_path(source, target, max_hops=4, limit=100, directed=false)` | Return one shortest path in the bounded exploration, with ordered relationships and explicit traversal direction. |
| `explain_relationship(relationship_id, source_limit=5, story_chars=2000)` | Inspect saved reasoning, provenance, date corrections, and the supersession reason for one relationship. |
| `explain_entity(name, limit=5, source_limit=3, story_chars=1000)` | Explain an entity's nearby relationships, with bounded source previews. |

Exploration and path requests allow at most six hops and 100 explored relationships.
They have a ten-second overall deadline; each frontier query has a two-second
database execution limit. An extra relationship may be read to detect truncation. A
`truncated: true` result is incomplete; `found: false` means only that the bounded
search found no route. Graph reads are not snapshots of concurrent writes.

Superseded relationships are excluded before every traversal step unless
`include_superseded=true`. They cannot act as hidden bridges in a current
exploration. A historical path may combine relationships whose validity dates do not
overlap. Reverse traversal preserves each relationship's original subject/object and
sets `traversed_forward=false`; a connection is not a new transitive or causal
claim.

Relationships may include `evidence_kind` (`reported`, `inferred`, `uncertain`, or
`unspecified`) and `rationale`. The label records how a memory was formed,
not a probability or independent truth verification. Legacy relationships remain
`unspecified`. Explanations report unavailable source links and truncation
explicitly; use `get_story` to read a full narrative when needed. Source previews
are limited to ten records per relationship and 5000 narrative characters per
record. Entity explanations cover at most 20 relationships, with a ten-second
overall deadline and explicit truncation. Use `explain_relationship` to inspect
a specific relationship outside the preview.

Deploy the updated graphiti-mcp image and restart session-manager after the
service is ready to discover these tools. No graph migration is required.

## Graph Memory Embeddings

Embeddings are optional. Leaving `services.graphitiMcp.embedding.model` empty keeps
keyword search, graph lookups, and memory writes available without an embedding
endpoint. Register `graph: http://graphiti-mcp:8005/mcp` under `mcp_servers` in the
agent configuration to expose the graph tools.

For NVIDIA-hosted text retrieval, the recommended starting model is
`nvidia/nvidia/nemotron-3-embed-1b` on a gateway using the `nvidia/` route prefix.
Use the exact model ID returned by your gateway's `/models` endpoint. NVIDIA's
direct public endpoint uses `nvidia/nemotron-3-embed-1b` instead.

```yaml
services:
  graphitiMcp:
    embedding:
      model: nvidia/nvidia/nemotron-3-embed-1b
      baseUrl: https://your-model-gateway.example/v1
      provider: nvidia
      dim: 2048
      timeoutSeconds: 20
      minScore: 0.6
      apiKeySecretRef:
        name: gateway-secrets
        key: OPENAI_API_KEY
```

The referenced Secret must be in the release namespace. To reuse the release's
existing `OPENAI_API_KEY`, leave `apiKeySecretRef.name` empty and set its `key` to
`OPENAI_API_KEY`. Alternatively, supply `embedding.apiKey` with
`secrets.create=true`; the chart writes `EMBEDDING_API_KEY` into its managed
Secret. The older `mcp.secretEnv.EMBEDDING_API_KEY` route remains supported, with
the dedicated `embedding.apiKey` taking precedence. Inline keys and an explicit
secret reference cannot be combined.

| Setting | Default | Behavior |
|---|---|---|
| `model` | empty | Exact API model ID; empty disables embeddings. |
| `baseUrl` | empty | API base URL, including its version path; required when a model is set. |
| `provider` | `nvidia` | `nvidia` sends `input_type=query` for searches and `passage` for saved text, with `truncate=END`. `openai` sends only standard compatible API fields. |
| `dim` | `0` | Zero requests native dimensions. A positive value is sent as `dimensions` and the response length must match. No local vector slicing occurs. |
| `timeoutSeconds` | `20` | Per-request timeout, with automatic retries disabled. Failed embeddings do not prevent saving facts or using keyword search. |
| `minScore` | `0.6` | Minimum Graphiti similarity score, between 0 and 1. On FalkorDB this is `(1 + cosine) / 2`, so 0.6 corresponds to raw cosine 0.2. Calibrate against representative memories. |
| `apiKey` | empty | Optional key for the Helm-managed Secret. |
| `apiKeySecretRef.name` | empty | Existing Secret name; empty uses the release's configured Secret. |
| `apiKeySecretRef.key` | `EMBEDDING_API_KEY` | Key within that Secret. |

Nemotron-3-Embed-1B's native output is 2048 dimensions. Its NVIDIA NIM API supports
omitted dimensions or `2048`; do not request the old 1024-dimensional default.
See the [model card](https://build.nvidia.com/nvidia/nemotron-3-embed-1b/modelcard)
and [NIM API contract](https://docs.nvidia.com/nim/nemo-retriever/embedding/2.2/reference.html).

Model or dimension changes require re-embedding existing vectors before combining
them with the new embedding space. Facts saved while embeddings were disabled
are not automatically backfilled. This chart configures requests; it does not
migrate stored vectors.

Changing Helm-managed credentials changes the graph pod template automatically.
After rotating credentials in an externally managed Secret, restart graphiti-mcp.
After replacing graphiti-mcp, restart session-manager once the graph service is
ready so its MCP connection is refreshed.

## What's deployed

| Service | Port | Volumes |
|---|---|---|
| session-manager | 5000 | agent-config (ConfigMap, ro), sessions, binaries |
| matrix-adapter | 8200 | matrix-crypto |
| workspace-fs-mcp | 8000 | workspace |
| system-mcp | 8002 | — |
| time-mcp | 8300 | time |
| graphiti-mcp | 8005 | — |
| falkordb | 6379 | graph at `/var/lib/falkordb/data` |

All ClusterIP. Nothing exposed externally.

## Upgrade

```bash
helm upgrade <release> ./charts/autonome --namespace <release> --reuse-values
```

## Optional Mail Services

IMAP and SMTP ports from aibs are disabled by default. Enable only the needed
services and register their MCP URLs in `agent.config`. IMAP pushes incoming-mail
events directly to session-manager; it needs no scheduled agent inbox check.
SMTP is explicit outbound sending, not an automatic responder.

```yaml
services:
  imapMcp:
    enabled: true
    server: imaps://imap.example.test
    username: agent@example.test
    passwordSecretRef: { name: mail-credentials, key: IMAP_PASSWORD }
    folders: [INBOX]
    eventEnergy: passive
    stateStorage: { size: 100Mi }
  smtpMcp:
    enabled: true
    server: starttls://smtp.example.test:587
    username: agent@example.test
    passwordSecretRef: { name: mail-credentials, key: SMTP_PASSWORD }
    from: agent@example.test
    allowedRecipients: [owner@example.test]
```

The referenced Kubernetes Secret must already exist in the release namespace;
these credentials are not copied into chart values or session-manager's env.
For an in-cluster relay that trusts pod networks, use `server: smtp://host:25`
and leave `username` and `passwordSecretRef.name` empty together; the service
then connects without TLS or AUTH, and no Secret is needed for SMTP.
External password rotation requires restarting the corresponding mail deployment.
Keep the IMAP deployment at one replica, with its Recreate strategy and persistent
`<release>-imap` PVC, to retain checkpoints and pending events. The PVC is only
created when IMAP is enabled. SMTP requires no additional volume.

Add whichever services are enabled to the existing agent configuration:

```yaml
mcp_servers:
  imap: http://imap-mcp:8006/mcp
  smtp: http://smtp-mcp:8007/mcp
```

Register IMAP's MCP even when only using push notifications, so the agent can
retrieve message bodies and attachment resources. For Proton Mail, run a Bridge pod in the
release namespace and set `server: imap://protonmail-bridge:143` with the
bridge-local mailbox password in the referenced Secret. First IMAP startup is quiet
for existing mail; subsequent arrivals trigger events. Servers lacking IDLE use
adapter-side polling. HTTP acceptance is not durable agent-processing acknowledgement;
ambiguous responses may repeat an event. See the [IMAP service documentation](../../services/imap-mcp/README.md)
and [SMTP service documentation](../../services/smtp-mcp/README.md) for guarantees,
environment variables, migration differences, and outbound restrictions.

## Optional iCal Feeds

iCal is also disabled by default. Put a JSON calendar-name/HTTPS-URL mapping in
the `ICAL_URLS` key of an existing Secret in the release namespace, for example
`{"Personal":"https://calendar.example.test/private.ics"}`. Treat the entire
URL as a credential; do not put real private feed URLs into a committed values file.

```yaml
services:
  icalMcp:
    enabled: true
    urlsSecretRef: { name: calendar-feeds, key: ICAL_URLS }
    refreshSeconds: 300
    eventEnergy: passive
    stateStorage: { size: 100Mi }
```

Register `ical: http://ical-mcp:8008/mcp` in `agent.config.mcp_servers`. The adapter
refreshes feeds and pushes source changes to session-manager; no recurring agent
calendar-check task is necessary. `services.icalMcp.timezone` defaults to the
chart's global timezone, then UTC, for feeds with floating dates and no declared
timezone. First loading is quiet. Calendar changes are not appointment reminders.

The `<release>-ical` PVC retains snapshots and pending events. Keep one replica
and the Recreate strategy. Rotate the external Secret by restarting the iCal
deployment; changing a feed URL establishes a new baseline. See the
[iCal service documentation](../../services/ical-mcp/README.md) for stale reads,
recurrence handling, token privacy, and delivery limitations. Outlook is not
being ported because it is superseded by the NVIDIA-provided integration.

### Notification History Cutoff

Both read adapters default `notifySince` to `startup`. The first startup with
cutoff support persists a date boundary in the existing adapter PVC, including
when upgrading an older deployment. Restarts preserve that boundary. Set an
explicit date to retain notifications from an earlier point:

```yaml
services:
  imapMcp:
    notifySince: "2026-09-14T00:00:00Z"
  icalMcp:
    notifySince: "2026-09-14"
```

`all` disables date filtering; the first snapshot still establishes a quiet
baseline. IMAP uses server receipt dates, so gradual Proton Bridge backfill of
old messages does not become new-mail wakeups. Calendar filtering checks actual
recurrences and both sides of a change, retaining future anniversaries and
cancellations/reschedules. Historical read/search tools are never date-filtered.

The updated adapters also filter their pending outboxes. They cannot retract
notifications already accepted or queued by session-manager. Advancing a cutoff
does not delete source data; moving it backward does not replay skipped messages
or source changes. With an unchanged deployment spec, pulling the updated IMAP
and iCal images enables the default `startup` policy without new env variables.
