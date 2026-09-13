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

## Graph Exploration

The graph MCP exposes structural exploration tools alongside semantic and keyword
search. These tools use recorded relationships and do not call an embedding or
language model.

| Tool | Purpose |
|---|---|
| `get_neighborhood(name, max_hops=2, limit=50, direction="both")` | Explore nearby facts and entities, including hop distances. Direction can be `both`, `outgoing`, or `incoming`. |
| `find_path(source, target, max_hops=4, limit=100, directed=false)` | Return one shortest path in the bounded exploration, with ordered facts and explicit traversal direction. |
| `explain_fact(fact_id, source_limit=5, story_chars=2000)` | Inspect the stored evidence label, rationale, provenance, date corrections, and supersession reason. |

Neighborhood and path requests allow at most six hops and 100 explored facts.
They have a ten-second overall deadline; each frontier query has a two-second
database execution limit. An extra fact may be read to detect truncation. A
`truncated: true` result is incomplete; `found: false` means only that the bounded
search found no route. Graph reads are not snapshots of concurrent writes.

Superseded facts are excluded before every traversal step unless
`include_superseded=true`. They cannot act as hidden bridges in a current
exploration. A historical path may combine facts whose validity dates do not
overlap. Reverse traversal preserves each fact's original subject/object and
sets `traversed_forward=false`; a connection is not a new transitive or causal
claim.

Facts may include `evidence_kind` (`reported`, `inferred`, `uncertain`, or
`unspecified`) and `rationale`. The label records how a claim was established,
not a probability or independent truth verification. Legacy facts remain
`unspecified`. Explanations report unavailable source links and truncation
explicitly; use `get_story` to read a full narrative when needed. Source previews
are limited to ten records and 5000 narrative characters per record.

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
| memory-mcp | 8001 | memory |
| system-mcp | 8002 | — |
| time-mcp | 8300 | time |
| graphiti-mcp | 8005 | — |
| falkordb | 6379 | graph |

All ClusterIP. Nothing exposed externally.

## Upgrade

```bash
helm upgrade <release> ./charts/autonome --namespace <release> --reuse-values
```
