# autonome chart

Deploys one Autonome agent — session-manager + matrix-adapter + MCP servers — into a Kubernetes namespace. One Helm release per agent.

## Prerequisites

- Kubernetes 1.24+
- A `StorageClass` (override `storage.storageClass` if not using the cluster default)
- Images published to GHCR by the repo's `Build images` CI workflow

## Install

```bash
helm install heather ./charts/autonome \
  --namespace heather --create-namespace \
  --set image.tag=unstable \
  --set storage.storageClass=local-path \
  --set matrix.homeserver=https://matrix.example.com \
  --set matrix.userId=@heather:matrix.example.com
```

Then bootstrap the prerequisites the chart's NOTES describe:

1. A Secret named per `secrets.name` (default `autonome-secrets`) with `NIM_API_KEY`, `MATRIX_PASSWORD`, and optionally `SEARCH_API_KEY`.
2. `agent.yaml` and `PERSONALITY.md` placed inside the config PVC (`<release>-config`).

## Values

See `values.yaml`. Anything in there is `--set`-able.

## What's deployed

| Service | Port | Volumes | Notes |
|---|---|---|---|
| session-manager | 5000 | config (ro), sessions, binaries | Orchestrator + LLM client |
| matrix-adapter | 8200 | matrix-crypto | Matrix sync + tools |
| workspace-fs-mcp | 8000 | workspace | Read/write workspace files |
| memory-mcp | 8001 | memory | Long-term memory store |
| system-mcp | 8002 | — | Web search bridge |
| time-mcp | 8300 | time | Continuity cron |

All Services are ClusterIP. Nothing is exposed externally — matrix-adapter syncs outbound to the homeserver.

## Upgrade

```bash
helm upgrade heather ./charts/autonome --namespace heather --reuse-values
```

Pods use `strategy: Recreate` because PVCs are RWO; the new pod can't bind the volume until the old one's gone.
