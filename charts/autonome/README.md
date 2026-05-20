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

## Secrets

Two modes, controlled by `secrets.create`:

- **Chart-managed** (default): set `secrets.values.openaiApiKey`, `secrets.values.matrixPassword`, and optionally `secrets.values.searchApiKey` in your values file. The chart creates `<release>-secrets`. Values are plain text in your values.yaml — fine for a local private file, **don't commit it to a public repo**. For GitOps, encrypt with SOPS, sealed-secrets, etc., or switch to the external mode.
- **External**: set `secrets.create: false` and either let it default to `<release>-secrets` or set `secrets.existingName: <your-secret-name>`. You create the Secret out of band with `kubectl create secret generic …` or via your secrets controller of choice.

Keys (both modes): `OPENAI_API_KEY` (required — passed straight to the LLM client; works with any OpenAI-compatible endpoint), `MATRIX_PASSWORD` (required), `SEARCH_API_KEY` (optional).

## Personality and agent.yaml

These live in the `<release>-config` PVC. Bootstrap however suits the cluster — `kubectl cp` from a temp pod, an init Job, pre-provisioned PV. Session-manager mounts them read-only at `/app/agent.yaml` and `/app/PERSONALITY.md`.

## Seeding PVCs from existing state

If your storage class exposes the backing directories on the host filesystem (NFS, hostPath, local-path) you can just `cp -r` your existing state into the bound PVC directories directly — no need to go through a pod.

If you don't have host access to the backing storage, `examples/seed-pod.yaml` is a scratch pod that mounts every PVC at `/mnt/<name>` so you can `kubectl cp` into it. Replace `HEATHER` with your release name, apply, copy, delete.

Matrix encryption keys (`matrix-crypto/`) are the load-bearing one either way — copy that intact or the agent loses E2E history.

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
