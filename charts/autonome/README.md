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

Keys: `OPENAI_API_KEY` (required), `MATRIX_PASSWORD` (required), `SEARCH_API_KEY` (optional).

## Seeding existing state

PVCs (`<release>-sessions`, `-binaries`, `-workspace`, `-memory`, `-matrix-crypto`, `-time`) start empty. If you're migrating an existing agent, copy your prior directories in before the pods successfully boot. For NFS / hostPath / local-path storage classes that expose backing directories, `cp -r` directly. Otherwise mount via a scratch pod and `kubectl cp`.

Matrix `credentials.json` is the load-bearing one — if matrix-adapter password-logs-in against an existing `device_id`, server-side keys rotate and the prior token dies. To avoid: set `matrix.password` to a placeholder so initial login fails harmlessly, copy `credentials.json` in, let matrix-adapter restart and restore.

## What's deployed

| Service | Port | Volumes |
|---|---|---|
| session-manager | 5000 | agent-config (ConfigMap, ro), sessions, binaries |
| matrix-adapter | 8200 | matrix-crypto |
| workspace-fs-mcp | 8000 | workspace |
| memory-mcp | 8001 | memory |
| system-mcp | 8002 | — |
| time-mcp | 8300 | time |

All ClusterIP. Nothing exposed externally.

## Upgrade

```bash
helm upgrade <release> ./charts/autonome --namespace <release> --reuse-values
```
