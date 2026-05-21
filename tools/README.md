# Tools

One-off operational scripts that don't belong in any service.

## matrix-cross-sign.py

Cross-sign a Matrix device using the user's SSSS recovery key. Used to bootstrap an agent's Matrix device into a user's cross-signing trust chain without an interactive Element session — particularly for the case where the device's keys exist but aren't yet signed by the user's self-signing key, which causes Element peers to silently refuse to share Megolm sessions.

### Install

```bash
python -m venv /tmp/cross-sign-venv
/tmp/cross-sign-venv/bin/pip install 'mautrix[e2e]'
```

### Run

```bash
/tmp/cross-sign-venv/bin/python tools/matrix-cross-sign.py \
  --homeserver https://matrix.example.com \
  --user-id @agent:example.com \
  --access-token "$TOKEN" \
  --device-id AGENT_DEVICE \
  --recovery-key "EsT9 RzbW ..."
```

### Where to get the inputs

- `--access-token`: from the agent's `credentials.json` inside the matrix-crypto PVC. `kubectl -n <ns> exec deploy/<release>-matrix-adapter -- cat /data/crypto/credentials.json | jq -r .access_token`
- `--device-id`: also in `credentials.json` (`.device_id`), or look it up with `keys/query`
- `--recovery-key`: from when you (or someone) ran "Reset identity" in Element-as-the-user. Stored in your password manager, hopefully.

### After running

The agent's device is now signed by the user's self-signing key. To close the trust chain to a human user (so e.g. Element-as-Max trusts the agent), the human must verify the user-level identity once: in their Element, open the user's profile → "Verify user". This signs the user's master key with the human's user-signing key. Persistent across restarts as long as the agent's `credentials.json` and crypto store survive.
