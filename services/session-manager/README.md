# Session Manager

## Model Retries

Model requests retry HTTP 429, 408, 409, 5xx, and connection/timeout failures.
The policy applies to both normal streaming turns and non-streaming compaction.
Authentication errors, invalid requests, known exhausted billing/quota errors,
and responses explicitly carrying `x-should-retry: false` do not retry.

Configure the policy in `agent.yaml`, beside `model.config`, not inside it:

```yaml
model:
  name: nvidia/moonshotai/kimi-k3
  retry:
    max_attempts: 8
    max_elapsed_seconds: 300
    initial_delay_seconds: 2
    max_delay_seconds: 60
```

These are the defaults, so existing deployments need no configuration changes.
`max_attempts` includes the initial request; setting it to 1 disables retries.
The elapsed budget limits when another attempt may start, including cooldown
waits and time spent on failed requests. It does not interrupt an in-flight
request; the existing 300-second SDK HTTP timeout still applies. Settings are
validated at startup and are not sent to the provider.

Delay precedence is:

1. `retry-after-ms` in milliseconds.
2. `Retry-After` in seconds or HTTP-date form.
3. For HTTP 429 only, an explicit `limit will reset in N seconds` message
   (also accepting milliseconds/minutes), including a nested LiteLLM error.
4. Exponential backoff starting at two seconds and capped at sixty seconds.

Positive jitter, up to 25% or one second, is added to avoid synchronized retries.
Missing, malformed, expired, and non-finite hints fall back to the next source.
`max_delay_seconds` caps only the exponential component, not server hints or
jitter. A server wait that exceeds the remaining budget is never shortened to
send an early retry. Retry logs contain status, attempt, delay, and hint source,
not provider error bodies.

One orchestrator shares a cooldown across its sessions and compaction requests.
This does not coordinate separate pods or other applications sharing the same
upstream account, nor does it recall requests already in flight. SDK automatic
retries are disabled to avoid multiplying attempts beneath this policy.
Cooperative cancellation wakes backoff waits; task cancellation propagates.

Only the model request retries. Previously completed tools are not re-executed.
Once an HTTP stream is established, stream failures are not retried: text and
tool fragments might already have arrived. Failed streams are closed, and partial
output is recorded as an unexecuted `model_error` event, not a completed answer or
executable tool call.

If a normal turn ultimately fails, its input events, completed tool calls/results,
and a safe error marker are persisted so the next turn retains what happened.
The next event in the same session, including one queued during backoff, is
processed together with that retained context. This also works after restarting
session-manager with the same session volume. Failed input is not enqueued a
second time, and completed tool calls are replayed only as history, not executed
again. The agent is instructed to consider unfinished work alongside the new
event rather than treating the failed turn as completed.
There is no automatic durable rescheduling after the retry budget is exhausted,
and pending in-memory work does not survive a process restart. Compaction failure
leaves the existing history version intact.

For Helm, continue passing `agent.yaml` through `--set-file agent.config=...`.
Restart session-manager after updating its image or configuration; MCP servers
and channel adapters do not need restarting for this change.
