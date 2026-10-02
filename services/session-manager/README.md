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

Every HTTP 429 also emits a `Model rate limit response` warning, including the
last attempt and non-retryable 429s. Its `response_headers` JSON contains only
allowlisted retry, rate-limit, response-date, and request/call-ID headers. This
includes `retry-after`, `retry-after-ms`, `x-should-retry`, `ratelimit` and
`ratelimit-policy`, the limit/remaining/reset fields (including `x-ratelimit-*`
request/token variants), and `x-request-id`, `request-id`, `x-litellm-call-id`.
Names are normalized to lowercase; each value is limited to 256 characters and
JSON-escaped to keep the diagnostic on one log line. `{}` means none of the
allowlisted headers reached the SDK. Request headers, cookies, authorization,
unlisted response headers, and the error body are not included in this diagnostic.
Logging is enabled by default and does not change retry decisions.

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

## Compaction During Turns

The same compaction routine runs before turns and is checked after every completed
tool batch, before the next model request. For parallel calls it waits for the
entire batch so no call loses its matching result. Each check compares the latest
provider-reported prompt usage with `session.compaction_trigger_tokens`; a long
turn can compact more than once.

`session.recency_tokens` selects the recent tail using usage from every model
iteration, not just the first call of each turn. Decreases in prompt usage do not
cancel later growth. If recorded growth is insufficient but the oldest measured
prompt already exceeds the recency target, its preceding history can be summarized
while the subsequently measured conversation remains intact. Cutoffs move backward
to preserve tool batches, event metadata/text pairs, and reasoning with responses.
This is a soft recency target: usage includes instructions and transient content,
and the latest tool results have not yet been measured by the provider.

The older portion is sent together through the normal summary prompt, with the
agent's usual system instructions, personality, model settings, and tools. The agent
can save memories before returning its summary. There is no separate emergency
prompt, character-sized chunk loop, or rolling summary of summaries. Logs identify
the fold/keep counts, summary requests and tool rounds, and publication elapsed time.

Before an in-turn compaction, input events and completed calls/results are appended
once to the current history version. Publication is atomic; old versions remain
the full audit trail. Empty, truncated, or filtered summaries and failed writes
cannot replace history. Compaction resumes the current tool loop without restarting
completed actions. Original routing metadata is carried forward within a bounded
16,000-character envelope, with larger metadata explicitly marked omitted.

Images belonging to retained tool batches stay in live context. Images from folded
batches are omitted explicitly in a `context_resumed` event and can be fetched again
using their resource references. Base64 is neither persisted nor included in textual
summaries. This does not add PDF page-range controls.

A known context-window HTTP 400/413 rejection can force one attempt through this
same compaction routine, even below the configured trigger. The reduced request is
retried once; other invalid requests do not compact. No token cutoffs are invented
when usage is unavailable. An individual oversized result, excessive fixed prompt
or tool overhead, or an already-overfull summary input can still fail. These failures
preserve history and record `model_error` for the next event instead of entering a
second summarization strategy. Configure the trigger with headroom for tool output.

This needs only an updated session-manager image and restart. No session-volume
migration or adapter/MCP restart is required.
