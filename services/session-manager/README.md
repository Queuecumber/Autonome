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

## Mid-Turn Context Recovery

Normal compaction still runs before a turn. Between tool rounds, session-manager
also checks the latest provider-reported prompt usage against
`session.compaction_trigger_tokens`. A known context-window HTTP 400/413 rejection
can trigger emergency compaction even when the previous usage was below that
threshold or the provider omitted usage. Other invalid requests do not compact,
and the ordinary retry layer does not blindly retry the same oversized payload.

Before compacting, input events and all completed current-turn calls/results are
appended once to the original session version. Recovery uses a separate tool-free
summarization request: it cannot execute tools, send messages, or save memories.
The continuation resumes the existing tool loop with a smaller context rather
than restarting the task or replaying completed actions.

Recovery preserves complete tool-call/result batches and event metadata/text
pairs in its recent tail. Original event-routing metadata is also retained verbatim
when it fits a 16,000-character bound, so summary generation need not reconstruct
room/recipient identifiers. Oversized routing metadata is explicitly marked omitted.
An oversized latest exchange is summarized as a whole,
not split into invalid orphaned tool messages. A provider rejection can also force
summarization of a transcript below the normal tail budget. Original full records remain in
the previous on-disk version. The new active version is published by atomic rename
only after serialization, writing, and summary validation succeed. Summary/write
failure leaves the original transcript available, including completed work.

Bounds per turn:

- At most two compaction attempts, including a proactive attempt between rounds.
- Verbatim tail budgets of 64,000 then 16,000 JSON characters. These are payload
  bounds, not an exact tokenizer count for arbitrary provider models.
- Recovery summaries consume transcript fragments of at most 64,000 characters,
  shrinking to as little as 1,024 if the provider rejects a summary request too.
- At most 32 logical summary requests per attempt; each uses the normal transient
  request retry policy and cooperatively observes cancellation.
- Summary output allowance is at most 8,192 tokens, respecting a smaller configured
  output allowance. Empty, oversized, truncated, filtered, or tool-calling responses
  cannot become a checkpoint.

Transient images are not put into textual summaries or saved as base64 in the new
history. Recovery retains the most recent image batch when possible and explicitly
records omitted image counts in a `context_recovered` event. A stronger recovery,
or image-only overflow with no smaller usable batch, can omit all transient images;
the agent is told to fetch only needed resources/pages again rather than assume
those images were read. This does not add PDF page-range controls.

The original system instructions, available tools, and normal generation output
budget are not silently reduced. If those fixed inputs alone cannot fit, the
summary provider fails, or recovery cannot produce smaller input within its limits,
the turn ends with a persisted `model_error` marker. Retained work joins the next
event as usual. No hard provider-window guarantee is inferred from character counts.

This needs only an updated session-manager image and restart. No session-volume
migration or adapter/MCP restart is required.
