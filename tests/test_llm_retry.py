"""Provider backoff, bounded retries, cancellation, and durable turn history."""

import asyncio
from datetime import datetime, timezone
from email.utils import format_datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
import pytest

from session_manager import llm_retry
from session_manager.event import Event
from session_manager.llm_retry import ModelRequests, RetryBudgetExceeded, RetryPolicy, retry_hint
from session_manager.orchestrator import SessionOrchestrator


def failure(status=429, *, headers=None, message="throttled", code="rate_limit_exceeded"):
    """Build a synthetic SDK error with its real response/header interface."""
    response = httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://model.test"))
    cls = RateLimitError if status == 429 else APIStatusError
    return cls(message, response=response, body={"code": code})


@pytest.fixture
def clock(monkeypatch):
    """Advance only retry-module time, leaving asyncio's monotonic clock intact."""
    state = SimpleNamespace(now=1000.0, waits=[])
    monkeypatch.setattr(llm_retry, "time", SimpleNamespace(monotonic=lambda: state.now, time=lambda: state.now))
    monkeypatch.setattr(llm_retry.random, "uniform", lambda low, high: 0)

    async def wait(self, seconds, cancel):
        """Record requested delays and simulate their passage without real sleeps."""
        state.waits.append(seconds)
        state.now += seconds
        return cancel is None or not cancel.is_set()

    monkeypatch.setattr(ModelRequests, "_wait", wait)
    return state


@pytest.mark.parametrize("kwargs", [
    {"max_attempts": value} for value in (0, -1, 101, 1.5, True, "8")
] + [
    {name: value} for name in ("initial_delay_seconds", "max_delay_seconds", "max_elapsed_seconds")
    for value in (0, -1, float("nan"), float("inf"), True, "2")
] + [{"initial_delay_seconds": 61}])
def test_invalid_policy(kwargs):
    """Invalid retry limits fail at configuration time rather than while backing off."""
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


@pytest.mark.parametrize("headers, expected", [
    ({"retry-after": "10"}, (10, "retry-after")),
    ({"Retry-After": "1.5"}, (1.5, "retry-after")),
    ({"retry-after-ms": "250", "retry-after": "10"}, (0.25, "retry-after-ms")),
    ({"retry-after-ms": "invalid", "retry-after": "10"}, (10, "retry-after")),
    ({"retry-after": format_datetime(datetime.fromtimestamp(1010, timezone.utc), usegmt=True)},
     (10, "retry-after-date")),
    ({"retry-after": format_datetime(datetime.fromtimestamp(999, timezone.utc), usegmt=True)}, None),
    ({"retry-after": "invalid"}, None),
    ({"retry-after": "nan", "retry-after-ms": "inf"}, None),
    ({"retry-after": "0"}, None),
    ({"retry-after": "-10"}, None),
    ({}, None),
])
def test_header_delays(clock, headers, expected):
    """Headers take precedence and malformed, expired, or non-finite waits are ignored."""
    assert retry_hint(failure(headers=headers)) == expected


@pytest.mark.parametrize("message, expected", [
    ('litellm.RateLimitError: {"error":{"message":"Reached overall message rate limit. '
     'Please try again later. Your limit will reset in 10 seconds."}}\\n. Received Model Group=nvidia/test', 10),
    ("Your limit resets in 1.5 minutes.", 90),
    ("LIMIT WILL RESET IN 250 milliseconds", 0.25),
    ("Your limit will reset in 0 seconds.", None),
    ("Please retry later. Request contained a meeting in 10 minutes.", None),
])
def test_explicit_reset_message_fallback(clock, message, expected):
    """The nested proxy error from the report works even when headers are stripped."""
    hint = retry_hint(failure(message=message))
    assert hint == ((expected, "reset-message") if expected else None)
    assert retry_hint(failure(500, message=message)) is None
    assert retry_hint(failure(message=message, headers={"retry-after": "20"})) == (20, "retry-after")


@pytest.mark.asyncio
async def test_default_attempt_budget_and_exponential_cap(clock, caplog):
    """Fallback retries are bounded, observable, and do not log provider message bodies."""
    request = AsyncMock(side_effect=failure(message="PRIVATE_PROVIDER_CONTENT"))
    with pytest.raises(RateLimitError):
        await ModelRequests(RetryPolicy()).run(request)
    assert request.await_count == 8
    assert clock.waits == [2, 4, 8, 16, 32, 60, 60]
    assert "PRIVATE_PROVIDER_CONTENT" not in caplog.text
    assert "source=exponential-backoff" in caplog.text


@pytest.mark.asyncio
async def test_server_wait_is_not_capped_or_shortened(clock, monkeypatch):
    """Positive jitter never retries early, even when a server delay exceeds the fallback cap."""
    monkeypatch.setattr(llm_retry.random, "uniform", lambda low, high: high)
    request = AsyncMock(side_effect=[failure(headers={"retry-after": "90"}), "ok"])
    assert await ModelRequests(RetryPolicy()).run(request) == "ok"
    assert clock.waits == [91]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_transient_statuses_retry(clock, status):
    """Transient HTTP errors retry the request and return its eventual response."""
    request = AsyncMock(side_effect=[failure(status), "ok"])
    assert await ModelRequests(RetryPolicy()).run(request) == "ok"
    assert request.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    failure(400), failure(401), failure(403), failure(404),
    failure(headers={"x-should-retry": "false"}),
    failure(code="insufficient_quota"), failure(code="billing_hard_limit_reached"),
    ValueError("not a transient failure"),
])
async def test_terminal_failures_do_not_retry(clock, error):
    """Authentication, invalid requests, billing failures, and programmer errors fail once."""
    request = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await ModelRequests(RetryPolicy()).run(request)
    assert request.await_count == 1
    assert clock.waits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [APIConnectionError, APITimeoutError])
async def test_connection_failures_retry(clock, cls):
    """Disabling SDK retries does not remove its transient connection/timeout recovery."""
    error = cls(request=httpx.Request("POST", "https://model.test"))
    request = AsyncMock(side_effect=[error, "ok"])
    assert await ModelRequests(RetryPolicy()).run(request) == "ok"


@pytest.mark.asyncio
async def test_elapsed_budget_does_not_shorten_server_delay(clock):
    """An excessive server delay ends the request and remains visible to other sessions."""
    retry = ModelRequests(RetryPolicy(max_elapsed_seconds=5))
    first = AsyncMock(side_effect=failure(headers={"retry-after": "10"}))
    with pytest.raises(RateLimitError):
        await retry.run(first)
    second = AsyncMock(return_value="must not send early")
    with pytest.raises(RetryBudgetExceeded):
        await retry.run(second)
    second.assert_not_called()
    assert clock.waits == []


@pytest.mark.asyncio
async def test_exhausted_attempts_still_cool_down_other_sessions(clock):
    """A new session honors the cooldown discovered by a failed request."""
    retry = ModelRequests(RetryPolicy(max_attempts=1))
    with pytest.raises(RateLimitError):
        await retry.run(AsyncMock(side_effect=failure(headers={"retry-after": "10"})))
    assert await retry.run(AsyncMock(return_value="other session")) == "other session"
    assert clock.waits == [10]


@pytest.mark.asyncio
async def test_shared_cooldown_can_be_extended_during_wait(clock, monkeypatch):
    """A waiter rechecks the shared deadline after another request extends it."""
    retry = ModelRequests(RetryPolicy())
    retry.not_before = clock.now + 10
    original_wait = retry._wait

    async def wait(seconds, cancel):
        """Inject one additional throttle while the first cooldown is being observed."""
        await original_wait(seconds, cancel)
        if len(clock.waits) == 1:
            retry.not_before += 5
        return True

    monkeypatch.setattr(retry, "_wait", wait)
    assert await retry.run(AsyncMock(return_value="ok")) == "ok"
    assert clock.waits == [10, 5]


@pytest.mark.asyncio
async def test_delayed_wakeup_cannot_start_retry_after_budget(clock, monkeypatch):
    """A paused process must recheck elapsed time after waking from a cooldown."""
    retry = ModelRequests(RetryPolicy(max_elapsed_seconds=3))

    async def delayed_wait(seconds, cancel):
        """Simulate the event loop resuming after the deadline."""
        clock.now += 5
        return True

    monkeypatch.setattr(retry, "_wait", delayed_wait)
    request = AsyncMock(side_effect=failure())
    with pytest.raises(RateLimitError):
        await retry.run(request)
    assert request.await_count == 1


@pytest.mark.asyncio
async def test_cancellation_before_request():
    """Already-cancelled work never contacts the provider."""
    cancel = asyncio.Event()
    cancel.set()
    request = AsyncMock()
    assert await ModelRequests(RetryPolicy()).run(request, cancel) is None
    request.assert_not_called()


@pytest.mark.asyncio
async def test_cancellation_during_backoff():
    """Cooperative cancellation wakes a long wait without sending another request."""
    cancel = asyncio.Event()
    request = AsyncMock(side_effect=failure(headers={"retry-after": "60"}))
    task = asyncio.create_task(ModelRequests(RetryPolicy()).run(request, cancel))
    await asyncio.sleep(0)
    cancel.set()
    assert await asyncio.wait_for(task, 1) is None
    assert request.await_count == 1


@pytest.mark.asyncio
async def test_wait_timeout_and_task_cancellation():
    """Ordinary waits expire and task cancellation propagates without leaking a sleeper."""
    retry = ModelRequests(RetryPolicy())
    assert await retry._wait(0.001, None)
    assert await retry._wait(0.001, asyncio.Event())
    task = asyncio.create_task(retry.run(AsyncMock(side_effect=failure(headers={"retry-after": "60"}))))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.fixture
async def orchestrator(tmp_path, monkeypatch):
    """Create a real orchestrator with isolated storage and no live credentials."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    orch = SessionOrchestrator({"model": {"name": "test-model"},
        "session": {"debounce_seconds": 0}, "binaries": {"store": str(tmp_path / "binaries")}}, tmp_path / "sessions")
    assert orch.llm.max_retries == 0
    yield orch
    await orch.llm.close()


def completed(*, text="done", tool=False, stream=True):
    """Return a synthetic response accepted by the real SDK, optionally with a tool call."""
    message = {"role": "assistant", "content": text}
    if tool:
        message["tool_calls"] = [{"id": "call-1", "type": "function", "index": 0,
                                  "function": {"name": "send", "arguments": "{}"}}]
    choice = {"index": 0, "finish_reason": "tool_calls" if tool else "stop",
              "delta" if stream else "message": message}
    data = {"id": "completion-1", "model": "test-model", "created": 1,
            "object": "chat.completion.chunk" if stream else "chat.completion", "choices": [choice]}
    if stream:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n")
    return httpx.Response(200, json=data)


def throttled(*, headers=None):
    """Return the reported nested-proxy 429 shape without private request data."""
    return httpx.Response(429, headers=headers, json={"error": {
        "message": 'litellm.RateLimitError: {"error":{"message":"Reached overall message rate limit. '
                   'Your limit will reset in 10 seconds."}}',
        "type": "throttling_error", "code": "429"}})


async def bind_transport(orch, *responses):
    """Replace network I/O with recorded HTTP responses while exercising actual SDK retries."""
    pending = list(responses)
    requests = []

    async def handle(request):
        """Capture the outgoing JSON and return the next predetermined response."""
        requests.append(json.loads(request.content))
        return pending.pop(0)

    await orch.llm.close()
    orch.llm = AsyncOpenAI(api_key="test-only", base_url="https://model.test/v1", max_retries=0,
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    return requests


@pytest.mark.asyncio
async def test_real_sdk_recovers_reported_proxy_429(orchestrator, clock):
    """The user's headerless 429 waits ten seconds and retries identical request content."""
    requests = await bind_transport(orchestrator, throttled(), completed())
    assert await orchestrator.handle_event(Event(text="hello")) == "done"
    assert clock.waits == [10]
    assert requests[0] == requests[1]
    history = orchestrator.session.load("main")
    assert sum(item.get("content") == "hello" for item in history) == 1
    assert sum(item.get("content") == "done" for item in history) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_retry_after_tools_does_not_reexecute_or_lose_results(orchestrator, clock, recover):
    """Only the next completion retries; executed tools persist even on terminal failure."""
    orchestrator.model_requests = ModelRequests(RetryPolicy(max_attempts=2))
    tool = AsyncMock(return_value=({"type": "function_call_output", "call_id": "call-1", "output": "sent"}, []))
    orchestrator._execute_tool_call = tool
    requests = await bind_transport(orchestrator, completed(tool=True), throttled(),
                                    completed() if recover else throttled())
    assert await orchestrator.handle_event(Event(text="send once")) == ("done" if recover else None)
    assert tool.await_count == 1
    assert requests[1] == requests[2]
    history = orchestrator.session.load("main")
    assert sum(item.get("type") == "function_call" for item in history) == 1
    assert [item["output"] for item in history if item.get("type") == "function_call_output"] == ["sent"]
    assert sum(item.get("content") == "send once" for item in history) == 1
    if not recover:
        event = json.loads(history[-1]["content"])
        assert event["event"] == "model_error"
        assert event["error_type"] == "RateLimitError"
        assert "litellm" not in json.dumps(history)


@pytest.mark.asyncio
async def test_compaction_retries_without_repeating_tools(orchestrator, clock):
    """Non-streaming summary calls use the same backoff and tool-once boundary."""
    tool = AsyncMock(return_value=({"type": "function_call_output", "call_id": "call-1", "output": "saved"}, []))
    orchestrator._execute_tool_call = tool
    requests = await bind_transport(orchestrator, completed(tool=True, stream=False), throttled(),
                                    completed(text="summary", stream=False))
    assert await orchestrator._summarize([{"role": "user", "content": "history"}], []) == "summary"
    assert clock.waits == [10]
    assert requests[1] == requests[2]
    assert tool.await_count == 1


@pytest.mark.asyncio
async def test_cancelled_backoff_preserves_turn(orchestrator, clock, monkeypatch):
    """Cancelling during a retry preserves the inbound event without marking it completed."""
    cancel = asyncio.Event()

    async def stop(seconds, signal):
        """Deliver cooperative cancellation when the provider asks for backoff."""
        cancel.set()
        return False

    monkeypatch.setattr(orchestrator.model_requests, "_wait", stop)
    requests = await bind_transport(orchestrator, throttled())
    assert await orchestrator._process_events("main", [Event(text="keep me")], cancel) is None
    assert len(requests) == 1
    assert any(item.get("content") == "keep me" for item in orchestrator.session.load("main"))


@pytest.mark.asyncio
async def test_cancelled_compaction_keeps_history_version(orchestrator):
    """A cancelled summary cannot replace the original session with an empty summary."""
    orchestrator.compaction_trigger_tokens = 2
    orchestrator.recency_tokens = 1
    history = [{"role": "user", "content": "old"},
               {"type": "comment", "kind": "usage", "input_tokens": 2},
               {"role": "user", "content": "recent"},
               {"type": "comment", "kind": "usage", "input_tokens": 4}]
    orchestrator.session.append("main", history)
    cancel = asyncio.Event()
    cancel.set()
    await orchestrator._compact_session_if_needed("main", cancel)
    assert orchestrator.session.load("main") == history


@pytest.mark.asyncio
async def test_partial_stream_is_closed_preserved_and_never_retried(orchestrator, clock):
    """A broken stream cannot replay or execute incomplete tool-call fragments."""
    first = completed(text="partial", tool=True).content.split(b"data: [DONE]")[0]

    class BrokenStream(httpx.AsyncByteStream):
        """Yield one model chunk and simulate a transport failure during generation."""
        closed = False

        async def __aiter__(self):
            """Yield partial output, then fail rather than finish the completion."""
            yield first
            raise httpx.ReadError("stream disconnected")

        async def aclose(self):
            """Record that the failed HTTP response was released."""
            self.closed = True

    stream = BrokenStream()
    requests = await bind_transport(orchestrator,
        httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    tool = AsyncMock()
    orchestrator._execute_tool_call = tool
    assert await orchestrator.handle_event(Event(text="hello")) is None
    assert len(requests) == 1
    assert clock.waits == []
    assert stream.closed
    tool.assert_not_called()
    history = orchestrator.session.load("main")
    event = json.loads(history[-1]["content"])
    assert event["event"] == "model_error"
    assert event["error_type"] == "ReadError"
    assert event["partial"]["content"] == "partial"
    assert event["partial"]["tool_calls"][0]["function"]["name"] == "send"
    assert not any(item.get("type") == "function_call" for item in history)
