"""Mid-turn context-window recovery without repeating tools or losing the audit trail."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
from openai import APIStatusError, AsyncOpenAI, BadRequestError
import pytest

from session_manager.context import context_limit_error, recovery_media, recovery_source, split_for_recovery
from session_manager.event import Event
from session_manager.orchestrator import _to_chat_messages
from session_manager.session import SessionManager
from test_llm_retry import completed, orchestrator


def overflow():
    """Return the reported LiteLLM error shape, without a machine-readable context code."""
    return httpx.Response(400, json={"error": {
        "message": "litellm.APIError: Your input exceeds the context window of this model. Please adjust your input and try again.",
        "type": None, "param": None, "code": "400"}})


def sdk_error(status=400, message="Invalid request", code=None):
    """Construct an SDK status error for narrowly testing overflow recognition."""
    return APIStatusError(message, response=httpx.Response(status, request=httpx.Request("POST", "https://model.test")),
                          body={"code": code})


@pytest.mark.parametrize("error, expected", [
    (sdk_error(code="context_length_exceeded"), True),
    (sdk_error(message="Your input exceeds the context window of this model."), True),
    (sdk_error(message="Maximum context length is 10000 tokens"), True),
    (sdk_error(413, code="prompt_too_long"), True),
    (sdk_error(message="Invalid tool arguments"), False),
    (sdk_error(429, code="context_length_exceeded"), False),
    (sdk_error(413, message="Request Entity Too Large"), False),
    (sdk_error(code={"not": "a string"}), False),
    (ValueError("input exceeds the context window"), False),
])
def test_context_error_detection(error, expected):
    """Only known context errors change the request; unrelated failures keep their existing policy."""
    assert context_limit_error(error) is expected


def test_safe_split_keeps_event_pairs_and_complete_tool_batches():
    """A recent tail cannot begin with orphaned tool results or detached event text."""
    head = [{"role": "user", "content": "old" * 1000}]
    event = [{"role": "developer", "content": "metadata"}, {"role": "user", "content": "request"}]
    batch = [{"type": "comment", "kind": "usage", "input_tokens": 100},
             {"type": "reasoning", "content": "private reasoning"},
             {"role": "assistant", "content": "Working"},
             {"type": "function_call", "call_id": "a", "name": "one", "arguments": "{}"},
             {"type": "function_call", "call_id": "b", "name": "two", "arguments": "{}"},
             {"type": "function_call_output", "call_id": "a", "output": "first"},
             {"type": "function_call_output", "call_id": "b", "output": "second"}]
    fold, keep = split_for_recovery(head + event + batch, len(json.dumps(batch)) + 5)
    assert fold == head + event and keep == batch
    fold, keep = split_for_recovery(head + batch, 10)
    assert fold == head + batch and not keep
    assert split_for_recovery([], 100) == ([], [])
    assert split_for_recovery([{"type": "comment"}], 100)[1] == [{"type": "comment"}]
    assert "private reasoning" not in recovery_source(batch)
    assert '"call_id": "a"' in recovery_source(batch)


def test_media_retention_counts_omissions_without_mutating_input():
    """Older image batches are omitted explicitly; stronger recovery drops all images."""
    image1 = {"type": "image_url", "image_url": {"url": "first"}}
    image2 = {"type": "image_url", "image_url": {"url": "second"}}
    messages = [{"role": "user", "content": "text"}, {"role": "user", "content": [image1]},
                {"role": "user", "content": [{"type": "text", "text": "x"}, image2]}]
    assert recovery_media(messages) == ([image2], 1)
    assert recovery_media(messages, discard=True) == ([], 2)
    assert recovery_media([]) == ([], 0)
    assert len(messages[-1]["content"]) == 2


async def transport(orch, handle):
    """Install an actual SDK client with a synthetic HTTP transport, never a live endpoint."""
    await orch.llm.close()
    orch.llm = AsyncOpenAI(api_key="test", base_url="https://model.test/v1", max_retries=0,
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))


@pytest.mark.asyncio
@pytest.mark.parametrize("large_latest_result", [False, True])
async def test_overflow_mid_tool_round_compacts_and_resumes_once(orchestrator, large_latest_result):
    """Completed actions run once; both historical growth and a giant latest result are recoverable."""
    orch = orchestrator
    orch.openai_tools = [{"name": "synthetic_action", "description": "Run a synthetic action",
                          "parameters": {"type": "object", "properties": {}}}]
    if not large_latest_result:
        orch.session.append("main", [{"role": "user", "content": "OLDER_CONTEXT " * 8000},
                                     {"role": "assistant", "content": "old answer"}])
    result_text = "RESULT_DATA " * 7000 if large_latest_result else "Action completed, receipt R123"
    tool = AsyncMock(return_value=({"type": "function_call_output", "call_id": "call-1", "output": result_text}, []))
    orch._execute_tool_call = tool
    main, summaries = [], []

    async def handle(request):
        """Reject only the oversized continuation, then accept its smaller checkpoint."""
        body = json.loads(request.content)
        if not body.get("stream"):
            assert "tools" not in body and "tool_choice" not in body
            summaries.append(body)
            return completed(stream=False, text="The send action completed. Continue the outstanding request; do not resend.")
        main.append(body)
        if len(main) == 1:
            return completed(tool=True)
        if len(main) == 2:
            return overflow()
        return completed(text="finished")

    await transport(orch, handle)
    assert await orch.handle_event(Event(text="Do this once", metadata={"room_id": "!synthetic"})) == "finished"
    assert tool.await_count == 1 and len(main) == 3 and summaries
    assert main[0]["tools"] == main[1]["tools"] == main[2]["tools"]
    assert main[-1]["tools"][0]["function"]["name"] == "synthetic_action"
    assert len(json.dumps(main[-1])) < len(json.dumps(main[-2]))
    assert "!synthetic" in json.dumps(main[-1])
    old_path = orch.session.store_dir / "main.0.jsonl"
    original = [json.loads(line) for line in old_path.read_text().splitlines()]
    assert sum(item.get("type") == "function_call" for item in original) == 1
    assert sum(item.get("type") == "function_call_output" for item in original) == 1
    assert any(item.get("output") == result_text for item in original)
    assert sum(item.get("content") == "Do this once" for item in original) == 1
    active = orch.session.load("main")
    assert active[-1]["content"] == "finished"
    assert any('"event": "context_recovered"' in item.get("content", "") for item in active)
    if not large_latest_result:
        calls = [call["id"] for message in main[-1]["messages"] for call in message.get("tool_calls", [])]
        results = [message["tool_call_id"] for message in main[-1]["messages"] if message["role"] == "tool"]
        assert calls == results == ["call-1"]
    fragments = [json.loads(body["messages"][-1]["content"])["transcript_fragment"] for body in summaries]
    fold, _keep = split_for_recovery(original, 64_000)
    assert "".join(fragments) == recovery_source(fold)


@pytest.mark.asyncio
async def test_proactive_compaction_checks_usage_between_tool_rounds(orchestrator):
    """High reported prompt usage can trigger compaction before the next request fails."""
    orch = orchestrator
    orch.session.append("main", [{"role": "user", "content": "OLD_CONTEXT " * 8000}])
    tool = AsyncMock(return_value=({"type": "function_call_output", "call_id": "call-1", "output": "done"}, []))
    orch._execute_tool_call = tool
    main = []

    async def handle(request):
        """Report over-threshold usage on the completed tool-call generation."""
        body = json.loads(request.content)
        if not body.get("stream"):
            return completed(stream=False, text="Older context checkpoint")
        main.append(body)
        if len(main) == 1:
            response = completed(tool=True)
            usage = {"id": "usage", "object": "chat.completion.chunk", "created": 1, "model": "test",
                     "choices": [], "usage": {"prompt_tokens": 100001, "completion_tokens": 10, "total_tokens": 100011}}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=response.text.replace("data: [DONE]", f"data: {json.dumps(usage)}\n\ndata: [DONE]"))
        return completed(text="finished")

    await transport(orch, handle)
    assert await orch.handle_event(Event(text="continue")) == "finished"
    assert len(main) == 2 and tool.await_count == 1
    assert "OLD_CONTEXT" not in json.dumps(main[-1])
    assert '"reason": "input_budget"' in json.dumps(orch.session.load("main")[1:]).replace('\\"', '"')


@pytest.mark.asyncio
async def test_failed_recovery_preserves_complete_current_turn(orchestrator):
    """A failed summary does not publish a new version or duplicate/discard completed tools."""
    orch = orchestrator
    output = "COMPLETED_ACTION " * 6000
    tool = AsyncMock(return_value=({"type": "function_call_output", "call_id": "call-1", "output": output}, []))
    orch._execute_tool_call = tool
    requests = []

    async def handle(request):
        """Reject the continuation and return an unusable summary."""
        body = json.loads(request.content)
        requests.append(body)
        if not body.get("stream"):
            return completed(stream=False, text="")
        return completed(tool=True) if len(requests) == 1 else overflow()

    await transport(orch, handle)
    assert await orch.handle_event(Event(text="original request")) is None
    assert tool.await_count == 1
    history = orch.session.load("main")
    assert sum(item.get("content") == "original request" for item in history) == 1
    assert sum(item.get("output") == output for item in history) == 1
    assert json.loads(history[-1]["content"])["context_recovery"] == "exhausted_or_unavailable"
    assert not (orch.session.store_dir / "main.1.jsonl").exists()


@pytest.mark.asyncio
async def test_cancellation_during_recovery_does_not_retry_or_publish(orchestrator):
    """A cancelled summary leaves the original input/results available to the next event."""
    orch = orchestrator
    orch.session.append("main", [{"role": "user", "content": "OLD " * 20000}])
    main = []

    async def handle(request):
        """Cancel after entering summary generation, before publication."""
        body = json.loads(request.content)
        if not body.get("stream"):
            orch._get_session("main").cancel.set()
            return completed(stream=False, text="must not publish")
        main.append(body)
        return overflow()

    await transport(orch, handle)
    assert await orch.handle_event(Event(text="pending")) is None
    assert len(main) == 1
    assert any(item.get("content") == "pending" for item in orch.session.load("main"))
    assert not (orch.session.store_dir / "main.1.jsonl").exists()


@pytest.mark.asyncio
async def test_unrelated_bad_request_does_not_compact(orchestrator):
    """Invalid arguments remain a failure, not an excuse to discard or summarize history."""
    orch = orchestrator
    orch._summarize_for_recovery = AsyncMock()

    async def handle(request):
        """Return a non-context validation error."""
        return httpx.Response(400, json={"error": {"message": "Invalid tool arguments", "code": "bad_request"}})

    await transport(orch, handle)
    assert await orch.handle_event(Event(text="request")) is None
    orch._summarize_for_recovery.assert_not_called()
    assert json.loads(orch.session.load("main")[-1]["content"])["error_type"] == "BadRequestError"


@pytest.mark.asyncio
async def test_recovery_never_runs_summarizer_tool_calls(orchestrator):
    """Even a provider that returns unsolicited summary tool calls cannot execute actions."""
    orch = orchestrator
    orch._execute_tool_call = AsyncMock()

    async def handle(request):
        """Ignore the requested tool-free response to exercise the guard."""
        assert "tools" not in json.loads(request.content)
        return completed(stream=False, tool=True)

    await transport(orch, handle)
    with pytest.raises(RuntimeError, match="cannot call tools"):
        await orch._summarize_for_recovery([{"role": "user", "content": "context"}], asyncio.Event())
    orch._execute_tool_call.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "content_filter"])
async def test_truncated_checkpoints_are_not_accepted(orchestrator, finish):
    """A partial summary cannot be mistaken for a complete checkpoint of the archived input."""
    async def handle(request):
        """Return nonempty but incomplete summary output."""
        payload = completed(stream=False, text="partial checkpoint").json()
        payload["choices"][0]["finish_reason"] = finish
        return httpx.Response(200, json=payload)

    await transport(orchestrator, handle)
    with pytest.raises(RuntimeError, match="truncated or filtered"):
        await orchestrator._summarize_for_recovery([{"role": "user", "content": "source"}], asyncio.Event())


@pytest.mark.asyncio
async def test_summary_chunks_shrink_when_provider_rejects_them(orchestrator):
    """The compaction request itself adapts instead of retrying identical oversized input."""
    orch = orchestrator
    requests, accepted = [], []
    orch.call_config = {"max_completion_tokens": 2048, "stream_options": {"include_usage": True},
                        "tool_choice": "required", "extra_body": {"tools": [{"name": "unsafe"}], "reasoning_effort": "low"}}

    async def handle(request):
        """Accept only small enough synthetic summary fragments."""
        body = json.loads(request.content)
        requests.append(body)
        assert body["max_completion_tokens"] == 2048 and "max_tokens" not in body
        assert "tools" not in body and "tool_choice" not in body and "stream_options" not in body
        fragment = json.loads(body["messages"][-1]["content"])["transcript_fragment"]
        if len(fragment) > 20000:
            return overflow()
        accepted.append(fragment)
        return completed(stream=False, text="checkpoint")

    await transport(orch, handle)
    items = [{"role": "user", "content": "source" * 14000}]
    assert await orch._summarize_for_recovery(items, asyncio.Event()) == "checkpoint"
    assert "".join(accepted) == recovery_source(items)
    assert len(requests) > len(accepted)


@pytest.mark.asyncio
async def test_image_only_overflow_recovery_is_explicit(orchestrator):
    """When there is no text to compact, image removal is visible rather than silently losing media."""
    orch = orchestrator
    items = [{"role": "user", "content": "Inspect the attachment"}]
    messages = [{"role": "system", "content": "instructions"}, *_to_chat_messages(items),
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:" + "A" * 10000}}]}]
    assert await orch._recovery_context(items, messages, 1, False, asyncio.Event()) is None
    checkpoint, live = await orch._recovery_context(items, messages, 1, True, asyncio.Event())
    assert json.loads(checkpoint[-1]["content"])["images_omitted"] == 1
    assert "data:" not in json.dumps(checkpoint) and "data:" not in json.dumps(live)
    assert checkpoint[0] == items[0]


@pytest.mark.asyncio
async def test_provider_rejection_overrides_recent_tail_size(orchestrator):
    """A small model window can require summarizing even the normally retained tail."""
    items = [{"role": "user", "content": "recent data " * 2500}]
    messages = [{"role": "system", "content": "instructions"}, *_to_chat_messages(items)]
    orchestrator._summarize_for_recovery = AsyncMock(return_value="Continue the recent request.")
    cancel = asyncio.Event()
    checkpoint, live = await orchestrator._recovery_context(items, messages, 1, True, cancel)
    orchestrator._summarize_for_recovery.assert_awaited_once_with(items, cancel)
    assert json.loads(checkpoint[-1]["content"])["summarized_items"] == 1
    assert len(json.dumps(live)) < len(json.dumps(messages))


@pytest.mark.asyncio
async def test_oversized_routing_metadata_is_explicitly_omitted(orchestrator):
    """Unbounded routing metadata cannot silently consume the recovered context budget."""
    items = [{"role": "user", "content": "large context " * 6000}]
    messages = [{"role": "system", "content": "instructions"}, *_to_chat_messages(items)]
    orchestrator._summarize_for_recovery = AsyncMock(return_value="Continue the request.")
    checkpoint, _live = await orchestrator._recovery_context(
        items, messages, 1, True, asyncio.Event(), origin_context=[{"extra": "x" * 16000}])
    event = json.loads(checkpoint[-1]["content"])
    assert event["origin_context"] == [] and event["origin_context_omitted"] is True


@pytest.mark.asyncio
async def test_latest_image_batch_survives_when_older_images_can_be_removed(orchestrator):
    """Recovery does not discard the most recent visual input unnecessarily."""
    items = [{"role": "user", "content": "compare these"}]
    old = {"type": "image_url", "image_url": {"url": "old:" + "A" * 10000}}
    recent = {"type": "image_url", "image_url": {"url": "recent:" + "B" * 10000}}
    messages = [{"role": "system", "content": "instructions"}, *_to_chat_messages(items),
                {"role": "user", "content": [old]}, {"role": "user", "content": [recent]}]
    checkpoint, live = await orchestrator._recovery_context(items, messages, 1, True, asyncio.Event())
    assert json.loads(checkpoint[-1]["content"])["images_omitted"] == 1
    assert "old:" not in json.dumps(live)
    assert live[-1]["content"][-1] == recent
    assert "recent:" not in json.dumps(checkpoint)


@pytest.mark.asyncio
async def test_repeated_context_failures_have_a_finite_recovery_budget(orchestrator):
    """Repeated rejections stop after two successful reductions instead of looping indefinitely."""
    orch = orchestrator
    orch.session.append("main", [{"role": "user", "content": "OLD_HISTORY " * 9000}])
    main = []

    async def handle(request):
        """Keep rejecting reduced main requests while returning progressively smaller summaries."""
        body = json.loads(request.content)
        if not body.get("stream"):
            return completed(stream=False, text="S" * 15990 if len(main) == 1 else "small checkpoint")
        main.append(body)
        return overflow()

    await transport(orch, handle)
    assert await orch.handle_event(Event(text="pending")) is None
    assert len(main) == 3
    assert len(json.dumps(main[0])) > len(json.dumps(main[1])) > len(json.dumps(main[2]))
    assert json.loads(orch.session.load("main")[-1]["content"])["context_recovery"] == "exhausted_or_unavailable"
    assert (orch.session.store_dir / "main.0.jsonl").exists()


@pytest.mark.asyncio
async def test_summary_request_budget_and_minimum_fragment_size(orchestrator):
    """An uncooperative provider cannot cause unbounded fragment-splitting retries."""
    orch = orchestrator
    count = 0

    async def handle(request):
        """Reject every summary, including the minimum-size fragment."""
        nonlocal count
        count += 1
        return overflow()

    await transport(orch, handle)
    with pytest.raises(BadRequestError):
        await orch._summarize_for_recovery([{"role": "user", "content": "x" * 70000}], asyncio.Event())
    assert count == 7

    async def accept(request):
        """A very large synthetic transcript still stops at the fixed summary-call budget."""
        return completed(stream=False, text="checkpoint")

    await transport(orch, accept)
    with pytest.raises(RuntimeError, match="32 summary requests"):
        await orch._summarize_for_recovery([{"role": "user", "content": "x" * 2_100_000}], asyncio.Event())


@pytest.mark.asyncio
async def test_empty_source_cancelled_source_and_nonreducing_summary(orchestrator):
    """Reasoning-only input needs no model call; cancellation and larger summaries cannot publish."""
    orch = orchestrator
    assert "reasoning/usage" in await orch._summarize_for_recovery([{"type": "reasoning", "content": "private"}], asyncio.Event())
    cancel = asyncio.Event()
    cancel.set()
    assert await orch._summarize_for_recovery([{"role": "user", "content": "source"}], cancel) is None
    items = [{"role": "user", "content": "small"}]
    messages = [{"role": "system", "content": "instructions"}, *_to_chat_messages(items)]
    orch._summarize_for_recovery = AsyncMock(return_value="larger" * 1000)
    assert await orch._recovery_context(items, messages, 1, True, asyncio.Event()) is None
    assert await orch._recovery_context(items, messages, 5, True, asyncio.Event()) is None


def test_serialization_failure_cannot_publish_partial_compaction(tmp_path):
    """Even a failure after some new lines were written leaves the previous version intact."""
    session = SessionManager(tmp_path)
    session.append("main", [{"role": "user", "content": "original"}])
    with pytest.raises(TypeError):
        session.bump_version("main", [{"role": "user", "content": "first"}, {"invalid": object()}])
    assert session.load("main") == [{"role": "user", "content": "original"}]
    assert [path.name for path in tmp_path.iterdir()] == ["main.0.jsonl"]


def test_compaction_file_publication_is_atomic(tmp_path, monkeypatch):
    """A failed publish leaves the prior version active and removes its temporary file."""
    import session_manager.session as module

    session = SessionManager(tmp_path)
    original = [{"role": "user", "content": "keep this"}]
    session.append("main", original)

    def fail(source, destination):
        """Simulate a filesystem failure at the atomic publication boundary."""
        raise OSError("publish failed")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        session.bump_version("main", [{"role": "user", "content": "summary"}])
    assert session.load("main") == original
    assert [path.name for path in tmp_path.iterdir()] == ["main.0.jsonl"]


def test_compaction_file_creation_failure_leaves_history_intact(tmp_path, monkeypatch):
    """A failure before the temporary file exists does not mask the original error."""
    import session_manager.session as module

    session = SessionManager(tmp_path)
    original = [{"role": "user", "content": "keep this"}]
    session.append("main", original)

    def fail(**kwargs):
        """Simulate a filesystem that cannot create a new checkpoint."""
        raise OSError("cannot create checkpoint")

    monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", fail)
    with pytest.raises(OSError, match="cannot create checkpoint"):
        session.bump_version("main", [{"role": "user", "content": "summary"}])
    assert session.load("main") == original
    assert [path.name for path in tmp_path.iterdir()] == ["main.0.jsonl"]
