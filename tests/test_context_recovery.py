"""One token-based compaction path before turns and after completed tool batches."""

import json
from unittest.mock import AsyncMock

import httpx
from openai import APIStatusError, AsyncOpenAI
import pytest

from session_manager.context import context_limit_error, paired_cutoff, retained_media
from session_manager.event import Event
from session_manager.session import SessionManager
from test_llm_retry import completed, orchestrator


def overflow():
    """Return the reported nested LiteLLM context rejection."""
    return httpx.Response(400, json={"error": {
        "message": "litellm.APIError: Your input exceeds the context window of this model. Please adjust your input and try again.",
        "type": None, "param": None, "code": "400"}})


def sdk_error(status=400, message="Invalid request", code=None):
    """Construct an SDK status error without accessing a live model."""
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
    """Unrelated validation failures cannot trigger compaction."""
    assert context_limit_error(error) is expected


def usage(tokens, iteration=0):
    """Build persisted provider usage at a specified tool iteration."""
    return {"type": "comment", "kind": "usage", "input_tokens": tokens, "iteration": iteration}


def batch(call_id):
    """Build one complete persisted tool exchange."""
    return [{"type": "function_call", "call_id": call_id, "name": "send", "arguments": "{}"},
            {"type": "function_call_output", "call_id": call_id, "output": "completed"}]


def test_recency_counts_iterations_and_preserves_paired_boundaries():
    """A single turn supplies usable recency measurements at each model/tool round."""
    items = [{"role": "user", "content": "old"}, usage(100), *batch("a"),
             usage(400, 1), *batch("b"), usage(900, 2), *batch("c")]
    assert SessionManager.recency_split(items, 300) == 5
    assert items[5:7] == batch("b")
    partial = [{"role": "user", "content": "old"}, *batch("a")[:1], usage(100),
               batch("a")[1], usage(500, 1), *batch("b")]
    assert SessionManager.recency_split(partial, 200) == 1


def test_recency_handles_missing_baseline_and_decreasing_prompt_usage():
    """Legacy prefixes can compact; a drop in transient input cannot cancel later growth."""
    items = [{"role": "user", "content": "legacy history"}, usage(1000), *batch("a")]
    assert SessionManager.recency_split(items, 800) == 2
    items = [{"role": "user", "content": "old"}, usage(100), *batch("a"), usage(600, 1),
             *batch("b"), usage(200), *batch("c"), usage(400, 1)]
    assert SessionManager.recency_split(items, 400) == 2


def test_cutoffs_keep_events_reasoning_and_tool_batches_together():
    """Candidate cutoffs move backwards, never dropping an incomplete exchange."""
    event = [{"role": "developer", "content": "metadata"}, {"role": "user", "content": "request"}]
    items = [*event, {"type": "reasoning", "content": "thinking"},
             {"role": "assistant", "content": "working"},
             batch("a")[0], batch("b")[0], batch("a")[1], batch("b")[1]]
    assert paired_cutoff(items, 1) == 0
    assert paired_cutoff(items, 3) == 2
    assert paired_cutoff(items, 7) == 2
    assert paired_cutoff(items, 8) == 8
    assert paired_cutoff([], 100) == 0
    assert paired_cutoff([batch("a")[0]], 1) == 0


def test_media_tracks_retained_tool_results():
    """Only images belonging to folded tool batches leave live context."""
    image = {"type": "image_url", "image_url": {"url": "data:synthetic", "detail": "high"}}
    batches = {key: {"role": "user", "content": [image]} for key in ("old", "recent")}
    kept, omitted = retained_media(batches, {"recent"})
    assert kept == {"recent": batches["recent"]} and omitted == 1
    assert retained_media(kept, set()) == ({}, 1)
    assert retained_media({}, set()) == ({}, 0)
    assert len(batches) == 2


async def transport(orch, handle):
    """Install an actual SDK with synthetic HTTP responses, never a live endpoint."""
    await orch.llm.close()
    orch.llm = AsyncOpenAI(api_key="test", base_url="https://model.test/v1", max_retries=0,
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))


def tool_response(tokens, call_id="call-1"):
    """Return a streaming tool call with an explicit provider prompt measurement."""
    response = completed(tool=True)
    report = {"id": "usage", "object": "chat.completion.chunk", "created": 1, "model": "test",
              "choices": [], "usage": {"prompt_tokens": tokens, "completion_tokens": 10, "total_tokens": tokens + 10}}
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
        text=response.text.replace("call-1", call_id).replace("data: [DONE]", f"data: {json.dumps(report)}\n\ndata: [DONE]"))


@pytest.mark.asyncio
async def test_normal_summary_runs_after_every_eligible_tool_batch(orchestrator, caplog):
    """Three compactions in one turn use the ordinary prompt/tools, without replaying actions."""
    orch = orchestrator
    orch.compaction_trigger_tokens, orch.recency_tokens = 1000, 200
    orch.session.append("main", [{"role": "user", "content": "OLD_CONTEXT " * 8000}])
    orch.openai_tools = [{"name": "send", "parameters": {"type": "object", "properties": {}}}]
    main, summaries, executed = [], [], []

    async def execute(call_id, name, arguments):
        """Record each completed main or summary action and return a durable result."""
        executed.append(call_id)
        return ({"type": "function_call_output", "call_id": call_id, "output": "receipt " + call_id},
                [{"content": [{"type": "input_image", "image_url": "data:image/" + call_id}]}])

    async def handle(request):
        """Each high-usage tool round gets one normal summary before the next round."""
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "send"
        assert body["messages"][0]["content"] == orch._build_instructions()
        if not body.get("stream"):
            summaries.append(body)
            assert "transcript_fragment" not in json.dumps(body)
            assert "structured summary" in body["messages"][-1]["content"]
            return completed(stream=False, text="Completed work and outstanding request.")
        main.append(body)
        images = [part["image_url"]["url"] for message in body["messages"]
                  if isinstance(message.get("content"), list) for part in message["content"]
                  if part.get("type") == "image_url"]
        assert images == ([f"data:image/action-{len(main) - 1}"] if len(main) > 1 else [])
        if len(main) <= 3:
            assert len(summaries) == len(main) - 1
            return tool_response(1200, f"action-{len(main)}")
        return completed(text="finished")

    orch._execute_tool_call = execute
    await transport(orch, handle)
    with caplog.at_level("INFO", logger="session_manager.orchestrator"):
        assert await orch.handle_event(Event(text="Do this once", metadata={"room_id": "!synthetic"})) == "finished"
    assert executed == ["action-1", "action-2", "action-3"]
    assert len(main) == 4 and len(summaries) == 3
    assert "!synthetic" in json.dumps(main[-1])
    assert "OLD_CONTEXT" not in json.dumps(main[-1])
    assert "summary request 1/20" in caplog.text and "compaction: wrote" in caplog.text
    original = [json.loads(line) for line in (orch.session.store_dir / "main.0.jsonl").read_text().splitlines()]
    assert sum(item.get("content") == "Do this once" for item in original) == 1
    assert sum(item.get("call_id") == "action-1" and item.get("type") == "function_call_output" for item in original) == 1
    assert (orch.session.store_dir / "main.3.jsonl").exists()
    assert orch.session.load("main")[-1]["content"] == "finished"
    assert "data:image/" not in json.dumps(orch.session.load("main"))
    assert any(json.loads(item["content"]).get("images_omitted") == 1
               for item in orch.session.load("main") if item.get("role") == "developer")


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_failure", [None, "empty", "cancel", "overflow"])
async def test_context_rejection_uses_normal_summary_once(orchestrator, summary_failure):
    """A provider rejection bypasses the trigger, not the normal summary machinery."""
    orch = orchestrator
    orch.recency_tokens = 200
    orch.session.append("main", [{"role": "user", "content": "OLDER_CONTEXT " * 8000}])
    tool = AsyncMock(return_value=({"type": "function_call_output", "call_id": "call-1", "output": "completed once"}, []))
    orch._execute_tool_call = tool
    main, summaries = [], []

    async def handle(request):
        """Reject the first continuation; summaries are ordinary full-prefix requests."""
        body = json.loads(request.content)
        if not body.get("stream"):
            summaries.append(body)
            assert "OLDER_CONTEXT" in json.dumps(body)
            if summary_failure == "overflow":
                return overflow()
            if summary_failure == "cancel":
                orch._get_session("main").cancel.set()
            return completed(stream=False, text="" if summary_failure == "empty" else "Earlier context summary")
        main.append(body)
        if len(main) == 1:
            return tool_response(500)
        return overflow() if len(main) == 2 else completed(text="finished")

    await transport(orch, handle)
    result = await orch.handle_event(Event(text="original request"))
    assert result == (None if summary_failure else "finished")
    assert len(summaries) == 1 and tool.await_count == 1
    assert len(main) == (2 if summary_failure else 3)
    history = orch.session.load("main")
    assert sum(item.get("output") == "completed once" for item in history) == 1
    if summary_failure:
        assert not (orch.session.store_dir / "main.1.jsonl").exists()
        assert sum(item.get("content") == "original request" for item in history) == 1
    else:
        calls = [call["id"] for message in main[-1]["messages"] for call in message.get("tool_calls", [])]
        results = [message["tool_call_id"] for message in main[-1]["messages"] if message["role"] == "tool"]
        assert calls == results == ["call-1"]


@pytest.mark.asyncio
async def test_long_turn_uses_iteration_cutoff_instead_of_folding_recent_work(orchestrator):
    """Usage within one turn selects the token window and keeps all newer tool exchanges."""
    orch = orchestrator
    orch.compaction_trigger_tokens, orch.recency_tokens = 800, 300
    orch.session.append("main", [{"role": "user", "content": "old history " * 1000}])
    main, summaries = [], []

    async def execute(call_id, name, arguments):
        """Give each completed action a distinct receipt in the transcript."""
        return {"type": "function_call_output", "call_id": call_id, "output": "receipt-" + call_id}, []

    async def handle(request):
        """Grow prompt usage across iterations until the configured trigger is crossed."""
        body = json.loads(request.content)
        if not body.get("stream"):
            summaries.append(body)
            source = json.dumps(body)
            assert "receipt-action-1" in source
            assert "receipt-action-2" not in source and "receipt-action-3" not in source
            return completed(stream=False, text="Older context and first action summarized.")
        main.append(body)
        return tool_response([100, 400, 900][len(main) - 1], f"action-{len(main)}") if len(main) <= 3 else completed()

    orch._execute_tool_call = execute
    await transport(orch, handle)
    assert await orch.handle_event(Event(text="continue")) == "done"
    assert len(summaries) == 1
    calls = [call["id"] for message in main[-1]["messages"] for call in message.get("tool_calls", [])]
    results = [message["tool_call_id"] for message in main[-1]["messages"] if message["role"] == "tool"]
    assert calls == results == ["action-2", "action-3"]


@pytest.mark.asyncio
async def test_cancel_after_completed_tools_skips_compaction_without_losing_results(orchestrator):
    """Cancellation at the batch boundary retains completed actions without starting a summary."""
    orch = orchestrator
    orch.compaction_trigger_tokens = 100
    orch._summarize = AsyncMock()

    async def execute(call_id, name, arguments):
        """Cancel after this action finishes, as a shutdown could do."""
        orch._get_session("main").cancel.set()
        return {"type": "function_call_output", "call_id": call_id, "output": "completed"}, []

    async def handle(request):
        """Return the only request that should reach the provider."""
        return tool_response(500)

    orch._execute_tool_call = execute
    await transport(orch, handle)
    assert await orch.handle_event(Event(text="do once")) is None
    orch._summarize.assert_not_called()
    assert sum(item.get("output") == "completed" for item in orch.session.load("main")) == 1


@pytest.mark.asyncio
async def test_repeated_context_rejection_stops_after_one_retry(orchestrator):
    """Successful compaction cannot create an unbounded provider-rejection loop."""
    orch = orchestrator
    orch.recency_tokens = 200
    orch.session.append("main", [{"role": "user", "content": "old " * 10000}, usage(500)])
    main, summaries = [], []

    async def handle(request):
        """Accept summaries, but reject every continuation."""
        body = json.loads(request.content)
        if not body.get("stream"):
            summaries.append(body)
            return completed(stream=False, text="summary")
        main.append(body)
        return overflow()

    await transport(orch, handle)
    assert await orch.handle_event(Event(text="pending")) is None
    assert len(main) == 2 and len(summaries) == 1
    assert json.loads(orch.session.load("main")[-1]["content"])["context_recovery"] == "exhausted_or_unavailable"


@pytest.mark.asyncio
async def test_unrelated_bad_request_does_not_compact(orchestrator):
    """A non-context validation error never starts a summary."""
    orch = orchestrator
    orch._summarize = AsyncMock()

    async def handle(request):
        """Return an unrelated provider validation error."""
        return httpx.Response(400, json={"error": {"message": "Invalid tool arguments", "code": "bad_request"}})

    await transport(orch, handle)
    assert await orch.handle_event(Event(text="request")) is None
    orch._summarize.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "content_filter"])
async def test_incomplete_summary_cannot_replace_history(orchestrator, finish):
    """The shared path rejects incomplete summaries instead of publishing partial notes."""
    orch = orchestrator
    orch.recency_tokens = 200
    original = [{"role": "user", "content": "old " * 1000}, usage(500)]
    orch.session.append("main", original)

    async def handle(request):
        """Return a nonempty summary whose finish reason says it is incomplete."""
        payload = completed(stream=False, text="partial checkpoint").json()
        payload["choices"][0]["finish_reason"] = finish
        return httpx.Response(200, json=payload)

    await transport(orch, handle)
    assert not await orch._compact_session_if_needed("main", forced=True)
    assert orch.session.load("main") == original


@pytest.mark.asyncio
async def test_nonreducing_forced_summary_and_missing_usage_keep_history(orchestrator):
    """A forced retry requires a known cutoff and genuinely reduced message input."""
    orch = orchestrator
    orch.recency_tokens = 200
    orch._summarize = AsyncMock(return_value="oversized summary " * 1000)
    original = [{"role": "user", "content": "old"}]
    orch.session.append("main", original)
    assert not await orch._compact_session_if_needed("main", forced=True)
    orch._summarize.assert_not_called()
    orch.session.append("main", [usage(500)])
    assert not await orch._compact_session_if_needed("main", forced=True)
    assert orch.session.load("main") == original + [usage(500)]


def test_serialization_failure_cannot_publish_partial_compaction(tmp_path):
    """A serialization failure leaves the previous version active and intact."""
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
