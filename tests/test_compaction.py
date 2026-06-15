"""Tests for the orchestrator's compaction routine.

Compaction is triggered when the most recent `usage` comment's
`input_tokens` exceeds `compaction_trigger_tokens`. The orchestrator
asks the agent (via a no-tools LLM call) to produce a structured
summary of the older context, then writes a new versioned session
file with the summary at the top followed by the recency window.
"""

import json
from unittest.mock import MagicMock

import pytest

from session_manager.orchestrator import SessionOrchestrator


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _mock_summary_response(text: str) -> MagicMock:
    """A non-streaming Responses-API response containing a single message."""
    block = MagicMock()
    block.text = text
    item = MagicMock()
    item.type = "message"
    item.content = [block]
    resp = MagicMock()
    resp.output = [item]
    return resp


def _orchestrator(tmp_path, *, trigger: int = 1000, recency: int = 800) -> SessionOrchestrator:
    config = {
        "model": {"name": "test-model"},
        "session": {
            "compaction_trigger_tokens": trigger,
            "recency_tokens": recency,
        },
        "binaries": {"store": str(tmp_path / "binaries"), "retention_days": 30},
    }
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return SessionOrchestrator(config=config, session_dir=sessions_dir)


@pytest.mark.asyncio
async def test_no_compaction_when_no_usage_data(tmp_path):
    """Fresh sessions with no usage comments yet must not compact."""
    orch = _orchestrator(tmp_path)
    orch.session.append("main", [{"role": "user", "content": "hi"}])

    orch.llm = MagicMock()
    orch.llm.responses = MagicMock()
    create_mock = MagicMock()
    orch.llm.responses.create = create_mock

    await orch._compact_session_if_needed("main")

    create_mock.assert_not_called()
    # Still on version 0; no bump.
    assert (tmp_path / "sessions" / "main.0.jsonl").exists()
    assert not (tmp_path / "sessions" / "main.1.jsonl").exists()


@pytest.mark.asyncio
async def test_no_compaction_when_under_trigger(tmp_path):
    orch = _orchestrator(tmp_path, trigger=1000)
    orch.session.append("main", [
        {"role": "user", "content": "hi"},
        {"type": "comment", "kind": "usage", "input_tokens": 500},
    ])

    orch.llm = MagicMock()
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = MagicMock()

    await orch._compact_session_if_needed("main")

    orch.llm.responses.create.assert_not_called()


@pytest.mark.asyncio
async def test_compaction_runs_summary_and_writes_new_version(tmp_path):
    orch = _orchestrator(tmp_path, trigger=1000, recency=200)
    # Construct usage deltas: 100, 300, 800, 1200 ⇒ deltas 200, 500, 400.
    # Walking back, the 400 delta alone reaches recency=200 — split after
    # the usage just before the latest content block.
    orch.session.append("main", [
        {"role": "user", "content": "very old"},                            # 0
        {"type": "comment", "kind": "usage", "input_tokens": 100},          # 1
        {"role": "user", "content": "old"},                                  # 2
        {"type": "comment", "kind": "usage", "input_tokens": 300},          # 3
        {"role": "user", "content": "mid"},                                  # 4
        {"type": "comment", "kind": "usage", "input_tokens": 800},          # 5
        {"role": "user", "content": "new"},                                  # 6
        {"type": "comment", "kind": "usage", "input_tokens": 1200},         # 7
    ])

    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _mock_summary_response("MY STRUCTURED SUMMARY")

    orch.llm = MagicMock()
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = fake_create

    await orch._compact_session_if_needed("main")

    # New version was written.
    new_path = tmp_path / "sessions" / "main.1.jsonl"
    assert new_path.exists()

    new_history = orch.session.load("main")
    # First message is the summary; payload carries the agent's text.
    summary_msg = new_history[0]
    assert summary_msg["role"] == "developer"
    summary_payload = json.loads(summary_msg["content"])
    assert summary_payload["event"] == "context_summary"
    assert summary_payload["content"] == "MY STRUCTURED SUMMARY"

    # Kept tail matches what we expected: messages from index 6 onward,
    # with usage comments stripped (they'd be stale in the new version).
    kept = new_history[1:]
    assert kept[0] == {"role": "user", "content": "new"}
    assert all(not (m.get("type") == "comment" and m.get("kind") == "usage") for m in kept)

    # Summary call carried our instructions and no tools.
    assert "tools" not in captured
    assert captured["model"] == "test-model"
    inputs = captured["input"]
    # Last two items: developer-event + user-content pair carrying the
    # summarize directive (same shape as a normal event arrival).
    prompt_msg = inputs[-2]
    content_msg = inputs[-1]
    assert prompt_msg["role"] == "developer"
    prompt_payload = json.loads(prompt_msg["content"])
    assert prompt_payload["event"] == "summarize"
    assert prompt_payload["fold_count"] == 6
    assert prompt_payload["keep_count"] == 2
    assert content_msg["role"] == "user"
    assert "structured summary" in content_msg["content"]
    # Fold messages ride as direct input items (not JSON-wrapped) so the
    # tokenizer sees them natively. Comments and reasoning are filtered.
    fold_items = inputs[:-2]
    assert {"role": "user", "content": "very old"} in fold_items
    assert {"role": "user", "content": "new"} not in fold_items
    assert all(
        not (m.get("type") == "comment" and m.get("kind") == "usage")
        for m in fold_items
    )


@pytest.mark.asyncio
async def test_compaction_skips_when_recency_split_is_zero(tmp_path):
    """If we trigger compaction but can't identify a recency cutoff, leave
    the session alone — better to defer than to write a bogus summary."""
    orch = _orchestrator(tmp_path, trigger=100, recency=999_999)
    orch.session.append("main", [
        {"role": "user", "content": "a"},
        {"type": "comment", "kind": "usage", "input_tokens": 200},
        {"role": "user", "content": "b"},
        {"type": "comment", "kind": "usage", "input_tokens": 250},
    ])

    create_mock = MagicMock()
    orch.llm = MagicMock()
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = create_mock

    await orch._compact_session_if_needed("main")

    create_mock.assert_not_called()
    assert not (tmp_path / "sessions" / "main.1.jsonl").exists()


@pytest.mark.asyncio
async def test_compaction_failure_leaves_session_unchanged(tmp_path):
    """A failing summary LLM call must not bump the version or corrupt state."""
    orch = _orchestrator(tmp_path, trigger=100, recency=10)
    orch.session.append("main", [
        {"role": "user", "content": "a"},
        {"type": "comment", "kind": "usage", "input_tokens": 50},
        {"role": "user", "content": "b"},
        {"type": "comment", "kind": "usage", "input_tokens": 500},
    ])

    async def fake_create(**kwargs):
        raise RuntimeError("model exploded")

    orch.llm = MagicMock()
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = fake_create

    await orch._compact_session_if_needed("main")

    assert not (tmp_path / "sessions" / "main.1.jsonl").exists()
    assert len(orch.session.load("main")) == 4


@pytest.mark.asyncio
async def test_summary_call_runs_tool_loop_then_emits_text(tmp_path):
    """Tool calls during summarization (e.g. memory writes) are executed,
    their results fed back, and the loop continues until the agent emits
    her final summary text."""
    orch = _orchestrator(tmp_path)

    tool_call_item = MagicMock()
    tool_call_item.type = "function_call"
    tool_call_item.call_id = "call-1"
    tool_call_item.name = "memory_write"
    tool_call_item.arguments = '{"note": "something important"}'

    first_resp = MagicMock()
    first_resp.output = [tool_call_item]
    second_resp = _mock_summary_response("FINAL SUMMARY")

    responses_iter = iter([first_resp, second_resp])

    async def fake_create(**kwargs):
        return next(responses_iter)

    orch.llm = MagicMock()
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = fake_create

    tool_outputs = []

    async def fake_tool(call_id, name, arguments):
        tool_outputs.append((name, arguments))
        return {"type": "function_call_output", "call_id": call_id, "output": "ok"}, []

    orch._execute_tool_call = fake_tool

    result = await orch._summarize(
        [{"role": "user", "content": "old"}],
        [{"role": "user", "content": "kept"}],
    )

    assert result == "FINAL SUMMARY"
    assert tool_outputs == [("memory_write", '{"note": "something important"}')]


@pytest.mark.asyncio
async def test_summary_call_raises_when_response_empty(tmp_path):
    """An LLM that returns no text content surfaces a clear error rather
    than silently writing an empty summary."""
    orch = _orchestrator(tmp_path)

    empty_resp = MagicMock()
    empty_resp.output = []

    async def fake_create(**kwargs):
        return empty_resp

    orch.llm = MagicMock()
    orch.llm.responses = MagicMock()
    orch.llm.responses.create = fake_create

    with pytest.raises(RuntimeError, match="no text content"):
        await orch._summarize(
            [{"role": "user", "content": "anything"}],
            [],
        )
