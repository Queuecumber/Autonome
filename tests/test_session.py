"""Tests for SessionManager — versioned JSONL storage and token accounting."""

import json

from session_manager.session import SessionManager


def test_load_empty_session(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions)
    assert mgr.load("main") == []


def test_session_file_naming_starts_at_version_zero(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions)
    path = mgr._active_path("main")
    assert path.name == "main.0.jsonl"


def test_session_file_naming_sanitizes_separators(tmp_sessions):
    """Slashes and backslashes get replaced so session_id can't escape store_dir."""
    mgr = SessionManager(store_dir=tmp_sessions)
    path = mgr._active_path("matrix:!room/with/slashes")
    assert "/" not in path.name
    assert path.parent == tmp_sessions


def test_separate_sessions(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions)
    mgr.append("main", [{"role": "user", "content": "main msg"}])
    mgr.append("alt", [{"role": "user", "content": "alt msg"}])

    assert len(mgr.load("main")) == 1
    assert mgr.load("main")[0]["content"] == "main msg"
    assert len(mgr.load("alt")) == 1
    assert mgr.load("alt")[0]["content"] == "alt msg"


def test_append_and_reload(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions)
    mgr.append("main", [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    history = mgr.load("main")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"


def test_unversioned_file_is_migrated(tmp_sessions):
    """Legacy <id>.jsonl files get renamed to <id>.0.jsonl on init."""
    legacy = tmp_sessions / "main.jsonl"
    legacy.write_text(json.dumps({"role": "user", "content": "old"}) + "\n")

    mgr = SessionManager(store_dir=tmp_sessions)

    assert not legacy.exists()
    assert (tmp_sessions / "main.0.jsonl").exists()
    assert mgr.load("main") == [{"role": "user", "content": "old"}]


def test_unversioned_migration_skips_when_target_exists(tmp_sessions):
    """If <id>.0.jsonl already exists, the legacy <id>.jsonl is left alone."""
    (tmp_sessions / "main.jsonl").write_text(json.dumps({"role": "user", "content": "legacy"}) + "\n")
    (tmp_sessions / "main.0.jsonl").write_text(json.dumps({"role": "user", "content": "newer"}) + "\n")

    SessionManager(store_dir=tmp_sessions)

    assert (tmp_sessions / "main.jsonl").exists()
    assert (tmp_sessions / "main.0.jsonl").exists()


def test_bump_version_creates_next_file(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions)
    mgr.append("main", [{"role": "user", "content": "v0"}])

    new_path = mgr.bump_version("main", [{"role": "user", "content": "v1"}])

    assert new_path.name == "main.1.jsonl"
    assert mgr.load("main") == [{"role": "user", "content": "v1"}]
    # Older version is preserved on disk as audit trail.
    assert (tmp_sessions / "main.0.jsonl").exists()


def test_append_after_bump_writes_to_latest_version(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions)
    mgr.append("main", [{"role": "user", "content": "v0"}])
    mgr.bump_version("main", [{"role": "user", "content": "v1-start"}])

    mgr.append("main", [{"role": "user", "content": "v1-added"}])

    assert mgr.load("main") == [
        {"role": "user", "content": "v1-start"},
        {"role": "user", "content": "v1-added"},
    ]


def test_latest_input_tokens_walks_newest_first():
    messages = [
        {"role": "user", "content": "a"},
        {"type": "comment", "kind": "usage", "input_tokens": 100},
        {"role": "user", "content": "b"},
        {"type": "comment", "kind": "usage", "input_tokens": 250},
    ]
    assert SessionManager.latest_input_tokens(messages) == 250


def test_latest_input_tokens_none_when_no_usage():
    assert SessionManager.latest_input_tokens([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]) is None


def test_recency_split_returns_zero_with_too_few_usages():
    """With <2 usage comments we can't compute any delta, so keep everything."""
    assert SessionManager.recency_split([], 100) == 0
    assert SessionManager.recency_split(
        [{"type": "comment", "kind": "usage", "input_tokens": 100}], 50
    ) == 0


def test_recency_split_finds_boundary_by_walking_deltas():
    """Construct usage comments with known deltas and verify the split lands
    at the line just past the boundary."""
    messages = [
        {"role": "user", "content": "old"},          # line 0
        {"type": "comment", "kind": "usage",          # line 1
         "input_tokens": 100},
        {"role": "user", "content": "mid"},          # line 2
        {"type": "comment", "kind": "usage",          # line 3
         "input_tokens": 200},  # delta 100
        {"role": "user", "content": "new"},          # line 4
        {"type": "comment", "kind": "usage",          # line 5
         "input_tokens": 350},  # delta 150
    ]
    # Want ≥120 tokens of recency. Walking back: delta 150 ≥ 120, so the
    # boundary is right after the usage at index 3.
    assert SessionManager.recency_split(messages, 120) == 4


def test_recency_split_returns_zero_when_recency_exceeds_all_deltas():
    """If even all known deltas together don't reach recency, fold nothing."""
    messages = [
        {"type": "comment", "kind": "usage", "input_tokens": 100},
        {"role": "user", "content": "x"},
        {"type": "comment", "kind": "usage", "input_tokens": 150},
    ]
    assert SessionManager.recency_split(messages, 10_000) == 0
