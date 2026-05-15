"""Tests for SessionManager — single-session-id storage with token-based truncation."""

from session_manager.session import SessionManager


def test_load_empty_session(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    assert mgr.load("main") == []


def test_session_file_naming(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    path = mgr._session_path("main")
    assert path.name == "main.jsonl"


def test_session_file_naming_sanitizes_separators(tmp_sessions):
    """Slashes and backslashes get replaced so session_id can't escape store_dir."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    path = mgr._session_path("matrix:!room/with/slashes")
    assert "/" not in path.name
    assert path.parent == tmp_sessions


def test_separate_sessions(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    mgr.append("main", [{"role": "user", "content": "main msg"}])
    mgr.append("alt", [{"role": "user", "content": "alt msg"}])

    assert len(mgr.load("main")) == 1
    assert mgr.load("main")[0]["content"] == "main msg"
    assert len(mgr.load("alt")) == 1
    assert mgr.load("alt")[0]["content"] == "alt msg"


def test_append_and_reload(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    mgr.append("main", [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    history = mgr.load("main")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"


def test_load_truncated_drops_oldest(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100)
    messages = []
    for i in range(10):
        messages.extend([
            {"role": "user", "content": f"Message {i} " + "x" * 100},
            {"role": "assistant", "content": f"Response {i} " + "y" * 100},
        ])
    mgr.append("main", messages)

    truncated = mgr.load_truncated("main")
    assert len(truncated) < len(messages)
    # Most recent items survive.
    assert any("Response 9" in m.get("content", "") for m in truncated[-2:])


def test_load_truncated_preserves_exchange_integrity(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    mgr.append("main", [
        {"role": "user", "content": "old " + "x" * 200},
        {"role": "assistant", "content": "old reply " + "y" * 200},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new reply"},
    ])
    truncated = mgr.load_truncated("main")
    if truncated:
        # Truncation respects exchange boundaries — first surviving item
        # should be a user message, never start mid-assistant-reply.
        assert truncated[0]["role"] == "user"
