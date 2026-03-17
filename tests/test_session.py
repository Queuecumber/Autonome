import json
from pathlib import Path

from agent_platform.session import SessionManager


def test_load_empty_session(tmp_sessions):
    """Loading a session for a new contact returns empty history."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    history = mgr.load("+11111111111")
    assert history == []


def test_session_file_naming(tmp_sessions):
    """Session files are named by phone number."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    path = mgr._session_path("+11111111111")
    assert path.name == "+11111111111.jsonl"


def test_append_and_reload(tmp_sessions):
    """Messages appended to session can be reloaded."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    mgr.append("+11111111111", [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    history = mgr.load("+11111111111")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"


def test_load_truncated_drops_oldest(tmp_sessions):
    """When over token budget, oldest exchanges are dropped."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    messages = []
    for i in range(10):
        messages.extend([
            {"role": "user", "content": f"Message {i} " + "x" * 100},
            {"role": "assistant", "content": f"Response {i} " + "y" * 100},
        ])
    mgr.append("+11111111111", messages)

    truncated = mgr.load_truncated("+11111111111")
    assert len(truncated) < len(messages)
    assert truncated[-1]["content"].startswith("Response 9")


def test_load_truncated_preserves_exchange_integrity(tmp_sessions):
    """Truncation drops complete exchanges, not partial ones."""
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    mgr.append("+11111111111", [
        {"role": "user", "content": "old " + "x" * 200},
        {"role": "assistant", "content": "old reply " + "y" * 200},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new reply"},
    ])

    truncated = mgr.load_truncated("+11111111111")
    if truncated:
        assert truncated[0]["role"] == "user"
