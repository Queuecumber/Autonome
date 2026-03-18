import json
from pathlib import Path

from agent_platform.session import SessionManager


def test_load_empty_session(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    history = mgr.load("signal", "+11111111111")
    assert history == []


def test_session_file_naming(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    path = mgr._session_path("signal", "+11111111111")
    assert path.name == "signal_+11111111111.jsonl"


def test_separate_channel_sessions(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    mgr.append("signal", "+11111111111", [{"role": "user", "content": "signal msg"}])
    mgr.append("coding", "project-x", [{"role": "user", "content": "coding msg"}])

    signal_history = mgr.load("signal", "+11111111111")
    coding_history = mgr.load("coding", "project-x")

    assert len(signal_history) == 1
    assert signal_history[0]["content"] == "signal msg"
    assert len(coding_history) == 1
    assert coding_history[0]["content"] == "coding msg"


def test_append_and_reload(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=100000)
    mgr.append("signal", "+11111111111", [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    history = mgr.load("signal", "+11111111111")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"


def test_load_truncated_drops_oldest(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    messages = []
    for i in range(10):
        messages.extend([
            {"role": "user", "content": f"Message {i} " + "x" * 100},
            {"role": "assistant", "content": f"Response {i} " + "y" * 100},
        ])
    mgr.append("signal", "+11111111111", messages)

    truncated = mgr.load_truncated("signal", "+11111111111")
    assert len(truncated) < len(messages)
    assert truncated[-1]["content"].startswith("Response 9")


def test_load_truncated_preserves_exchange_integrity(tmp_sessions):
    mgr = SessionManager(store_dir=tmp_sessions, max_history_tokens=50)
    mgr.append("signal", "+11111111111", [
        {"role": "user", "content": "old " + "x" * 200},
        {"role": "assistant", "content": "old reply " + "y" * 200},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new reply"},
    ])

    truncated = mgr.load_truncated("signal", "+11111111111")
    if truncated:
        assert truncated[0]["role"] == "user"
