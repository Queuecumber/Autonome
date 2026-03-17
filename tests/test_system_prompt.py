from pathlib import Path

from agent_platform.callbacks.system_prompt import discover_workspace_md, build_system_message


def test_discover_workspace_md(tmp_workspace):
    """Discovers all .md files in workspace root (not subdirectories)."""
    files = discover_workspace_md(tmp_workspace)
    names = [f.name for f in files]
    assert "SOUL.md" in names
    assert "USER.md" in names


def test_discover_workspace_md_sorted(tmp_workspace):
    """Files are returned in alphabetical order."""
    (tmp_workspace / "AGENTS.md").write_text("# Agents\n")
    (tmp_workspace / "ZZZZZ.md").write_text("# Last\n")
    files = discover_workspace_md(tmp_workspace)
    names = [f.name for f in files]
    assert names == sorted(names)


def test_discover_workspace_md_ignores_subdirs(tmp_workspace):
    """Files in subdirectories (like memory/) are not included."""
    (tmp_workspace / "memory" / "2026-03-17.md").write_text("# Daily\n")
    files = discover_workspace_md(tmp_workspace)
    names = [f.name for f in files]
    assert "2026-03-17.md" not in names


def test_build_system_message_content(tmp_workspace):
    """System message contains content from all workspace markdown files."""
    msg = build_system_message(tmp_workspace)
    assert "I am a test agent" in msg
    assert "Test user" in msg


def test_build_system_message_empty(tmp_path):
    """Empty workspace produces empty string."""
    empty = tmp_path / "empty_workspace"
    empty.mkdir()
    msg = build_system_message(empty)
    assert msg == ""
