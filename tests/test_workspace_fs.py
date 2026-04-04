"""Tests for workspace filesystem MCP server."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def workspace_server(tmp_workspace, monkeypatch):
    """Import the server module with WORKSPACE_DIR pointed at tmp_workspace."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_workspace))
    import importlib
    import workspace_fs.server as mod
    importlib.reload(mod)
    return mod


def test_read_file(workspace_server, tmp_workspace):
    result = workspace_server.read_file("SOUL.md")
    assert "I am a test agent" in result.data
    assert result.content_type.startswith("text/")


def test_read_file_not_found(workspace_server):
    with pytest.raises(FileNotFoundError):
        workspace_server.read_file("nonexistent.md")


def test_read_file_traversal(workspace_server):
    with pytest.raises(ValueError, match="traversal"):
        workspace_server.read_file("../../etc/passwd")


def test_write_file(workspace_server, tmp_workspace):
    result = workspace_server.write_file("test.txt", "hello world")
    assert "11 bytes" in result
    assert (tmp_workspace / "test.txt").read_text() == "hello world"


def test_write_file_creates_parents(workspace_server, tmp_workspace):
    workspace_server.write_file("subdir/deep/test.txt", "nested")
    assert (tmp_workspace / "subdir" / "deep" / "test.txt").read_text() == "nested"


def test_write_file_traversal(workspace_server):
    with pytest.raises(ValueError, match="traversal"):
        workspace_server.write_file("../../evil.txt", "bad")


def test_list_directory(workspace_server):
    result = workspace_server.list_directory()
    assert any("SOUL.md" in item for item in result)
    assert any("USER.md" in item for item in result)


def test_list_directory_subdir(workspace_server, tmp_workspace):
    (tmp_workspace / "memory" / "2026-03-18.md").write_text("# Today\n")
    result = workspace_server.list_directory("memory")
    assert any("2026-03-18.md" in item for item in result)


def test_search_files(workspace_server):
    result = workspace_server.search_files("*.md")
    assert any("SOUL.md" in item for item in result)
    assert any("USER.md" in item for item in result)


def test_search_files_recursive(workspace_server, tmp_workspace):
    (tmp_workspace / "memory" / "daily.md").write_text("# Daily\n")
    result = workspace_server.search_files("*.md")
    assert any("daily.md" in item for item in result)


def test_search_files_traversal(workspace_server):
    with pytest.raises(ValueError, match="traversal"):
        workspace_server.search_files("*", "../../")
