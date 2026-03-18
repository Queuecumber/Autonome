"""Tests for workspace filesystem MCP server."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def workspace_server(tmp_workspace, monkeypatch):
    """Import the server module with WORKSPACE_DIR pointed at tmp_workspace."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_workspace))
    # Need to reimport so WORKSPACE picks up the env var
    import importlib
    import mcp_servers.workspace_fs.server as mod
    importlib.reload(mod)
    return mod


def test_read_file(workspace_server, tmp_workspace):
    """read_file returns file content."""
    result = workspace_server.read_file("SOUL.md")
    assert "I am a test agent" in result


def test_read_file_not_found(workspace_server):
    """read_file returns error for missing files."""
    result = workspace_server.read_file("nonexistent.md")
    assert "not found" in result


def test_read_file_traversal(workspace_server):
    """read_file rejects path traversal."""
    result = workspace_server.read_file("../../etc/passwd")
    assert "traversal" in result


def test_write_file(workspace_server, tmp_workspace):
    """write_file creates files in workspace."""
    result = workspace_server.write_file("test.txt", "hello world")
    assert "11 bytes" in result
    assert (tmp_workspace / "test.txt").read_text() == "hello world"


def test_write_file_creates_parents(workspace_server, tmp_workspace):
    """write_file creates parent directories."""
    workspace_server.write_file("subdir/deep/test.txt", "nested")
    assert (tmp_workspace / "subdir" / "deep" / "test.txt").read_text() == "nested"


def test_write_file_traversal(workspace_server):
    """write_file rejects path traversal."""
    result = workspace_server.write_file("../../evil.txt", "bad")
    assert "traversal" in result


def test_list_directory(workspace_server):
    """list_directory returns workspace contents."""
    result = workspace_server.list_directory()
    assert any("SOUL.md" in item for item in result)
    assert any("USER.md" in item for item in result)


def test_list_directory_subdir(workspace_server, tmp_workspace):
    """list_directory works on subdirectories."""
    (tmp_workspace / "memory" / "2026-03-18.md").write_text("# Today\n")
    result = workspace_server.list_directory("memory")
    assert any("2026-03-18.md" in item for item in result)


def test_search_files(workspace_server):
    """search_files finds files by glob pattern."""
    result = workspace_server.search_files("*.md")
    assert any("SOUL.md" in item for item in result)
    assert any("USER.md" in item for item in result)


def test_search_files_recursive(workspace_server, tmp_workspace):
    """search_files searches recursively."""
    (tmp_workspace / "memory" / "daily.md").write_text("# Daily\n")
    result = workspace_server.search_files("*.md")
    assert any("daily.md" in item for item in result)


def test_search_files_traversal(workspace_server):
    """search_files rejects path traversal."""
    result = workspace_server.search_files("*", "../../")
    assert "traversal" in result[0]
