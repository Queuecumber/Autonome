"""Tests for memory MCP server."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def memory_server(tmp_path, monkeypatch):
    """Import the memory server with MEMORY_DIR pointed at a temp directory."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setenv("MEMORY_DIR", str(memory_dir))
    import mcp_servers.memory.server as mod
    importlib.reload(mod)
    return mod


@pytest.fixture
def memory_dir(memory_server):
    """The resolved memory directory path."""
    return memory_server.MEMORY_DIR


# --- read/edit daily memories ---


def test_read_memory_empty(memory_server):
    result = memory_server.read_memory("2026-03-18")
    assert "No memory entry" in result


def test_edit_and_read_memory(memory_server, memory_dir):
    memory_server.edit_memory("2026-03-18", "# Today\nStuff happened.\n")
    result = memory_server.read_memory("2026-03-18")
    assert "Stuff happened" in result
    assert (memory_dir / "2026-03-18.md").exists()


def test_edit_memory_overwrites(memory_server):
    memory_server.edit_memory("2026-03-18", "version 1")
    memory_server.edit_memory("2026-03-18", "version 2")
    result = memory_server.read_memory("2026-03-18")
    assert result == "version 2"


def test_read_memory_invalid_date(memory_server):
    result = memory_server.read_memory("not-a-date")
    assert "Error" in result


def test_edit_memory_invalid_date(memory_server):
    result = memory_server.edit_memory("13-99-2026", "bad")
    assert "Error" in result


# --- global memory ---


def test_read_global_memory_empty(memory_server):
    result = memory_server.read_global_memory()
    assert "No global memory" in result


def test_edit_and_read_global_memory(memory_server, memory_dir):
    memory_server.edit_global_memory("# Memory Index\n- [2026-03-18](2026-03-18.md)\n")
    result = memory_server.read_global_memory()
    assert "Memory Index" in result
    assert (memory_dir / "MEMORY.md").exists()


# --- list memories ---


def test_list_memories_empty(memory_server):
    result = memory_server.list_memories()
    assert result == []


def test_list_memories_returns_dates(memory_server):
    memory_server.edit_memory("2026-03-16", "day 1")
    memory_server.edit_memory("2026-03-18", "day 3")
    memory_server.edit_memory("2026-03-17", "day 2")

    result = memory_server.list_memories()
    assert result == ["2026-03-16", "2026-03-17", "2026-03-18"]


def test_list_memories_excludes_global(memory_server):
    """MEMORY.md should not appear in the date list."""
    memory_server.edit_global_memory("global stuff")
    memory_server.edit_memory("2026-03-18", "daily stuff")

    result = memory_server.list_memories()
    assert "MEMORY" not in str(result)
    assert "2026-03-18" in result


def test_list_memories_ignores_non_date_files(memory_server, memory_dir):
    """Non-date markdown files are ignored."""
    (memory_dir / "random-notes.md").write_text("not a date")
    memory_server.edit_memory("2026-03-18", "real entry")

    result = memory_server.list_memories()
    assert result == ["2026-03-18"]
