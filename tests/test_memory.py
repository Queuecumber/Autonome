"""Tests for memory MCP server."""

import importlib
from datetime import date
from pathlib import Path

import pytest


@pytest.fixture
def memory_server(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setenv("MEMORY_DIR", str(memory_dir))
    import memory_mcp.server as mod
    importlib.reload(mod)
    return mod


@pytest.fixture
def memory_dir(memory_server):
    return memory_server.MEMORY_DIR


# --- read/edit daily memories ---


def test_read_memory_empty(memory_server):
    with pytest.raises(FileNotFoundError):
        memory_server.read_memory(date(2026, 3, 18))


def test_edit_and_read_memory(memory_server, memory_dir):
    memory_server.edit_memory(date(2026, 3, 18), "# Today\nStuff happened.\n")
    result = memory_server.read_memory(date(2026, 3, 18))
    assert "Stuff happened" in result
    assert (memory_dir / "2026-03-18.md").exists()


def test_edit_memory_overwrites(memory_server):
    memory_server.edit_memory(date(2026, 3, 18), "version 1")
    memory_server.edit_memory(date(2026, 3, 18), "version 2")
    result = memory_server.read_memory(date(2026, 3, 18))
    assert result == "version 2"


# --- global memory ---


def test_read_global_memory_empty(memory_server):
    with pytest.raises(FileNotFoundError):
        memory_server.read_global_memory()


def test_edit_and_read_global_memory(memory_server, memory_dir):
    memory_server.edit_global_memory("# Memory Index\n- [2026-03-18](2026-03-18.md)\n")
    result = memory_server.read_global_memory()
    assert "Memory Index" in result
    assert (memory_dir / "MEMORY.md").exists()


# --- list memories ---


def test_list_memories_empty(memory_server):
    assert memory_server.list_memories() == []


def test_list_memories_returns_dates(memory_server):
    memory_server.edit_memory(date(2026, 3, 16), "day 1")
    memory_server.edit_memory(date(2026, 3, 18), "day 3")
    memory_server.edit_memory(date(2026, 3, 17), "day 2")

    result = memory_server.list_memories()
    assert result == [date(2026, 3, 16), date(2026, 3, 17), date(2026, 3, 18)]


def test_list_memories_excludes_global(memory_server):
    memory_server.edit_global_memory("global stuff")
    memory_server.edit_memory(date(2026, 3, 18), "daily stuff")

    result = memory_server.list_memories()
    assert date(2026, 3, 18) in result
    assert len(result) == 1


def test_list_memories_ignores_non_date_files(memory_server, memory_dir):
    (memory_dir / "random-notes.md").write_text("not a date")
    memory_server.edit_memory(date(2026, 3, 18), "real entry")

    result = memory_server.list_memories()
    assert result == [date(2026, 3, 18)]
