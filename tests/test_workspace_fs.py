"""Tests for workspace filesystem MCP server.

Reads are exposed as resources at `workspace:///{path}`. Writes go through
the flat `write_file(path, content_type, data)` tool. Discovery via
`list_directory` and `search_files`.
"""

import pytest


@pytest.fixture
def workspace_server(tmp_workspace, monkeypatch):
    """Import the server module with WORKSPACE_DIR pointed at tmp_workspace."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_workspace))
    import importlib
    import workspace_fs.server as mod
    importlib.reload(mod)
    return mod


# ── workspace:/// resource (reads) ───────────────────────


def test_read_text_file(workspace_server, tmp_workspace):
    result = workspace_server.workspace_resource("SOUL.md")
    # Text files come back as decoded utf-8 inside a ResourceResult.
    content = result.contents[0]
    assert isinstance(content.content, str)
    assert "I am a test agent" in content.content
    assert content.mime_type.startswith("text/")


def test_read_nested_text_file(workspace_server, tmp_workspace):
    (tmp_workspace / "memory" / "2026-05-06.md").write_text("# Today\n")
    result = workspace_server.workspace_resource("memory/2026-05-06.md")
    assert "Today" in result.contents[0].content


def test_read_binary_file(workspace_server, tmp_workspace):
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    (tmp_workspace / "icon.png").write_bytes(png_magic)
    result = workspace_server.workspace_resource("icon.png")
    content = result.contents[0]
    assert isinstance(content.content, bytes)
    assert content.content.startswith(b"\x89PNG")
    assert content.mime_type == "image/png"


def test_read_file_not_found(workspace_server):
    with pytest.raises(FileNotFoundError):
        workspace_server.workspace_resource("nonexistent.md")


def test_read_directory_raises(workspace_server, tmp_workspace):
    (tmp_workspace / "subdir").mkdir()
    with pytest.raises(IsADirectoryError):
        workspace_server.workspace_resource("subdir")


def test_read_path_traversal(workspace_server):
    with pytest.raises(ValueError, match="traversal"):
        workspace_server.workspace_resource("../../etc/passwd")


# ── write_file (flat) ────────────────────────────────────


def test_write_text(workspace_server, tmp_workspace):
    result = workspace_server.write_file(
        "test.txt", content_type="text/plain", data="hello world",
    )
    assert "11 chars" in result
    assert (tmp_workspace / "test.txt").read_text() == "hello world"


def test_write_binary_from_base64(workspace_server, tmp_workspace):
    import base64
    raw = b"\x89PNG\r\n\x1a\n"
    data = base64.b64encode(raw).decode()
    workspace_server.write_file(
        "icon.png", content_type="image/png", data=data,
    )
    assert (tmp_workspace / "icon.png").read_bytes() == raw


def test_write_creates_parents(workspace_server, tmp_workspace):
    workspace_server.write_file(
        "deep/nested/test.txt", content_type="text/plain", data="nested",
    )
    assert (tmp_workspace / "deep" / "nested" / "test.txt").read_text() == "nested"


def test_write_traversal(workspace_server):
    with pytest.raises(ValueError, match="traversal"):
        workspace_server.write_file(
            "../../evil.txt", content_type="text/plain", data="bad",
        )


def test_write_overwrites(workspace_server, tmp_workspace):
    workspace_server.write_file("note.txt", content_type="text/plain", data="first")
    workspace_server.write_file("note.txt", content_type="text/plain", data="second")
    assert (tmp_workspace / "note.txt").read_text() == "second"


# ── list_directory ───────────────────────────────────────


def test_list_directory(workspace_server):
    result = workspace_server.list_directory()
    assert any("SOUL.md" in item for item in result)
    assert any("USER.md" in item for item in result)


def test_list_directory_subdir(workspace_server, tmp_workspace):
    (tmp_workspace / "memory" / "2026-03-18.md").write_text("# Today\n")
    result = workspace_server.list_directory("memory")
    assert any("2026-03-18.md" in item for item in result)


# ── search_files ─────────────────────────────────────────


def test_search_files(workspace_server):
    result = workspace_server.search_files("*.md")
    assert any("SOUL.md" in item for item in result)
    assert any("USER.md" in item for item in result)


def test_search_files_recursive(workspace_server, tmp_workspace):
    (tmp_workspace / "memory" / "daily.md").write_text("# Daily\n")
    result = workspace_server.search_files("**/*.md")
    assert any("daily.md" in item for item in result)


def test_search_files_traversal(workspace_server):
    with pytest.raises(ValueError, match="traversal"):
        workspace_server.search_files("*", "../../")
