"""Workspace filesystem MCP server.

Exposes a `workspace:///{path}` resource for reads, plus write_file,
list_directory, search_files tools. Path traversal outside the workspace
is rejected.
"""

import base64
import mimetypes
import os
from pathlib import Path

import filetype
from fastmcp import FastMCP
from fastmcp.resources import ResourceContent, ResourceResult
from pydantic import Base64Bytes

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve()

mcp = FastMCP("workspace-fs", instructions=(
  """
# Workspace Tools

The workspace tools give you access to your personal files. Store anything
you want here.

Files are exposed as MCP resources at `workspace:///{path}` URIs. Use
`resources_read("workspace:///path/to/file.ext")` to view content. Pass these URIs as
binary arguments to other tools to forward content without reading it
into your context first.

Use `list_directory` and `search_files` to discover what's there.
"""
))


def _safe_resolve(path: str) -> Path:
    """Resolve a path relative to WORKSPACE. Raises ValueError on traversal."""
    target = (WORKSPACE / path).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        raise ValueError(f"Path traversal not allowed: {path}")
    return target


TEXT_TYPES = {"application/json", "application/xml", "application/yaml", "application/x-yaml"}


def _is_text_type(content_type: str) -> bool:
    """Check if a MIME type represents text content."""
    return content_type.startswith("text/") or content_type in TEXT_TYPES


@mcp.resource("workspace:///{path*}")
def workspace_resource(path: str) -> ResourceResult:
    """Serve workspace files as MCP resources at `workspace:///{path}`.

    Args:
        path: Multi-segment path relative to workspace root (e.g.
            `Pictures/cat.jpg`).

    Returns:
        A `ResourceResult` with the detected mime. Text-shaped content is
        decoded to a string so it lands as `TextResourceContents`; everything
        else stays as bytes.

    Raises:
        ValueError: If the path attempts traversal outside the workspace.
        FileNotFoundError: If no file exists at that path.
        IsADirectoryError: If the path resolves to a directory.
    """
    target = _safe_resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"{path} not found")
    if not target.is_file():
        raise IsADirectoryError(f"{path} is not a file")

    raw = target.read_bytes()
    kind = filetype.guess(raw)
    content_type = (kind.mime if kind else None) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"

    if _is_text_type(content_type):
        try:
            return ResourceResult([ResourceContent(raw.decode("utf-8"), mime_type=content_type)])
        except (UnicodeDecodeError, ValueError):
            pass
    return ResourceResult([ResourceContent(raw, mime_type=content_type)])


@mcp.tool
def write_file(path: str, content_type: str, data: str | Base64Bytes) -> str:
    """Write a file to the workspace.

    Creates parent directories if needed. Overwrites any existing file
    at the path.

    Args:
        path: Where to write, relative to the workspace root.
        content_type: MIME type of the content. Determines whether `data`
            is interpreted as text or binary.
        data: Text contents for text MIME types, base64-encoded bytes
            for binary.

    Returns:
        Short confirmation with byte/char count and the path written.

    Raises:
        ValueError: If the path attempts traversal outside the workspace.
    """
    target = _safe_resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_text_type(content_type):
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        target.write_text(text)
        return f"Wrote {len(text)} chars to {path}"
    raw = data if isinstance(data, bytes) else base64.b64decode(data)
    target.write_bytes(raw)
    return f"Wrote {len(raw)} bytes to {path}"


@mcp.tool
def list_directory(path: str = ".") -> list[str]:
    """List entries at a path in the workspace.

    Args:
        path: Relative to the workspace root. Default `.` is the
            workspace root itself.

    Returns:
        Paths of entries directly contained at that path, sorted
        alphabetically, relative to the workspace root.

    Raises:
        ValueError: If the path attempts traversal outside the workspace.
        FileNotFoundError: If the path doesn't exist.
        NotADirectoryError: If the path resolves to a file.
    """
    target = _safe_resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"{path} not found")
    if not target.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")
    return [str(p.relative_to(WORKSPACE)) for p in sorted(target.iterdir())]


@mcp.tool
def search_files(pattern: str, path: str = ".") -> list[str]:
    """Recursively search for files matching a glob pattern.

    Args:
        pattern: A glob (e.g. `*.md`, `**/PERSONALITY.md`,
            `Pictures/*.jpg`). Standard pathlib syntax.
        path: Subdirectory to search within, relative to the workspace
            root. Default `.` searches the whole workspace.

    Returns:
        Matching file paths (only files, not directories) relative to
        the workspace root, sorted alphabetically.

    Raises:
        ValueError: If the path attempts traversal outside the workspace.
        FileNotFoundError: If the search root doesn't exist.
    """
    target = _safe_resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"{path} not found")
    return [str(p.relative_to(WORKSPACE)) for p in sorted(target.rglob(pattern)) if p.is_file()]



if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
