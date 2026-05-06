"""Workspace filesystem MCP server.

Exposes read_file, write_file, list_directory, search_files scoped to a
workspace root. Path traversal outside the workspace is rejected.
"""

import base64
import mimetypes
import os
from pathlib import Path

import filetype
from fastmcp import FastMCP
from pydantic import Base64Bytes

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve()

mcp = FastMCP("workspace-fs", instructions=(
  """
# Workspace Tools

The workspace tools give you access to your personal files. Store anything
you want here.

Files are exposed as MCP resources at `workspace:///{path}` URIs. Use
`resources_read("workspace:///path/to/file.ext")` to view content (the
platform handles mime detection and image rendering). Pass these URIs as
the binary argument to other tools to forward content without reading it
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
def workspace_resource(path: str) -> bytes | str:
    """Serve workspace files as MCP resources at `workspace:///{path}`.

    The path is relative to the workspace root and may contain slashes
    (`{path*}` captures multi-segment paths). Returns text decoded as
    UTF-8 for text MIME types, raw bytes for binary; the platform's
    content pipeline handles mime detection and image rendering.

    Args:
        path: Multi-segment path relative to workspace root (e.g.
            `Pictures/cat.jpg`).

    Returns:
        File content. The orchestrator's resource pipeline displays it
        appropriately based on detected mime.

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

    # Text MIME types come back as decoded text; everything else as bytes.
    # fastmcp wraps bytes as BlobResourceContents and str as
    # TextResourceContents at the protocol layer.
    if _is_text_type(content_type):
        try:
            return raw.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            pass
    return raw


@mcp.tool
def write_file(path: str, content_type: str, data: str | Base64Bytes) -> str:
    """Write a file to the workspace.

    Creates parent directories if needed. Overwrites any existing file
    at the path.

    Args:
        path: Where to write, relative to the workspace root.
        content_type: MIME type of the content. Determines whether `data`
            is interpreted as text or binary.
        data: For text MIME types, the file contents as a string. For
            binary MIME types, base64-encoded bytes (or a resource URI
            like `pointer://...` or `mxc://...` — the platform resolves
            it to bytes before this tool runs).

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
