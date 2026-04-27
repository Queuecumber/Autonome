"""Workspace filesystem MCP server.

Exposes read_file, write_file, list_directory, search_files scoped to a
workspace root. Path traversal outside the workspace is rejected.
"""

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

import filetype
from fastmcp import FastMCP
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from pydantic import Base64Bytes

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve()

mcp = FastMCP("workspace-fs", instructions=(
  """
# Workspace Tools

The workspace tools allow file access to your personal files, you can store anything you want in these
files.
"""
))


@dataclass
class File:
    """A file's content with its MIME type and path.

    `data` accepts either plain text (for text MIME types) or base64-
    encoded bytes (for binary). The orchestrator's pointer-rewriting
    layer treats the bytes branch of the union as eligible for pointer
    auto-resolution, so callers can pass `pointer://...` URIs directly
    instead of pre-fetching and base64-encoding."""
    content_type: str
    data: str | Base64Bytes
    path: str | None = None


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


@mcp.tool
def read_file(path: str) -> ImageContent | AudioContent | EmbeddedResource | TextContent:
    """Read a file from the workspace.

    Return shape depends on the detected content:
    - Images come back so you can see them.
    - Audio comes back as audio content.
    - Text files (text/*, JSON, XML, YAML) come back as text.
    - Other binaries (PDFs, zips, etc.) come back as an embedded resource
      that the session-manager persists as a pointer you can forward to
      other tools.

    Args:
        path: Relative to the workspace root (e.g. `Pictures/cat.jpg`).
            Traversal outside the workspace is rejected.

    Returns:
        The appropriate content block per the file's detected type.

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

    if kind and kind.mime.startswith("image/"):
        return ImageContent(type="image", data=base64.b64encode(raw).decode(), mimeType=kind.mime)
    if kind and kind.mime.startswith("audio/"):
        return AudioContent(type="audio", data=base64.b64encode(raw).decode(), mimeType=kind.mime)

    content_type = (kind.mime if kind else None) or mimetypes.guess_type(str(target))[0] or "text/plain"

    if _is_text_type(content_type):
        try:
            return TextContent(type="text", text=raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            pass

    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=f"file:///{path}",
            mimeType=content_type,
            blob=base64.b64encode(raw).decode(),
        ),
    )


@mcp.tool
def write_file(path: str, file: File) -> str:
    """Write a file to the workspace.

    Creates parent directories if needed. Overwrites any existing file
    at the path.

    Args:
        path: Where to write, relative to the workspace root.
        file: The file to write — its content type and contents.

    Returns:
        Short confirmation with byte/char count and the path written.

    Raises:
        ValueError: If the path attempts traversal outside the workspace.
    """
    target = _safe_resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = file.data
    if _is_text_type(file.content_type):
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
