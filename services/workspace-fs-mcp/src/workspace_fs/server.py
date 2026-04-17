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
from mcp.types import AudioContent, ImageContent, TextContent

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve()

mcp = FastMCP("workspace-fs", instructions=(
    "Workspace filesystem access. Contains your personality files (SOUL.md, USER.md, etc.), "
    "configuration (HEARTBEAT.md, TOOLS.md), and other workspace files. "
    "Use at startup to read your identity and context. Paths are relative to workspace root."
))


@dataclass
class File:
    """A file's content with its MIME type and path."""
    content_type: str
    data: str
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
def read_file(path: str) -> ImageContent | AudioContent | TextContent:
    """Read a file from the workspace.

    Images and audio are returned as their MCP content blocks so the session
    manager can persist them as binary pointers. Text files come back as
    TextContent with the file content. Unsupported binaries come back as a
    TextContent placeholder describing the content type and size.
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

    return TextContent(type="text", text=f"[binary: content_type={content_type}, size={len(raw)} bytes, path={path}]")


@mcp.tool
def write_file(path: str, file: File) -> str:
    """Write a file to the workspace. Accepts a File with content_type and data."""
    target = _safe_resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_text_type(file.content_type):
        target.write_text(file.data)
        return f"Wrote {len(file.data)} chars to {path}"
    else:
        raw = base64.b64decode(file.data)
        target.write_bytes(raw)
        return f"Wrote {len(raw)} bytes to {path}"


@mcp.tool
def list_directory(path: str = ".") -> list[str]:
    """List files and directories at a path in the workspace."""
    target = _safe_resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"{path} not found")
    if not target.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")
    return [str(p.relative_to(WORKSPACE)) for p in sorted(target.iterdir())]


@mcp.tool
def search_files(pattern: str, path: str = ".") -> list[str]:
    """Search for files matching a glob pattern within the workspace."""
    target = _safe_resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"{path} not found")
    return [str(p.relative_to(WORKSPACE)) for p in sorted(target.rglob(pattern)) if p.is_file()]



if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
