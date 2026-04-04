"""Workspace filesystem MCP server.

Exposes read_file, write_file, list_directory, search_files scoped to a
workspace root. Path traversal outside the workspace is rejected.
"""

import base64
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve()

mcp = FastMCP("workspace-fs", instructions=(
    "Workspace filesystem access. Contains your personality files (SOUL.md, USER.md, etc.), "
    "configuration (HEARTBEAT.md, TOOLS.md), and other workspace files. "
    "Use at startup to read your identity and context. Paths are relative to workspace root."
))


@dataclass
class File:
    """A file's content with its MIME type."""
    content_type: str
    data: str


def _safe_resolve(path: str) -> Path:
    """Resolve a path relative to WORKSPACE. Raises ValueError on traversal."""
    target = (WORKSPACE / path).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        raise ValueError(f"Path traversal not allowed: {path}")
    return target


def _is_text_file(path: Path) -> bool:
    """Check if a file is likely text based on MIME type."""
    mime = mimetypes.guess_type(str(path))[0] or ""
    return mime.startswith("text/") or mime in (
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    )


@mcp.tool
def read_file(path: str) -> File:
    """Read a file from the workspace. Returns content_type and data (text or base64 for binary)."""
    target = _safe_resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"{path} not found")
    if not target.is_file():
        raise IsADirectoryError(f"{path} is not a file")

    content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"

    # Try text first, fall back to base64 for binary
    try:
        data = target.read_text()
        if content_type == "application/octet-stream" and _is_text_file(target):
            content_type = "text/plain"
        elif content_type == "application/octet-stream":
            # Successfully read as text but unknown type — assume text
            content_type = "text/plain"
    except UnicodeDecodeError:
        data = base64.b64encode(target.read_bytes()).decode()

    return File(content_type=content_type, data=data)


@mcp.tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace. Creates parent directories as needed."""
    target = _safe_resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


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


@mcp.tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """Get the current date and time. Pass a timezone name like 'America/New_York' or 'UTC'."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
