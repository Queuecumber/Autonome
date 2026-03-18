"""Workspace filesystem MCP server.

Exposes read_file, write_file, list_directory, search_files scoped to a
workspace root. Path traversal outside the workspace is rejected.
"""

import os
from pathlib import Path

from fastmcp import FastMCP

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve()

mcp = FastMCP("workspace-fs")


def _safe_resolve(path: str) -> Path | None:
    """Resolve a path relative to WORKSPACE, rejecting traversal."""
    target = (WORKSPACE / path).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        return None
    return target


@mcp.tool
def read_file(path: str) -> str:
    """Read a file from the workspace."""
    target = _safe_resolve(path)
    if target is None:
        return "Error: path traversal not allowed"
    if not target.exists():
        return f"Error: {path} not found"
    if not target.is_file():
        return f"Error: {path} is not a file"
    return target.read_text()


@mcp.tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace. Creates parent directories as needed."""
    target = _safe_resolve(path)
    if target is None:
        return "Error: path traversal not allowed"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


@mcp.tool
def list_directory(path: str = ".") -> list[str]:
    """List files and directories at a path in the workspace."""
    target = _safe_resolve(path)
    if target is None:
        return ["Error: path traversal not allowed"]
    if not target.exists():
        return [f"Error: {path} not found"]
    if not target.is_dir():
        return [f"Error: {path} is not a directory"]
    return [str(p.relative_to(WORKSPACE)) for p in sorted(target.iterdir())]


@mcp.tool
def search_files(pattern: str, path: str = ".") -> list[str]:
    """Search for files matching a glob pattern within the workspace."""
    target = _safe_resolve(path)
    if target is None:
        return ["Error: path traversal not allowed"]
    if not target.exists():
        return [f"Error: {path} not found"]
    return [str(p.relative_to(WORKSPACE)) for p in sorted(target.rglob(pattern)) if p.is_file()]


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
