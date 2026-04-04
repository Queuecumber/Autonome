"""Memory MCP server — daily markdown journals + global memory.

Drop-in compatible with OpenClaw's memory file pattern. Stores daily
entries as YYYY-MM-DD.md and a global index as MEMORY.md. Users can
swap this out for a richer implementation (vector store, graph DB, etc.)
by providing a different MCP server with its own tools.
"""

import os
from datetime import date
from pathlib import Path

from fastmcp import FastMCP

MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", "/memory")).resolve()

mcp = FastMCP("memory", instructions=(
    "Your long-term memory system. Daily entries are stored by date (YYYY-MM-DD), "
    "and a global MEMORY.md serves as your curated long-term index. "
    "Read recent daily memories and global memory at startup for context. "
    "Write memories to preserve important events, decisions, and learnings."
))


def _date_path(d: date) -> Path:
    return MEMORY_DIR / f"{d.isoformat()}.md"


@mcp.tool
def read_memory(date: date) -> str:
    """Read the memory entry for a specific date."""
    path = _date_path(date)
    if not path.exists():
        raise FileNotFoundError(f"No memory entry for {date}")
    return path.read_text()


@mcp.tool
def edit_memory(date: date, content: str) -> str:
    """Write or replace the memory entry for a specific date."""
    path = _date_path(date)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"Updated memory for {date} ({len(content)} bytes)"


@mcp.tool
def read_global_memory() -> str:
    """Read the global MEMORY.md index file."""
    path = MEMORY_DIR / "MEMORY.md"
    if not path.exists():
        raise FileNotFoundError("No global memory file exists yet")
    return path.read_text()


@mcp.tool
def edit_global_memory(content: str) -> str:
    """Write or replace the global MEMORY.md index file."""
    path = MEMORY_DIR / "MEMORY.md"
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"Updated global memory ({len(content)} bytes)"


@mcp.tool
def list_memories() -> list[date]:
    """List all dates that have memory entries, sorted chronologically."""
    if not MEMORY_DIR.exists():
        return []
    dates = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        try:
            dates.append(date.fromisoformat(f.stem))
        except ValueError:
            continue
    return dates


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001)
