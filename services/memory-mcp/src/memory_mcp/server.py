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
    """
# Long-Term Memory Tools

This tool implements a basic long-term memory that you should use liberally to
make sure information persists outside of the active session.

You are provided with a global memory that stores curated important events as
well as a daily memory for more granular notes. Write early and often to your daily
memory and think about what can be promoted to global memory. Periodically examine your
global memory to think about what can be removed, summarized, or cleaned up
as no longer relevant.

At the start of a new session, always read your global memory as well as the last two
days memories. If these memories don't exist or if you want to, you may read more
after that minimal set.
"""
))


def _date_path(d: date) -> Path:
    return MEMORY_DIR / f"{d.isoformat()}.md"


@mcp.tool
def read_memory(date: date) -> str:
    """Read the memory entry for a specific date, formatted as YYYY-MM-DD"""
    path = _date_path(date)
    if not path.exists():
        raise FileNotFoundError(f"No memory entry for {date}")
    return path.read_text()


@mcp.tool
def edit_memory(date: date, content: str) -> str:
    """Write or replace the memory entry for a specific date, formatted as YYYY-MM-DD"""
    path = _date_path(date)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"Updated memory for {date} ({len(content)} bytes)" # FOLLOWUP: no news is good news


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
