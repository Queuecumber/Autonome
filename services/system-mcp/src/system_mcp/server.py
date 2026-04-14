"""System MCP server — web search, web fetch, and general system tools."""

import os

import html2text
import httpx
from fastmcp import FastMCP

SEARCH_URL = os.environ.get("SEARCH_URL", "https://api.perplexity.ai/search")
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY", "")
MAX_FETCH_CHARS = int(os.environ.get("MAX_FETCH_CHARS", "20000"))

mcp = FastMCP("system", instructions=(
    "System tools. Use web_search to find information online, "
    "and web_fetch to retrieve the full content of a specific URL."
))

_http = httpx.AsyncClient(timeout=30, headers={"User-Agent": "AgentPlatform/1.0"})

_h2t = html2text.HTML2Text()
_h2t.ignore_links = False
_h2t.ignore_images = True
_h2t.body_width = 0  # no line wrapping


@mcp.tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Returns ranked results with title, URL, and snippet."""
    resp = await _http.post(
        SEARCH_URL,
        headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
        json={"query": query, "max_results": max_results},
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        lines.append(f"### {title}\n{url}\n{snippet}")
    return "\n\n".join(lines)


@mcp.tool
async def web_fetch(url: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """Fetch a URL and return its content as markdown. Large pages are truncated."""
    resp = await _http.get(url, follow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "html" in content_type:
        text = _h2t.handle(resp.text)
    else:
        text = resp.text

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"
    return text


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8002)
