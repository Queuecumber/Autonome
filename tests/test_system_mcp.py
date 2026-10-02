"""PDF web fetches must reach the model as binary MCP resources, not decoded text."""

import base64
import gzip
import io
import json
from contextlib import closing

from fastmcp import Client
from fastmcp.tools.tool import ToolResult
import httpx
from mcp.types import EmbeddedResource
import pytest

from system_mcp import server


@pytest.fixture
async def fetch_client(monkeypatch):
    """Replace the system MCP HTTP client with a local, deterministic transport."""
    responses = {}

    def handle(request):
        return responses[str(request.url)]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(server, "_http", client)
        yield responses


@pytest.mark.asyncio
@pytest.mark.parametrize("mime", ["application/pdf", "APPLICATION/PDF; version=1.7",
                                     "application/octet-stream", "binary/octet-stream", ""])
async def test_pdf_fetch_returns_binary_resource_with_final_url(fetch_client, mime):
    """Declared and signature-detected PDFs retain bytes and their final URL."""
    raw = b"%PDF-1.7\n\xff\x00payload"
    source = "https://example.org/source"
    final = "https://example.org/final.pdf"
    fetch_client[source] = httpx.Response(302, headers={"location": final})
    fetch_client[final] = httpx.Response(200, headers={"content-type": mime}, content=raw)

    result = await server.web_fetch(source, max_chars=1)

    assert isinstance(result, ToolResult)
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, EmbeddedResource)
    assert str(block.resource.uri) == final
    assert block.resource.mimeType == "application/pdf"
    assert base64.b64decode(block.resource.blob) == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("mime,raw,expected", [
    ("text/html; charset=utf-8", b"<h1>Welcome</h1>", "# Welcome"),
    ("text/plain; charset=iso-8859-1", b"caf\xe9", "café"),
])
async def test_compressed_text_is_decoded_once(fetch_client, mime, raw, expected):
    """Compressed HTML/text is decompressed once and retains its charset."""
    url = "https://example.org/compressed"
    fetch_client[url] = httpx.Response(
        200, headers={"content-type": mime, "content-encoding": "gzip"},
        content=gzip.compress(raw),
    )
    result = await server.web_fetch(url)
    assert isinstance(result, str)
    assert result.strip() == expected


@pytest.mark.asyncio
async def test_compressed_pdf_returns_exact_decompressed_bytes(fetch_client):
    """PDF resource payloads are decompressed exactly once, not raw gzip bytes."""
    url = "https://example.org/compressed.pdf"
    raw = b"%PDF-1.7\n\xff\x00payload"
    fetch_client[url] = httpx.Response(
        200, headers={"content-type": "application/pdf", "content-encoding": "gzip"},
        content=gzip.compress(raw),
    )
    result = await server.web_fetch(url, max_chars=1)
    assert isinstance(result, ToolResult)
    assert base64.b64decode(result.content[0].resource.blob) == raw


@pytest.mark.asyncio
async def test_generic_pdf_signature_split_across_chunks(fetch_client):
    """Generic MIME PDF recognition works when the signature spans chunks."""
    url = "https://example.org/chunked.pdf"
    raw = b"%PDF-1.7\nchunked"

    class Body(httpx.AsyncByteStream):
        """Emit a PDF signature in separate streaming chunks."""

        async def __aiter__(self):
            for chunk in (b"%", b"PD", b"F", b"-1.7\nchunked"):
                yield chunk

    fetch_client[url] = httpx.Response(
        200, headers={"content-type": "application/octet-stream"}, stream=Body(),
    )
    result = await server.web_fetch(url, max_chars=1)
    assert isinstance(result, ToolResult)
    assert base64.b64decode(result.content[0].resource.blob) == raw


@pytest.mark.asyncio
async def test_pdf_exact_byte_limit_is_accepted(fetch_client, monkeypatch):
    """A PDF whose decoded size equals the byte cap is returned in full."""
    raw = b"%PDF-123"
    monkeypatch.setattr(server, "MAX_PDF_BYTES", len(raw))
    url = "https://example.org/exact.pdf"
    fetch_client[url] = httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=raw,
    )
    result = await server.web_fetch(url, max_chars=1)
    assert isinstance(result, ToolResult)
    assert base64.b64decode(result.content[0].resource.blob) == raw


@pytest.mark.asyncio
async def test_pdf_tool_wire_returns_embedded_resource(fetch_client):
    """FastMCP sends an MCP resource block, not a serialized tool-result string."""
    url = "https://example.org/calendar.pdf"
    raw = b"%PDF-1.7\nwire-test"
    fetch_client[url] = httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=raw)

    async with Client(server.mcp) as client:
        result = await client.call_tool_mcp("web_fetch", {"url": url, "max_chars": 1})

    assert not result.isError
    assert len(result.content) == 1
    assert isinstance(result.content[0], EmbeddedResource)
    assert str(result.content[0].resource.uri) == url
    assert result.content[0].resource.mimeType == "application/pdf"
    assert base64.b64decode(result.content[0].resource.blob) == raw


@pytest.mark.asyncio
async def test_web_fetch_pdf_wire_result_renders_as_page_images(fetch_client):
    """A real PDF from web_fetch traverses MCP and session-manager conversion."""
    import pypdfium2 as pdfium

    from session_manager.mcp import mcp_content_to_openai

    pdf_bytes = io.BytesIO()
    with pdfium.PdfDocument.new() as document:
        with closing(document.new_page(100, 120)) as page:
            page.gen_content()
        document.save(pdf_bytes)

    url = "https://example.org/rendered.pdf"
    fetch_client[url] = httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=pdf_bytes.getvalue(),
    )
    async with Client(server.mcp) as client:
        result = await client.call_tool_mcp("web_fetch", {"url": url, "max_chars": 1})

    assert not result.isError
    assert len(result.content) == 1
    assert isinstance(result.content[0], EmbeddedResource)
    parts = mcp_content_to_openai(result.content)
    assert len(parts) == 2
    assert json.loads(parts[0]["text"]) == {"pdf": {
        "uri": url, "content_type": "application/pdf", "total_pages": 1,
        "rendered_pages": 1, "truncated": False,
    }}
    assert parts[1]["type"] == "input_image"
    assert parts[1]["detail"] == "high"
    assert parts[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(parts[1]["image_url"].split(",", 1)[1]).startswith(b"\xff\xd8")


@pytest.mark.asyncio
async def test_text_tool_wire_remains_text(fetch_client):
    """Non-PDF tool results retain their existing MCP text block shape."""
    url = "https://example.org/plain"
    fetch_client[url] = httpx.Response(200, headers={"content-type": "text/plain"}, content=b"hello")
    async with Client(server.mcp) as client:
        result = await client.call_tool_mcp("web_fetch", {"url": url})
    assert not result.isError
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "hello"


@pytest.mark.asyncio
async def test_pdf_over_limit_fails_before_full_download(fetch_client, monkeypatch):
    """A PDF download fails early without returning partial resource content."""
    monkeypatch.setattr(server, "MAX_PDF_BYTES", 8)
    url = "https://example.org/large.pdf"
    chunks_read = []

    class Body(httpx.AsyncByteStream):
        """Track reads so an oversized response cannot consume its final chunk."""

        async def __aiter__(self):
            for chunk in (b"%PDF-", b"1234", b"unread"):
                chunks_read.append(chunk)
                yield chunk

    fetch_client[url] = httpx.Response(200, headers={"content-type": "application/pdf"},
                                       stream=Body())
    with pytest.raises(ValueError, match="25 MiB input limit"):
        await server.web_fetch(url)
    assert chunks_read == [b"%PDF-", b"1234"]

    fetch_client[url] = httpx.Response(200, content=b"%PDF-12345678")
    with pytest.raises(ValueError, match="25 MiB input limit"):
        await server.web_fetch(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("mime,body,expected", [
    ("text/html; charset=utf-8", b"<h1>Welcome</h1>", "# W\n\n[truncated at 3 chars]"),
    ("text/plain", b"abcdef", "abc\n\n[truncated at 3 chars]"),
    ("text/plain; charset=iso-8859-1", b"caf\xe9", "caf\n\n[truncated at 3 chars]"),
    ("application/octet-stream", b"abcdef", "abc\n\n[truncated at 3 chars]"),
    ("text/plain", b"%PDF-1.7", "%PD\n\n[truncated at 3 chars]"),
])
async def test_text_fetch_remains_text(fetch_client, mime, body, expected):
    """HTML, ordinary text, and PDFs explicitly declared as text keep text behavior."""
    url = "https://example.org/page"
    fetch_client[url] = httpx.Response(200, headers={"content-type": mime}, content=body)
    result = await server.web_fetch(url, max_chars=3)
    assert isinstance(result, str)
    assert result == expected


@pytest.mark.asyncio
async def test_web_search_formats_ranked_results(fetch_client):
    """Search results render as markdown sections in rank order."""
    fetch_client["https://api.perplexity.ai/search"] = httpx.Response(
        200, json={"results": [
            {"title": "One", "url": "https://one.example", "snippet": "first"},
            {"title": "Two", "url": "https://two.example", "snippet": "second"},
        ]},
    )
    result = await server.web_search("query", max_results=2)
    assert result == (
        "### One\nhttps://one.example\nfirst"
        "\n\n### Two\nhttps://two.example\nsecond"
    )


@pytest.mark.asyncio
async def test_web_search_no_results(fetch_client):
    """An empty result set yields a clear message rather than blank output."""
    fetch_client["https://api.perplexity.ai/search"] = httpx.Response(
        200, json={"results": []},
    )
    assert await server.web_search("query") == "No results found."


@pytest.mark.asyncio
async def test_web_search_http_error_is_preserved(fetch_client):
    """Search backend failures propagate rather than returning empty results."""
    fetch_client["https://api.perplexity.ai/search"] = httpx.Response(429)
    with pytest.raises(httpx.HTTPStatusError):
        await server.web_search("query")


@pytest.mark.asyncio
async def test_fetch_http_error_is_preserved(fetch_client):
    """HTTP failures still propagate rather than becoming resource content."""
    url = "https://example.org/missing.pdf"
    fetch_client[url] = httpx.Response(404, headers={"content-type": "application/pdf"})
    with pytest.raises(httpx.HTTPStatusError):
        await server.web_fetch(url)
