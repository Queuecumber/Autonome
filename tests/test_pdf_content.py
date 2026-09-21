"""PDF resource previews, bounded rendering, and actual model image delivery."""

import asyncio
import base64
from contextlib import closing
import ctypes
import io
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from mcp.types import BlobResourceContents, EmbeddedResource, TextContent
from PIL import Image
import pypdfium2 as pdfium
import pytest

from session_manager import mcp, pdf
from session_manager.binaries import BinaryStore
from session_manager.event import Event
from session_manager.orchestrator import SessionOrchestrator, _media_user_message


def document(pages=2, width=612, height=792):
    """Create synthetic vector pages with colored squares and readable text using PDFium."""
    output = io.BytesIO()
    with pdfium.PdfDocument.new() as doc:
        for number in range(pages):
            with closing(doc.new_page(width, height)) as page:
                square = pdfium.raw.FPDFPageObj_CreateNewRect(20, 20, 100, 100)
                pdfium.raw.FPDFPageObj_SetFillColor(square, 220 if number % 2 == 0 else 20,
                                                  20 if number % 2 == 0 else 180, 20, 255)
                pdfium.raw.FPDFPath_SetDrawMode(square, pdfium.raw.FPDF_FILLMODE_WINDING, 0)
                pdfium.raw.FPDFPage_InsertObject(page, square)
                text = pdfium.raw.FPDFPageObj_NewTextObj(doc, b"Helvetica", 12)
                value = f"Page {number + 1}: PDF rendering test. Small text remains readable.\0".encode("utf-16-le")
                chars = (ctypes.c_ushort * (len(value) // 2)).from_buffer_copy(value)
                assert pdfium.raw.FPDFText_SetText(text, chars)
                pdfium.raw.FPDFPageObj_Transform(text, 1, 0, 0, 1, 20, height - 40)
                pdfium.raw.FPDFPage_InsertObject(page, text)
                page.gen_content()
        doc.save(output)
    return output.getvalue()


def resource(raw, mime="application/pdf", uri="imap://attachments/message/document"):
    """Wrap synthetic bytes in the actual MCP binary-resource types."""
    return EmbeddedResource(type="resource", resource=BlobResourceContents(
        uri=uri, mimeType=mime, blob=base64.b64encode(raw).decode("ascii")))


@pytest.mark.parametrize("mime", ["application/pdf", "APPLICATION/PDF; version=1.7",
                                  "application/octet-stream", "binary/octet-stream", None])
def test_pdf_resources_become_ordered_images(mime, tmp_path):
    """Declared and unspecified MIME PDFs render at readable resolution with the source URI intact."""
    block = resource(document(), mime)
    store = BinaryStore(tmp_path / "binaries")
    parts = mcp.mcp_content_to_openai([block], store)
    assert json.loads(parts[0]["text"]) == {"pdf": {
        "uri": str(block.resource.uri), "content_type": "application/pdf",
        "total_pages": 2, "rendered_pages": 2, "truncated": False}}
    assert len(parts) == 3
    for number, part in enumerate(parts[1:]):
        assert part["type"] == "input_image" and part["detail"] == "high"
        assert part["image_url"].startswith("data:image/jpeg;base64,")
        with Image.open(io.BytesIO(base64.b64decode(part["image_url"].split(",", 1)[1]))) as image:
            assert image.format == "JPEG" and image.mode == "RGB"
            assert image.size == (1224, 1584)
            red, green, _blue = image.getpixel((100, 1484))
            assert (red > green) == (number == 0)
            assert image.crop((35, 50, 750, 85)).convert("L").getextrema()[0] < 100
    assert not list(store.store_dir.iterdir())


@pytest.mark.parametrize("block", [
    TextContent(type="text", text="ordinary text"),
    resource(b"not PDF", "application/octet-stream"),
    resource(b"%PDF-1.7", "text/plain"),
    SimpleNamespace(type="resource", resource=None),
    SimpleNamespace(type="resource", resource=SimpleNamespace(blob="invalid!", mimeType=None)),
])
def test_pdf_detection_does_not_claim_other_content(block):
    """Ordinary text/images and invalid generic bytes retain their existing conversion paths."""
    assert not mcp.is_pdf_resource(block)


def test_page_limit_is_explicit():
    """Large documents expose truncation rather than implying all pages were read."""
    parts = mcp.mcp_content_to_openai([resource(document(pages=11))])
    metadata = json.loads(parts[0]["text"])["pdf"]
    assert metadata["total_pages"] == 11
    assert metadata["rendered_pages"] == 10
    assert metadata["truncated"]
    assert len(parts) == 11


def test_large_page_dimensions_are_bounded():
    """Oversized page boxes cannot allocate full-resolution posters."""
    preview = pdf._render_pages(document(pages=1, width=10000, height=5000))
    with Image.open(io.BytesIO(base64.b64decode(preview.pages[0]))) as image:
        assert image.size == (2000, 1000)


def test_page_rotation_is_preserved():
    """Rotated source pages retain their displayed orientation rather than being stretched."""
    output = io.BytesIO()
    with pdfium.PdfDocument(document(pages=1)) as doc:
        with closing(doc[0]) as page:
            page.set_rotation(90)
        doc.save(output)
    preview = pdf.render_pdf(output.getvalue())
    with Image.open(io.BytesIO(base64.b64decode(preview.pages[0]))) as image:
        assert image.size == (1584, 1224)


def test_image_byte_budget_truncates_or_fails_cleanly(monkeypatch):
    """A preview keeps complete pages only and cannot exceed its encoded-image budget."""
    raw = document()
    preview = pdf._render_pages(raw)
    first_size = len(base64.b64decode(preview.pages[0]))
    monkeypatch.setattr(pdf, "MAX_IMAGE_BYTES", first_size)
    truncated = pdf._render_pages(raw)
    assert truncated.total_pages == 2 and len(truncated.pages) == 1
    monkeypatch.setattr(pdf, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="image size limit"):
        pdf._render_pages(raw)


def test_invalid_pdf_returns_readable_error_without_losing_other_content():
    """A bad attachment cannot discard unrelated text from the same tool result."""
    parts = mcp.mcp_content_to_openai([TextContent(type="text", text="mail metadata"),
                                      resource(b"%PDF-invalid PRIVATE_CONTENT")])
    assert parts[0]["text"] == "mail metadata"
    error = json.loads(parts[1]["text"])["pdf"]
    assert "Cannot render PDF" in error["error"]
    assert "PRIVATE_CONTENT" not in json.dumps(parts)
    assert error["uri"] == "imap://attachments/message/document"


def test_input_size_and_encoding_limits(monkeypatch):
    """Input limits are enforced before process creation and invalid base64 is explained safely."""
    run = MagicMock()
    monkeypatch.setattr(pdf.subprocess, "run", run)
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", 8)
    for raw in (b"", b"x" * 9):
        with pytest.raises(ValueError, match="nonempty"):
            pdf.render_pdf(raw)
    monkeypatch.setattr(mcp, "MAX_PDF_BYTES", 8)
    assert "input limit" in mcp.mcp_content_to_openai([resource(b"x" * 30)])[0]["text"]
    block = resource(b"pdf")
    block.resource.blob = "invalid!"
    assert "Invalid base64 PDF content" in mcp.mcp_content_to_openai([block])[0]["text"]
    run.assert_not_called()


@pytest.mark.parametrize("error, message", [
    (subprocess.TimeoutExpired("worker", 20), "time limit"),
    (subprocess.CalledProcessError(1, "worker", stderr=b"PRIVATE_ERROR"), "Cannot render PDF"),
    (OSError("PRIVATE_ERROR"), "Cannot render PDF"),
])
def test_worker_errors_are_sanitized(monkeypatch, error, message):
    """Timeout and process failures never expose parser output or source bytes."""
    monkeypatch.setattr(pdf.subprocess, "run", MagicMock(side_effect=error))
    with pytest.raises(ValueError, match=message) as caught:
        pdf.render_pdf(b"%PDF-test")
    assert "PRIVATE_ERROR" not in str(caught.value)


@pytest.mark.parametrize("output", [b"not JSON", b"[]"])
def test_invalid_worker_output_is_sanitized(monkeypatch, output):
    """Malformed worker output produces the same bounded failure as a renderer crash."""
    monkeypatch.setattr(pdf.subprocess, "run", MagicMock(return_value=SimpleNamespace(stdout=output)))
    with pytest.raises(ValueError, match="Cannot render PDF"):
        pdf.render_pdf(b"%PDF-test")


@pytest.mark.parametrize("size", [(0, 10), (-1, 20), (float("inf"), 100), (float("nan"), 100)])
def test_invalid_page_geometry_is_rejected(monkeypatch, size):
    """Non-finite or empty page boxes cannot reach the bitmap allocator."""
    doc = MagicMock()
    doc.__enter__.return_value = doc
    doc.__len__.return_value = 1
    doc.__getitem__.return_value.get_size.return_value = size
    monkeypatch.setattr(pdfium, "PdfDocument", MagicMock(return_value=doc))
    with pytest.raises(ValueError, match="dimensions"):
        pdf._render_pages(b"test")


def test_empty_document_rejected(monkeypatch):
    """A document with no pages cannot produce a misleading successful preview."""
    doc = MagicMock()
    doc.__enter__.return_value = doc
    doc.__len__.return_value = 0
    monkeypatch.setattr(pdfium, "PdfDocument", MagicMock(return_value=doc))
    with pytest.raises(ValueError, match="no pages"):
        pdf._render_pages(b"test")


@pytest.mark.parametrize("raw", [b"", b"x" * 9, b"pdf"])
def test_worker_limits_and_protocol(monkeypatch, raw):
    """The worker sets CPU/memory limits before accepting input and emits structured results."""
    import resource as limits

    set_limit = MagicMock()
    monkeypatch.setattr(limits, "setrlimit", set_limit)
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", 8)
    monkeypatch.setattr(pdf.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw)))
    output = io.StringIO()
    monkeypatch.setattr(pdf.sys, "stdout", output)
    render = MagicMock(return_value=pdf.PdfPreview(1, ["page"]))
    monkeypatch.setattr(pdf, "_render_pages", render)
    if raw == b"pdf":
        pdf._worker()
        assert json.loads(output.getvalue()) == {"total_pages": 1, "pages": ["page"]}
        render.assert_called_once_with(raw)
    else:
        with pytest.raises(ValueError, match="size"):
            pdf._worker()
        render.assert_not_called()
    assert set_limit.call_args_list[0].args == (limits.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    assert set_limit.call_args_list[1].args == (limits.RLIMIT_CPU, (15, 15))


@pytest.fixture
async def orchestrator(tmp_path, monkeypatch):
    """Provide an isolated session manager with a synthetic PDF-returning MCP tool."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    orch = SessionOrchestrator({"model": {"name": "test"}, "session": {"debounce_seconds": 0},
        "binaries": {"store": str(tmp_path / "binaries")}}, tmp_path / "sessions")
    conn = SimpleNamespace(binary_params={}, call_tool=AsyncMock(return_value=[resource(document())]))
    orch._tool_to_mcp["fetch_pdf"] = conn
    yield orch
    await orch.llm.close()


@pytest.mark.asyncio
async def test_pdf_tool_result_reaches_model_as_images(orchestrator):
    """The next completion gets ordered high-detail image content, not PDF/base64 text."""
    calls = []

    async def respond(kwargs, cancel):
        """Request the PDF once, then record the actual next-call image payload."""
        calls.append(kwargs)
        if len(calls) == 1:
            return {"content": "", "tool_calls": [{"id": "pdf-1", "type": "function",
                    "function": {"name": "fetch_pdf", "arguments": "{}"}}]}, None
        return {"content": "read the pages", "tool_calls": []}, None

    orchestrator._stream_response = respond
    assert await orchestrator.handle_event(Event(text="read the PDF")) == "read the pages"
    messages = calls[1]["messages"]
    images = messages[-1]["content"]
    assert len(images) == 2
    assert all(image["type"] == "image_url" and image["image_url"]["detail"] == "high" for image in images)
    assert all(image["image_url"]["url"].startswith("data:image/jpeg;base64,") for image in images)
    assert json.loads(messages[-2]["content"])["pdf"]["total_pages"] == 2
    assert "data:image/" not in json.dumps(orchestrator.session.load("main"))
    assert "imap://attachments/message/document" in json.dumps(orchestrator.session.load("main"))


@pytest.mark.asyncio
async def test_rendering_does_not_block_event_loop(orchestrator, monkeypatch):
    """A renderer waiting in its worker thread does not prevent event-loop progress."""
    import threading

    started, release = threading.Event(), threading.Event()

    def render(raw):
        """Wait for an independent async task to permit conversion to finish."""
        started.set()
        assert release.wait(timeout=2)
        return pdf.PdfPreview(1, ["image"])

    monkeypatch.setattr(mcp, "render_pdf", render)
    task = asyncio.create_task(orchestrator._execute_tool_call("id", "fetch_pdf", "{}"))
    try:
        async with asyncio.timeout(1):
            while not started.is_set():
                await asyncio.sleep(0.005)
        assert not task.done()
    finally:
        release.set()
        result, media = await task
    assert len(media) == 1 and "total_pages" in result["output"]


@pytest.mark.asyncio
async def test_pdf_preview_errors_retain_other_tool_content(orchestrator):
    """The async tool path preserves ordinary text when a PDF cannot be rendered."""
    orchestrator._tool_to_mcp["fetch_pdf"].call_tool.return_value = [
        TextContent(type="text", text="attachment metadata"), resource(b"invalid PDF")]
    result, media = await orchestrator._execute_tool_call("id", "fetch_pdf", "{}")
    assert result["output"].startswith("attachment metadata\n")
    assert "Cannot render PDF" in result["output"]
    assert not media


def test_image_detail_is_preserved():
    """PDF readability hints survive the Responses-style to Chat Completions conversion."""
    message = _media_user_message([{"content": [{"type": "input_image", "detail": "high", "image_url": "data"}]}])
    assert message["content"][0]["image_url"] == {"url": "data", "detail": "high"}
