"""Bounded PDF previews, rendered in a disposable process rather than the agent."""

import base64
from contextlib import closing
from dataclasses import dataclass
import io
import json
import math
import subprocess
import sys
import threading

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES = 10
MAX_SIDE = 2000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
RENDER_TIMEOUT = 20
_RENDER_LOCK = threading.Lock()


@dataclass
class PdfPreview:
    """Preview result: total source pages and ordered base64-encoded JPEG pages."""

    total_pages: int
    pages: list[str]


def render_pdf(raw: bytes) -> PdfPreview:
    """Render up to ten PDF pages at 144 DPI, bounded to 2000 pixels per side.

    Args:
        raw: Original PDF bytes, at most 25 MiB. No password is supplied.

    Returns:
        Source page count and JPEG images in page order, capped at 10 MiB of
        decoded images. Fewer images than source pages means a truncated preview.

    Raises:
        ValueError: Empty, oversized, invalid, encrypted, or unrenderable PDF,
            or the renderer exceeded its 20-second deadline/resource limits.

    This synchronous operation belongs in a worker thread when called by async
    code. Rendering processes are serialized to bound concurrent memory use.
    """
    if not raw or len(raw) > MAX_PDF_BYTES:
        raise ValueError("PDF must be nonempty and no larger than 25 MiB")
    try:
        with _RENDER_LOCK:
            result = subprocess.run([sys.executable, "-m", "session_manager.pdf"], input=raw,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    timeout=RENDER_TIMEOUT, check=True)
        return PdfPreview(**json.loads(result.stdout))
    except subprocess.TimeoutExpired:
        raise ValueError("PDF rendering exceeded the 20-second time limit") from None
    except (subprocess.CalledProcessError, OSError, ValueError, TypeError):
        raise ValueError("Cannot render PDF: invalid, password-protected, or resource limit exceeded") from None


def _render_pages(raw: bytes) -> PdfPreview:
    """Render validated-size bytes inside the isolated worker; PDF/parser errors propagate.

    Returns the total source page count and ordered JPEGs. Callers must not run
    PDFium concurrently in threads; the production entrypoint uses a process.
    """
    import pypdfium2 as pdfium

    images = []
    encoded_bytes = 0
    with pdfium.PdfDocument(raw) as document:
        document.init_forms()
        total = len(document)
        if not total:
            raise ValueError("PDF has no pages")
        for index in range(min(total, MAX_PAGES)):
            with closing(document[index]) as page:
                width, height = page.get_size()
                if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                    raise ValueError("Invalid PDF page dimensions")
                scale = min(2, MAX_SIDE / max(width, height))
                with closing(page.render(scale=scale, rev_byteorder=True)) as bitmap:
                    with bitmap.to_pil() as image, image.convert("RGB") as rgb:
                        output = io.BytesIO()
                        rgb.save(output, format="JPEG", quality=90)
            data = output.getvalue()
            if encoded_bytes + len(data) > MAX_IMAGE_BYTES:
                break
            images.append(base64.b64encode(data).decode("ascii"))
            encoded_bytes += len(data)
    if not images:
        raise ValueError("PDF preview exceeds image size limit")
    return PdfPreview(total_pages=total, pages=images)


def _worker() -> None:
    """Read PDF bytes from stdin and write a bounded preview JSON document to stdout.

    Applies Linux process memory/CPU limits before parsing untrusted PDF bytes.
    Parser errors terminate the worker; the parent returns a sanitized error.
    """
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    raw = sys.stdin.buffer.read(MAX_PDF_BYTES + 1)
    if not raw or len(raw) > MAX_PDF_BYTES:
        raise ValueError("Invalid PDF size")
    preview = _render_pages(raw)
    json.dump({"total_pages": preview.total_pages, "pages": preview.pages}, sys.stdout)


if __name__ == "__main__":
    _worker()
