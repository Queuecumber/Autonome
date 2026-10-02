# PDF Image Content

When an MCP tool returns a PDF binary resource, including an attachment fetched
through `resources_read` or a PDF downloaded with system-mcp's `web_fetch`,
session-manager renders its pages into image content for the model. `web_fetch`
returns the PDF bytes as an embedded MCP resource with the final response URL
as its URI; `max_chars` applies only to text responses, not PDF downloads.
This follows the existing attachment flow: receiving an event containing a
resource URI does not itself download the attachment.

Declared `application/pdf` resources are supported, including MIME parameters.
Resources with missing or generic binary MIME types are also recognized when
their bytes start with the PDF signature. Ordinary text and image handling is
unchanged.

Each PDF produces a text descriptor with its original URI, total page count,
rendered page count, and `truncated` status, followed by the first rendered pages
in document order. Pages are RGB JPEGs at quality 90, rendered at 144 DPI and
scaled down when necessary to fit within 2000 pixels per side. They reach Chat
Completions as `image_url` content with `detail: high`; the model must support
image inputs.

Limits per document:

- 25 MiB input PDF. `web_fetch` also rejects PDF responses exceeding 25 MiB
  rather than returning a partial document.
- First 10 pages maximum.
- 10 MiB total JPEG bytes, keeping complete pages only.
- 20-second renderer timeout, with a 15-second CPU limit and 512 MiB address-space
  limit in the Linux rendering process.

Page or image-byte limits set `truncated: true`; omitted pages were not read.
Invalid, password-protected, or resource-exhausting PDFs produce a safe preview
error alongside the original URI. Other content in the tool result is retained.
This feature does not provide password entry, OCR text extraction, or page-range
selection; larger documents need splitting or a separate document tool.

Rendering uses [pypdfium2](https://pypdfium2.readthedocs.io/en/stable/python_api.html)
and Pillow, installed with session-manager. PDFium runs in a disposable process,
with one renderer at a time per session-manager process; waiting happens outside
the event loop. Process limits contain crashes and excessive resource use but
are not a security sandbox. Embedded attachments and links are not fetched.

Original PDF resources are not replaced or duplicated in the binary store.
Generated images are sent in the current turn, not persisted as base64 in session
history. The original URI and preview metadata remain in history so the document
can be fetched again through its owning MCP.

Deploy the updated session-manager image and restart session-manager for PDF
resource rendering. PDF URLs fetched through `web_fetch` additionally require
the updated system-mcp image and a system-mcp restart; older system-mcp versions
return decoded, possibly truncated PDF bytes as text. No Helm values or adapter
changes are required.
