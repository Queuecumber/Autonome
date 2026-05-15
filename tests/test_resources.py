"""Tests for the resource-bridge architecture.

Covers:
- Scheme registration + dispatch on the orchestrator (`_scheme_to_mcp`,
  `resolve_uri`).
- Resource-content rendering in `mcp_content_to_openai`: images surface
  as input_image, generic-mime fallback to filetype.guess, no
  cache+pointer round-trip.
- platform_mcp's `pointer://` resource (binary cache exposed as MCP
  resource).
- matrix-adapter's `mxc://{server}/{media_id}{?k,iv,hash}` template:
  RFC 6570 query expansion captures encryption parameters.
- Encryption-info-in-URL helpers in matrix-adapter.
"""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from session_manager import platform_mcp
from session_manager.binaries import BinaryStore
from session_manager.mcp import mcp_content_to_openai
from session_manager.orchestrator import SessionOrchestrator


# ── Orchestrator scheme map / resolve_uri ────────────────


@pytest.fixture
def orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return SessionOrchestrator(
        config={
            "model": {"name": "test"},
            "session": {"max_history_tokens": 100000},
            "binaries": {"store": str(tmp_path / "bins"), "retention_days": 30},
        },
        session_dir=tmp_path / "sessions",
    )


def _mock_template(uri_template: str) -> MagicMock:
    t = MagicMock()
    t.uriTemplate = uri_template
    return t


def _mock_conn(name: str, templates: list[str]) -> MagicMock:
    conn = MagicMock()
    conn.name = name
    conn.list_resource_templates = AsyncMock(
        return_value=[_mock_template(t) for t in templates]
    )
    return conn


@pytest.mark.asyncio
async def test_register_schemes_picks_scheme_from_template(orchestrator):
    conn = _mock_conn("matrix", ["mxc://{server}/{media_id}{?k,iv,hash}"])
    await orchestrator._register_schemes(conn)
    assert orchestrator._scheme_to_mcp["mxc"] is conn


@pytest.mark.asyncio
async def test_register_schemes_collision_raises(orchestrator):
    """Two MCP servers claiming the same scheme is a config error."""
    a = _mock_conn("a", ["pointer://{name}"])
    b = _mock_conn("b", ["pointer://{name}"])
    await orchestrator._register_schemes(a)
    with pytest.raises(RuntimeError, match="claimed by both"):
        await orchestrator._register_schemes(b)


@pytest.mark.asyncio
async def test_register_schemes_idempotent_for_same_conn(orchestrator):
    """Re-registering the same conn for the same scheme is a no-op."""
    a = _mock_conn("a", ["pointer://{name}"])
    await orchestrator._register_schemes(a)
    await orchestrator._register_schemes(a)
    assert orchestrator._scheme_to_mcp["pointer"] is a


@pytest.mark.asyncio
async def test_resolve_uri_dispatches_blob(orchestrator):
    blob = MagicMock()
    blob.blob = base64.b64encode(b"hello").decode()
    blob.text = None
    conn = _mock_conn("matrix", ["mxc://{server}/{media_id}"])
    conn.read_resource = AsyncMock(return_value=[blob])
    await orchestrator._register_schemes(conn)

    result = await orchestrator.resolve_uri("mxc://server.com/abc")
    assert result == b"hello"


@pytest.mark.asyncio
async def test_resolve_uri_dispatches_text(orchestrator):
    text = MagicMock(spec=["text"])
    text.text = "hello text"
    conn = _mock_conn("workspace-fs", ["workspace:///{path*}"])
    conn.read_resource = AsyncMock(return_value=[text])
    await orchestrator._register_schemes(conn)

    result = await orchestrator.resolve_uri("workspace:///note.md")
    assert result == b"hello text"


@pytest.mark.asyncio
async def test_resolve_uri_unknown_scheme(orchestrator):
    with pytest.raises(ValueError, match="No MCP server registered"):
        await orchestrator.resolve_uri("ftp://example.com/file")


@pytest.mark.asyncio
async def test_resolve_uri_no_scheme(orchestrator):
    with pytest.raises(ValueError, match="No MCP server registered"):
        await orchestrator.resolve_uri("just-a-name")


# ── mcp_content_to_openai resource branch ────────────────


def _resource_block(uri: str, blob: bytes, mime: str | None) -> MagicMock:
    """Build a content block matching what session.read_resource returns."""
    block = MagicMock()
    block.type = "resource"
    res = MagicMock()
    res.uri = uri
    res.blob = base64.b64encode(blob).decode()
    res.text = None
    res.mimeType = mime
    block.resource = res
    return block


def test_resource_image_emits_input_image_no_pointer(tmp_path):
    """Image resource: agent gets the visual but no new pointer descriptor.

    Pre-fix this would also emit a pointer descriptor pushing the agent
    into a redundant second read.
    """
    store = BinaryStore(store_dir=tmp_path / "bins", retention_days=30)
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    block = _resource_block("mxc://srv/abc", png_magic, "image/png")

    parts = mcp_content_to_openai([block], store=store)

    types = [p.get("type") for p in parts]
    assert "input_image" in types
    # No pointer JSON should be emitted for resource reads.
    pointer_descriptors = [
        p for p in parts
        if p.get("type") == "input_text" and '"pointer"' in p.get("text", "")
    ]
    assert pointer_descriptors == []
    # Nothing should be persisted to BinaryStore.
    assert not list(store.store_dir.iterdir())


def test_resource_opaque_binary_raises(tmp_path):
    """A resource that's neither image nor text can't be inlined as model
    context — raise so the agent knows to use a tool that handles the URI."""
    store = BinaryStore(store_dir=tmp_path / "bins", retention_days=30)
    block = _resource_block("mxc://srv/doc", b"%PDF-1.4 ...", "application/pdf")
    with pytest.raises(ValueError, match="Cannot inline"):
        mcp_content_to_openai([block], store=store)
    assert not list(store.store_dir.iterdir())


# ── platform_mcp pointer:// resource ─────────────────────


@pytest.mark.asyncio
async def test_platform_mcp_pointer_serves_bytes(tmp_path):
    """The platform's in-process FastMCP exposes the binary cache as
    `pointer://` resources. read_pointer returns a ResourceResult with the
    bytes and the stored mime."""
    store = BinaryStore(store_dir=tmp_path / "bins", retention_days=30)
    pointer = store.save(b"cached bytes", "text/plain", filename="x.txt")
    platform_mcp.binary_store = store
    try:
        result = await platform_mcp.read_pointer(pointer)
        content = result.contents[0]
        assert content.content == b"cached bytes"
        assert content.mime_type
    finally:
        platform_mcp.binary_store = None


def test_platform_mcp_requires_orchestrator_when_unset():
    """Bridge tools fail fast with a clear error if main.py forgot to wire
    them up."""
    platform_mcp.orchestrator = None
    with pytest.raises(RuntimeError, match="orchestrator not initialized"):
        platform_mcp._require_orchestrator()


def test_platform_mcp_requires_binary_store_when_unset():
    platform_mcp.binary_store = None
    with pytest.raises(RuntimeError, match="binary_store not initialized"):
        platform_mcp._require_binary_store()


# ── platform_mcp bridge tools ────────────────────────────


def _mock_resource(uri: str, name: str = "n", mime: str | None = None) -> MagicMock:
    r = MagicMock()
    r.uri = uri
    r.name = name
    r.description = None
    r.mimeType = mime
    return r


@pytest.mark.asyncio
async def test_resources_list_aggregates_across_connections():
    """resources_list flattens resources from every connected MCP server."""
    matrix_conn = MagicMock()
    matrix_conn.name = "matrix"
    matrix_conn.list_resources = AsyncMock(return_value=[
        _mock_resource("mxc://srv/a", mime="image/png"),
    ])
    workspace_conn = MagicMock()
    workspace_conn.name = "workspace-fs"
    workspace_conn.list_resources = AsyncMock(return_value=[
        _mock_resource("workspace:///b.md", mime="text/markdown"),
    ])
    orch = MagicMock()
    orch.mcp_connections = {"matrix": matrix_conn, "workspace-fs": workspace_conn}

    platform_mcp.orchestrator = orch
    try:
        out = await platform_mcp.resources_list()
    finally:
        platform_mcp.orchestrator = None

    uris = {entry.uri for entry in out}
    assert uris == {"mxc://srv/a", "workspace:///b.md"}


@pytest.mark.asyncio
async def test_resources_list_filters_by_scheme():
    matrix_conn = MagicMock()
    matrix_conn.name = "matrix"
    matrix_conn.list_resources = AsyncMock(return_value=[
        _mock_resource("mxc://srv/a"),
        _mock_resource("workspace:///stray.txt"),
    ])
    orch = MagicMock()
    orch.mcp_connections = {"matrix": matrix_conn}

    platform_mcp.orchestrator = orch
    try:
        out = await platform_mcp.resources_list(scheme="mxc")
    finally:
        platform_mcp.orchestrator = None

    assert [e.uri for e in out] == ["mxc://srv/a"]


@pytest.mark.asyncio
async def test_resources_template_list_aggregates_templates():
    conn = MagicMock()
    conn.name = "matrix"
    tmpl = MagicMock()
    tmpl.uriTemplate = "mxc://{server}/{media_id}{?k,iv,hash}"
    tmpl.name = "mxc"
    tmpl.description = None
    tmpl.mimeType = None
    conn.list_resource_templates = AsyncMock(return_value=[tmpl])
    orch = MagicMock()
    orch.mcp_connections = {"matrix": conn}

    platform_mcp.orchestrator = orch
    try:
        out = await platform_mcp.resources_template_list()
    finally:
        platform_mcp.orchestrator = None

    assert out[0].uri == "mxc://{server}/{media_id}{?k,iv,hash}"
    assert out[0].server == "matrix"


@pytest.mark.asyncio
async def test_resources_read_dispatches_by_scheme():
    """resources_read uses the orchestrator's scheme map to pick the right
    connection, then wraps each returned ResourceContents in an
    EmbeddedResource."""
    from mcp.types import TextResourceContents
    contents = [TextResourceContents(uri="mxc://srv/abc", text="hi", mimeType="text/plain")]
    conn = MagicMock()
    conn.read_resource = AsyncMock(return_value=contents)
    orch = MagicMock()
    orch._scheme_to_mcp = {"mxc": conn}

    platform_mcp.orchestrator = orch
    try:
        result = await platform_mcp.resources_read("mxc://srv/abc")
    finally:
        platform_mcp.orchestrator = None

    conn.read_resource.assert_awaited_once_with("mxc://srv/abc")
    assert len(result) == 1
    assert result[0].type == "resource"


@pytest.mark.asyncio
async def test_resources_read_rejects_uri_without_scheme():
    orch = MagicMock()
    orch._scheme_to_mcp = {}
    platform_mcp.orchestrator = orch
    try:
        with pytest.raises(ValueError, match="No MCP server registered"):
            await platform_mcp.resources_read("not-a-uri")
    finally:
        platform_mcp.orchestrator = None


@pytest.mark.asyncio
async def test_resources_read_rejects_unknown_scheme():
    orch = MagicMock()
    orch._scheme_to_mcp = {}
    platform_mcp.orchestrator = orch
    try:
        with pytest.raises(ValueError, match="No MCP server registered"):
            await platform_mcp.resources_read("unknown://thing")
    finally:
        platform_mcp.orchestrator = None


# ── matrix-adapter mxc URI ───────────────────────────────


def test_matrix_extract_media_inlines_mime_unencrypted():
    """_extract_media adds the sender-declared mime to plain mxc URLs so the
    resource handler downstream can return it via ResourceResult."""
    from matrix_adapter.model import MatrixClient

    event = MagicMock()
    event.source = {"content": {"info": {"mimetype": "text/markdown"}}}
    event.url = "mxc://srv/abc"
    event.body = "doc.md"

    client = MatrixClient.__new__(MatrixClient)
    att = client._extract_media(event)
    assert "mime=text%2Fmarkdown" in att.url
    assert att.content_type == "text/markdown"


def test_matrix_extract_media_inlines_mime_encrypted():
    """Same inlining for encrypted attachments, alongside k/iv/hash."""
    from matrix_adapter.model import MatrixClient

    event = MagicMock()
    event.source = {"content": {
        "info": {"mimetype": "image/png"},
        "file": {
            "url": "mxc://srv/enc",
            "key": {"k": "KEY"},
            "iv": "IV",
            "hashes": {"sha256": "HASH"},
        },
    }}
    # nio's RoomEncryptedMedia sets event.url from content.file.url for both
    # encrypted and unencrypted attachments — see nio/events/room_events.py.
    event.url = "mxc://srv/enc"
    event.body = "img.png"

    client = MatrixClient.__new__(MatrixClient)
    att = client._extract_media(event)
    assert "mime=image%2Fpng" in att.url
    assert "k=KEY" in att.url


@pytest.mark.asyncio
async def test_mxc_resource_returns_resourceresult_with_mime():
    """matrix-adapter wraps downloaded bytes in a ResourceResult carrying
    the sender-declared mime from the URI query param."""
    from matrix_adapter import server as matrix_server
    matrix_server.client = MagicMock()
    matrix_server.client.download_attachment = AsyncMock(return_value=b"# notes")

    result = await matrix_server.mxc_resource(
        server="srv", media_id="abc", mime="text/markdown",
    )
    content = result.contents[0]
    assert content.content == b"# notes"
    assert content.mime_type == "text/markdown"


@pytest.mark.asyncio
async def test_mxc_resource_defaults_mime_when_missing():
    """No `mime` in URI → fall back to octet-stream."""
    from matrix_adapter import server as matrix_server
    matrix_server.client = MagicMock()
    matrix_server.client.download_attachment = AsyncMock(return_value=b"...")

    result = await matrix_server.mxc_resource(server="srv", media_id="abc")
    assert result.contents[0].mime_type == "application/octet-stream"


# ── signal-adapter signal:// resource ────────────────────


@pytest.mark.asyncio
async def test_signal_attachment_resource_returns_resourceresult_with_mime():
    """signal-adapter returns ResourceResult with signal-cli's content_type."""
    from signal_adapter import server as signal_server
    from signal_adapter.model import Attachment

    att = Attachment(
        id="x", content_type="image/jpeg",
        content_base64=base64.b64encode(b"jpeg bytes").decode(),
    )
    signal_server.client = MagicMock()
    signal_server.client.fetch_attachment = AsyncMock(return_value=att)

    result = await signal_server.signal_attachment_resource("x")
    content = result.contents[0]
    assert content.content == b"jpeg bytes"
    assert content.mime_type == "image/jpeg"


# ── mcp_content_to_openai text-mime branch ───────────────


def test_resource_text_blob_decodes_to_input_text():
    """Text-mime blob resources decode UTF-8 and emit as input_text."""
    body = b"# Notes\n\nThe agent should be able to read this."
    block = _resource_block("workspace:///notes.md", body, "text/markdown")
    parts = mcp_content_to_openai([block])
    decoded = [p["text"] for p in parts if p.get("type") == "input_text"]
    assert any("# Notes" in t for t in decoded)


def test_resource_text_blob_invalid_utf8_raises():
    """If MIME claims text but bytes aren't valid UTF-8, the producer is
    lying; surface that as an error rather than silently corrupting."""
    block = _resource_block("workspace:///bad.txt", b"\xff\xfe\xfd not utf-8", "text/plain")
    with pytest.raises(UnicodeDecodeError):
        mcp_content_to_openai([block])
