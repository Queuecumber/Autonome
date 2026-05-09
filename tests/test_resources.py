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
    with pytest.raises(ValueError, match="no scheme"):
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


def test_resource_generic_mime_falls_back_to_filetype(tmp_path):
    """When the resource declares a generic mime, mcp_content_to_openai
    re-detects from the bytes so images surface correctly even when the
    server-side template's mime_type is `application/octet-stream`.
    """
    store = BinaryStore(store_dir=tmp_path / "bins", retention_days=30)
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    block = _resource_block("mxc://srv/abc", png_magic, "application/octet-stream")

    parts = mcp_content_to_openai([block], store=store)
    img = next((p for p in parts if p.get("type") == "input_image"), None)
    assert img is not None
    assert "image/png" in img["image_url"]


def test_resource_non_image_describes_via_uri(tmp_path):
    """Non-visual binary resources: describe by uri + size, no caching."""
    import json as _json
    store = BinaryStore(store_dir=tmp_path / "bins", retention_days=30)
    block = _resource_block("mxc://srv/doc", b"%PDF-1.4 ...", "application/pdf")

    parts = mcp_content_to_openai([block], store=store)
    text_part = next((p for p in parts if p.get("type") == "input_text"), None)
    assert text_part is not None
    payload = _json.loads(text_part["text"])
    assert payload["uri"] == "mxc://srv/doc"
    assert payload["mimeType"] == "application/pdf"
    assert not list(store.store_dir.iterdir())


# ── platform_mcp pointer:// resource ─────────────────────


@pytest.mark.asyncio
async def test_platform_mcp_pointer_serves_bytes(tmp_path):
    """The platform's in-process FastMCP exposes the binary cache as
    `pointer://` resources. read_pointer returns whatever the BinaryStore
    has for that name."""
    store = BinaryStore(store_dir=tmp_path / "bins", retention_days=30)
    pointer = store.save(b"cached bytes", "text/plain", filename="x.txt")
    platform_mcp.binary_store = store
    try:
        result = await platform_mcp.read_pointer(pointer)
        assert result == b"cached bytes"
    finally:
        platform_mcp.binary_store = None


# ── matrix-adapter mxc URI ───────────────────────────────


def test_matrix_attach_query_inlines_encryption_params():
    """`_attach_query` builds an mxc URL with encryption params in the
    query string so resources are self-contained."""
    from matrix_adapter.model import _attach_query, _split_query
    base = "mxc://server.com/abc123"
    enriched = _attach_query(base, {"k": "KEY", "iv": "IV", "hash": "HASH"})
    assert enriched.startswith("mxc://server.com/abc123?")
    assert "k=KEY" in enriched
    bare, params = _split_query(enriched)
    assert bare == base
    assert params == {"k": "KEY", "iv": "IV", "hash": "HASH"}


def test_matrix_split_query_round_trip_with_special_chars():
    """Round-trip preserves base64 chars (including `+`, `/`, `=`) which
    get URL-encoded on the way out and decoded on the way in."""
    from matrix_adapter.model import _attach_query, _split_query
    params = {"k": "Ezi-RoXOM_HvEm", "iv": "y/UHn==", "hash": "abc+def/123=="}
    enriched = _attach_query("mxc://srv/id", params)
    _, parsed = _split_query(enriched)
    assert parsed == params


def test_matrix_attach_query_skips_empty_dict():
    from matrix_adapter.model import _attach_query
    assert _attach_query("mxc://srv/id", {}) == "mxc://srv/id"
