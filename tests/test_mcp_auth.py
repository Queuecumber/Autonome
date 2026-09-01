"""MCP server auth: per-server headers with env-sourced secrets.

Servers like Home Assistant need a bearer token. The token must not live in
agent.yaml (it sits on a PVC and gets backed up), so header values expand
${VAR} from the environment at connect time.
"""

import pytest

from session_manager.mcp import MCPConnection, parse_server_spec


def test_plain_url_shorthand_still_works():
    """Existing configs are untouched — a bare URL means no headers."""
    assert parse_server_spec("memory", "http://memory-mcp:8001/mcp") == (
        "http://memory-mcp:8001/mcp", None)


def test_mapping_with_headers(monkeypatch):
    monkeypatch.setenv("HA_TOKEN", "secret-token")
    url, headers = parse_server_spec("home_assistant", {
        "url": "http://homeassistant.local:8123/mcp",
        "headers": {"Authorization": "Bearer ${HA_TOKEN}"},
    })
    assert url == "http://homeassistant.local:8123/mcp"
    assert headers == {"Authorization": "Bearer secret-token"}


def test_multiple_headers_and_partial_interpolation(monkeypatch):
    """Only ${VAR} spans are substituted; surrounding text is preserved,
    and non-Authorization headers work too (some servers use X-API-Key)."""
    monkeypatch.setenv("K", "abc")
    _, headers = parse_server_spec("svc", {
        "url": "http://x/mcp",
        "headers": {"X-API-Key": "${K}", "X-Client": "autonome/${K}/v1"},
    })
    assert headers == {"X-API-Key": "abc", "X-Client": "autonome/abc/v1"}


def test_missing_env_var_raises_naming_the_variable(monkeypatch):
    """Fail at startup rather than sending a literal '${HA_TOKEN}', which
    would surface much later as an opaque 401 from the server."""
    monkeypatch.delenv("HA_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HA_TOKEN"):
        parse_server_spec("home_assistant", {
            "url": "http://x/mcp",
            "headers": {"Authorization": "Bearer ${HA_TOKEN}"},
        })


def test_mapping_without_headers_is_fine():
    assert parse_server_spec("s", {"url": "http://x/mcp"}) == ("http://x/mcp", None)


def test_empty_headers_normalise_to_none():
    assert parse_server_spec("s", {"url": "http://x/mcp", "headers": {}})[1] is None


@pytest.mark.parametrize("spec, match", [
    ({"headers": {"a": "b"}}, "missing required 'url'"),
    ({"url": ""}, "missing required 'url'"),
    ({"url": "http://x", "headers": "Bearer z"}, "expected a mapping"),
    (["http://x"], "expected a URL string or a mapping"),
])
def test_malformed_specs_are_rejected(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_server_spec("bad", spec)


def test_connection_carries_headers_through():
    conn = MCPConnection("ha", "http://x/mcp", headers={"Authorization": "Bearer t"})
    assert conn.headers == {"Authorization": "Bearer t"}
    assert MCPConnection("plain", "http://x/mcp").headers is None


# ── Optional server capabilities ─────────────────────────
#
# Resources are optional in the protocol. Home Assistant serves tools only,
# and asking it to enumerate resource templates fails the whole connection
# with "Method not found" unless we check first.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from mcp.shared.exceptions import McpError
from mcp.types import METHOD_NOT_FOUND, ErrorData


def _conn_with(capabilities, session):
    conn = MCPConnection("srv", "http://x/mcp")
    conn.capabilities = capabilities
    conn.session = session
    return conn


@pytest.mark.asyncio
async def test_resource_enumeration_skipped_when_not_advertised():
    """A tools-only server is never asked to enumerate resources."""
    session = MagicMock()
    session.list_resources = AsyncMock()
    session.list_resource_templates = AsyncMock()
    conn = _conn_with(SimpleNamespace(resources=None), session)

    assert await conn.list_resources() == []
    assert await conn.list_resource_templates() == []
    session.list_resources.assert_not_called()
    session.list_resource_templates.assert_not_called()


@pytest.mark.asyncio
async def test_method_not_found_is_tolerated_when_advertised():
    """Servers can advertise `resources` and still not implement the
    templates half — the two are separate methods."""
    err = McpError(ErrorData(code=METHOD_NOT_FOUND, message="Method not found"))
    session = MagicMock()
    session.list_resource_templates = AsyncMock(side_effect=err)
    session.list_resources = AsyncMock(side_effect=err)
    conn = _conn_with(SimpleNamespace(resources=object()), session)

    assert await conn.list_resource_templates() == []
    assert await conn.list_resources() == []


@pytest.mark.asyncio
async def test_other_mcp_errors_still_propagate():
    """Only method-not-found is swallowed; a real failure must surface."""
    err = McpError(ErrorData(code=-32000, message="boom"))
    session = MagicMock()
    session.list_resource_templates = AsyncMock(side_effect=err)
    conn = _conn_with(SimpleNamespace(resources=object()), session)

    with pytest.raises(McpError, match="boom"):
        await conn.list_resource_templates()


@pytest.mark.asyncio
async def test_templates_returned_when_supported():
    tmpl = SimpleNamespace(uriTemplate="mxc://{server}/{id}")
    session = MagicMock()
    session.list_resource_templates = AsyncMock(
        return_value=SimpleNamespace(resourceTemplates=[tmpl]))
    conn = _conn_with(SimpleNamespace(resources=object()), session)

    assert await conn.list_resource_templates() == [tmpl]
