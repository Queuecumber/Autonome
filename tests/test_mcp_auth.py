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
