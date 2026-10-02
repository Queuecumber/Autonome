"""The browser_request tool forwards the full curl_cffi API and shapes responses."""

import base64
import datetime
import json

import pytest
from curl_cffi import CurlHttpVersion
from curl_cffi.requests import Cookies, Headers
from fastmcp import Client

from system_mcp import server


class FakeResponse:
    """Stand-in for a curl_cffi Response with the attributes the tool reads."""

    def __init__(self, *, status_code=200, reason="OK", url="https://example.org/",
                 content=b"", content_type="text/plain", headers=None, cookies=None,
                 history=(), http_version=2, elapsed=0.25):
        self.status_code = status_code
        self.reason = reason
        self.ok = status_code < 400
        self.url = url
        self.content = content
        merged = {"content-type": content_type}
        merged.update(headers or {})
        self.headers = Headers(merged)
        self.cookies = Cookies(cookies or {})
        self.history = list(history)
        self.http_version = http_version
        self.elapsed = datetime.timedelta(seconds=elapsed)

    @property
    def text(self):
        """Decode the body as UTF-8, like curl_cffi's default encoding."""
        return self.content.decode("utf-8")


class FakeBrowser:
    """Stand-in for the curl_cffi AsyncSession; records requests, replays responses."""

    def __init__(self):
        self.cookies = Cookies()
        self.calls = []
        self.queue = []

    def enqueue(self, response):
        self.queue.append(response)

    async def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.queue.pop(0)


@pytest.fixture
def browser(monkeypatch):
    """Replace the shared browser session with a recording fake."""
    fake = FakeBrowser()
    monkeypatch.setattr(server, "_browser", fake)
    return fake


async def test_defaults_forwarded(browser):
    """A plain GET sends browser defaults and omits unset options."""
    browser.enqueue(FakeResponse(content=b"hi"))
    result = await server.browser_request("https://example.org/")

    call = browser.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "https://example.org/"
    assert call["timeout"] == 30
    assert call["allow_redirects"] is True
    assert call["max_redirects"] == 30
    assert call["verify"] is True
    assert call["impersonate"] == "chrome"
    assert call["default_headers"] is True
    assert call["default_encoding"] == "utf-8"
    assert call["max_recv_speed"] == 0
    assert call["discard_cookies"] is False
    for unset in ("params", "headers", "cookies", "data", "json", "auth", "proxy",
                  "proxies", "proxy_auth", "referer", "accept_encoding", "ja3",
                  "akamai", "extra_fp", "http_version", "interface", "dns",
                  "doh_url", "cert", "quote"):
        assert unset not in call
    assert result["status"] == 200
    assert result["body"] == "hi"


async def test_full_options_forwarded(browser):
    """Every curl_cffi option reaches the session; auth pairs become tuples."""
    browser.enqueue(FakeResponse(status_code=404, reason="Not Found"))
    await server.browser_request(
        "https://example.org/login",
        method="POST",
        params={"next": "/home"},
        headers={"X-Custom": "1"},
        cookies={"one-off": "yes"},
        data={"user": "u"},
        auth=("u", "p"),
        timeout=5.5,
        allow_redirects=False,
        max_redirects=3,
        proxy="http://proxy:8080",
        proxy_auth=("pu", "pp"),
        verify=False,
        referer="https://example.org/start",
        accept_encoding="gzip",
        impersonate="chrome124",
        default_headers=False,
        ja3="ja3-string",
        akamai="akamai-string",
        extra_fp={"tls_signature_algorithms": ["ecdsa_secp256r1_sha256"]},
        http_version="1.1",
        default_encoding="latin-1",
        interface="eth0",
        dns=["1.1.1.1"],
        doh_url="https://1.1.1.1/dns-query",
        cert=("/tmp/cert.pem", "/tmp/key.pem"),
        max_recv_speed=1024,
        quote=False,
        discard_cookies=True,
    )

    call = browser.calls[0]
    assert call["method"] == "POST"
    assert call["params"] == {"next": "/home"}
    assert call["headers"] == {"X-Custom": "1"}
    assert call["cookies"] == {"one-off": "yes"}
    assert call["data"] == {"user": "u"}
    assert call["auth"] == ("u", "p")
    assert call["timeout"] == 5.5
    assert call["allow_redirects"] is False
    assert call["max_redirects"] == 3
    assert call["proxy"] == "http://proxy:8080"
    assert call["proxy_auth"] == ("pu", "pp")
    assert call["verify"] is False
    assert call["referer"] == "https://example.org/start"
    assert call["accept_encoding"] == "gzip"
    assert call["impersonate"] == "chrome124"
    assert call["default_headers"] is False
    assert call["ja3"] == "ja3-string"
    assert call["akamai"] == "akamai-string"
    assert call["extra_fp"] == {"tls_signature_algorithms": ["ecdsa_secp256r1_sha256"]}
    assert call["http_version"] is CurlHttpVersion.V1_1
    assert call["default_encoding"] == "latin-1"
    assert call["interface"] == "eth0"
    assert call["dns"] == ["1.1.1.1"]
    assert call["doh_url"] == "https://1.1.1.1/dns-query"
    assert call["cert"] == ("/tmp/cert.pem", "/tmp/key.pem")
    assert call["max_recv_speed"] == 1024
    assert call["quote"] is False
    assert call["discard_cookies"] is True


async def test_json_body_forwarded(browser):
    """A JSON body is passed through as the curl_cffi json option."""
    browser.enqueue(FakeResponse())
    await server.browser_request("https://example.org/api", method="PUT",
                                 json={"a": [1, 2]})
    assert browser.calls[0]["json"] == {"a": [1, 2]}


async def test_unknown_http_version_rejected(browser):
    """An unrecognized HTTP version fails before any request is made."""
    with pytest.raises(ValueError, match="unknown http_version"):
        await server.browser_request("https://example.org/", http_version="9")
    assert browser.calls == []


async def test_response_shape(browser):
    """Metadata, redirect chain, and response cookies are all reported."""
    hop = FakeResponse(status_code=301, url="https://example.org/old")
    browser.enqueue(FakeResponse(
        url="https://example.org/new",
        headers={"x-server": "test"},
        cookies={"session": "abc"},
        history=[hop],
        http_version=2,
        elapsed=1.5,
        content=b"done",
    ))
    result = await server.browser_request("https://example.org/old")

    assert result["status"] == 200
    assert result["reason"] == "OK"
    assert result["ok"] is True
    assert result["url"] == "https://example.org/new"
    assert result["redirects"] == [{"status": 301, "url": "https://example.org/old"}]
    assert result["http_version"] == "1.1"
    assert result["elapsed_ms"] == 1500
    assert result["headers"]["x-server"] == "test"
    assert result["cookies"] == {"session": "abc"}
    assert result["content_type"] == "text/plain"
    assert result["encoding"] == "text"
    assert result["truncated"] is False


async def test_non_2xx_returned_not_raised(browser):
    """Error statuses come back as ordinary results for the agent to inspect."""
    browser.enqueue(FakeResponse(status_code=403, reason="Forbidden",
                                 content=b"nope"))
    result = await server.browser_request("https://example.org/secret")
    assert result["status"] == 403
    assert result["ok"] is False
    assert result["body"] == "nope"


async def test_unknown_http_version_label_falls_back(browser):
    """HTTP versions outside the known map still produce a readable label."""
    browser.enqueue(FakeResponse(http_version=99))
    result = await server.browser_request("https://example.org/")
    assert result["http_version"] == "99"


async def test_html_converted_to_markdown(browser):
    """HTML bodies become markdown by default."""
    browser.enqueue(FakeResponse(content=b"<h1>Welcome</h1>",
                                 content_type="text/html; charset=utf-8"))
    result = await server.browser_request("https://example.org/")
    assert result["body"].strip() == "# Welcome"


async def test_raw_keeps_html_source(browser):
    """raw=True skips the markdown conversion."""
    browser.enqueue(FakeResponse(content=b"<h1>Welcome</h1>",
                                 content_type="text/html"))
    result = await server.browser_request("https://example.org/", raw=True)
    assert result["body"] == "<h1>Welcome</h1>"


async def test_text_truncated_with_marker(browser):
    """Long text bodies are cut with a truncation marker."""
    browser.enqueue(FakeResponse(content=b"abcdef"))
    result = await server.browser_request("https://example.org/", max_chars=3)
    assert result["body"] == "abc\n\n[truncated at 3 chars]"
    assert result["truncated"] is True


async def test_binary_body_base64_encoded(browser):
    """Non-text MIME bodies are returned base64-encoded without truncation."""
    raw = b"\xff\xd8\xff\xe0binary"
    browser.enqueue(FakeResponse(content=raw, content_type="image/jpeg"))
    result = await server.browser_request("https://example.org/img", max_chars=3)
    assert result["encoding"] == "base64"
    assert base64.b64decode(result["body"]) == raw
    assert result["truncated"] is False


async def test_oversized_body_rejected(browser, monkeypatch):
    """Bodies beyond the byte cap fail instead of flooding the context."""
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 4)
    browser.enqueue(FakeResponse(content=b"12345", content_type="image/png"))
    with pytest.raises(ValueError, match="25 MiB body limit"):
        await server.browser_request("https://example.org/big")


async def test_tool_wire_roundtrip(browser):
    """The MCP wire shape carries structured content and coerces auth lists."""
    browser.enqueue(FakeResponse(content=b"wired"))
    async with Client(server.mcp) as client:
        result = await client.call_tool("browser_request", {
            "url": "https://example.org/api",
            "method": "POST",
            "json": {"k": "v"},
            "auth": ["u", "p"],
            "http_version": "2",
        })
    assert not result.is_error
    assert result.data["status"] == 200
    assert result.data["body"] == "wired"
    call = browser.calls[0]
    assert call["auth"] == ("u", "p")
    assert call["json"] == {"k": "v"}
    assert call["http_version"] is CurlHttpVersion.V2_0


async def test_cookies_list_empty(browser):
    """Listing an untouched jar returns no cookies."""
    assert await server.browser_cookies("list") == {"cookies": []}


async def test_cookies_set_then_list(browser):
    """A planted cookie appears in the jar with its full attributes."""
    result = await server.browser_cookies(
        "set", name="token", value="xyz", domain="example.org",
        path="/app", secure=True,
    )
    assert result == {"set": {"name": "token", "value": "xyz",
                              "domain": "example.org", "path": "/app",
                              "secure": True}}

    listed = await server.browser_cookies("list")
    assert len(listed["cookies"]) == 1
    cookie = listed["cookies"][0]
    assert cookie["name"] == "token"
    assert cookie["value"] == "xyz"
    assert cookie["domain"] == "example.org"
    assert cookie["path"] == "/app"
    assert cookie["secure"] is True


async def test_cookies_set_defaults(browser):
    """Cookies default to host-only at the root path."""
    await server.browser_cookies("set", name="plain", value="1")
    cookie = (await server.browser_cookies("list"))["cookies"][0]
    assert cookie["domain"] == ""
    assert cookie["path"] == "/"
    assert cookie["secure"] is False


async def test_cookies_set_requires_name(browser):
    """Setting a cookie without a name is rejected."""
    with pytest.raises(ValueError, match="name is required"):
        await server.browser_cookies("set", value="orphan")


async def test_cookies_delete(browser):
    """Deleting by name removes matching cookies regardless of domain."""
    await server.browser_cookies("set", name="a", value="1", domain="one.org")
    await server.browser_cookies("set", name="a", value="2", domain="two.org")
    await server.browser_cookies("set", name="b", value="3")

    assert await server.browser_cookies("delete", name="a") == {"deleted": "a"}
    remaining = (await server.browser_cookies("list"))["cookies"]
    assert [c["name"] for c in remaining] == ["b"]


async def test_cookies_delete_requires_name(browser):
    """Deleting without a name is rejected."""
    with pytest.raises(ValueError, match="name is required"):
        await server.browser_cookies("delete")


async def test_cookies_clear_all_and_by_domain(browser):
    """Clearing empties the jar, optionally restricted to one domain."""
    await server.browser_cookies("set", name="a", value="1", domain="one.org")
    await server.browser_cookies("set", name="b", value="2", domain="two.org")

    assert await server.browser_cookies("clear", domain="one.org") == {"cleared": True}
    remaining = (await server.browser_cookies("list"))["cookies"]
    assert [c["domain"] for c in remaining] == ["two.org"]

    await server.browser_cookies("clear")
    assert await server.browser_cookies("list") == {"cookies": []}


async def test_cookies_unknown_action_rejected(browser):
    """An unrecognized action is rejected without touching the jar."""
    with pytest.raises(ValueError, match="unknown action"):
        await server.browser_cookies("explode")


async def test_cookies_wire_list(browser):
    """The cookie tool returns structured content over MCP."""
    await server.browser_cookies("set", name="wired", value="1")
    async with Client(server.mcp) as client:
        result = await client.call_tool("browser_cookies", {"action": "list"})
    assert not result.is_error
    assert json.loads(result.content[0].text)["cookies"][0]["name"] == "wired"
