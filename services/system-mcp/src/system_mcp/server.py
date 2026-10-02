"""System MCP server — web search, web fetch, and general system tools."""

import base64
import os
from typing import Any, Literal

import html2text
import httpx
from curl_cffi import CurlHttpVersion
from curl_cffi.requests import AsyncSession
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import BlobResourceContents, EmbeddedResource

SEARCH_URL = os.environ.get("SEARCH_URL", "https://api.perplexity.ai/search")
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY", "")
MAX_FETCH_CHARS = int(os.environ.get("MAX_FETCH_CHARS", "20000"))
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_BODY_BYTES = int(os.environ.get("MAX_BROWSER_BODY_BYTES", str(25 * 1024 * 1024)))

mcp = FastMCP("system", instructions=(
  """
# System Tools

These tools are general system actions you can take. They
allow you to search the web, fetch URLs, and drive a Chrome-impersonating
browser session with full control over request options and cookies.
"""
))

_http = httpx.AsyncClient(timeout=30, headers={"User-Agent": "Autonome/1.0"})

_browser = AsyncSession(impersonate="chrome")

_HTTP_VERSIONS = {
    "1.0": CurlHttpVersion.V1_0,
    "1.1": CurlHttpVersion.V1_1,
    "2": CurlHttpVersion.V2_0,
    "2tls": CurlHttpVersion.V2TLS,
    "2-prior-knowledge": CurlHttpVersion.V2_PRIOR_KNOWLEDGE,
    "3": CurlHttpVersion.V3,
    "3-only": CurlHttpVersion.V3ONLY,
}
_HTTP_VERSION_LABELS = {v.value: k for k, v in _HTTP_VERSIONS.items()}

def _is_text_mime(mime: str) -> bool:
    """Return True when a response MIME type is safe to decode and return as text.

    Args:
        mime: Lowercased MIME type without parameters (e.g. "text/html").

    Returns:
        True for missing types, text/*, JSON, XML, JavaScript,
        form-encoded, and SVG bodies.
    """
    return (
        mime.startswith("text/")
        or mime.endswith("+json")
        or mime.endswith("+xml")
        or mime in {
            "",
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-javascript",
            "application/x-www-form-urlencoded",
            "image/svg+xml",
        }
    )

_h2t = html2text.HTML2Text()
_h2t.ignore_links = False
_h2t.ignore_images = True
_h2t.body_width = 0  # no line wrapping


@mcp.tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web.

    Args:
        query: What to search for.
        max_results: Cap on the number of results.

    Returns:
        Ranked results with title, URL, and snippet (markdown). "No
        results found." if the search returned nothing.

    Raises:
        httpx.HTTPStatusError: If the search backend returns a non-2xx
            (rate limit, auth, server error).
    """
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
async def web_fetch(url: str, max_chars: int = MAX_FETCH_CHARS) -> str | ToolResult:
    """Fetch a URL, returning text or a PDF binary resource.

    HTML becomes markdown; other non-PDF content is decoded as text.
    Declared application/pdf responses (case-insensitive, including MIME
    parameters) and PDFs with missing or generic binary MIME types are
    returned as MCP embedded resources with the final response URL as URI.
    Redirects are followed. PDF responses are limited to 25 MiB; text
    responses retain their existing character truncation behavior.

    Args:
        url: The URL to fetch.
        max_chars: Text-only truncation threshold; longer text gets a
            `[truncated at N chars]` marker. Does not truncate PDFs.

    Returns:
        Markdown for HTML, decoded text for other non-PDF responses, or
        a PDF embedded binary resource for rendering by the session manager.

    Raises:
        httpx.HTTPStatusError: If the fetch returns a non-2xx response.
        ValueError: If a PDF exceeds 25 MiB; no partial PDF is returned.
    """
    async with _http.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        mime = content_type.split(";", 1)[0].strip().lower()
        generic = mime in ("", "application/octet-stream", "binary/octet-stream")
        pdf = mime == "application/pdf"
        chunks = bytearray()
        async for chunk in resp.aiter_bytes():
            chunks.extend(chunk)
            if generic and len(chunks) >= 5:
                pdf = chunks.startswith(b"%PDF-")
                generic = False
            if pdf and len(chunks) > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds the 25 MiB input limit")

        if pdf:
            return ToolResult(content=[EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=str(resp.url), mimeType="application/pdf",
                    blob=base64.b64encode(chunks).decode("ascii"),
                ),
            )])

        # aiter_bytes has already decoded Content-Encoding; retain only charset metadata.
        decoded = httpx.Response(
            resp.status_code,
            headers={"content-type": content_type},
            content=bytes(chunks),
            default_encoding=resp.default_encoding,
        ).text
        text = _h2t.handle(decoded) if "html" in content_type else decoded

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"
    return text


@mcp.tool
async def browser_request(
    url: str,
    method: str = "GET",
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    data: dict[str, str] | str | None = None,
    json: dict[str, Any] | list[Any] | None = None,
    auth: tuple[str, str] | None = None,
    timeout: float = 30,
    allow_redirects: bool = True,
    max_redirects: int = 30,
    proxy: str | None = None,
    proxies: dict[str, str] | None = None,
    proxy_auth: tuple[str, str] | None = None,
    verify: bool = True,
    referer: str | None = None,
    accept_encoding: str | None = None,
    impersonate: str = "chrome",
    default_headers: bool = True,
    ja3: str | None = None,
    akamai: str | None = None,
    extra_fp: dict[str, Any] | None = None,
    http_version: str | None = None,
    default_encoding: str = "utf-8",
    interface: str | None = None,
    dns: str | list[str] | None = None,
    doh_url: str | None = None,
    cert: tuple[str, str] | None = None,
    max_recv_speed: int = 0,
    quote: str | bool | None = None,
    discard_cookies: bool = False,
    raw: bool = False,
    max_chars: int = MAX_FETCH_CHARS,
) -> dict[str, Any]:
    """Make an HTTP request through the shared Chrome-impersonating browser session.

    Exposes the full curl_cffi request API. TLS and HTTP fingerprints mimic
    real Chrome, which bypasses most bot detection. Cookies set by servers
    persist in the session across calls (like a real browser profile);
    manage them with the browser_cookies tool. Non-2xx responses are
    returned, not raised, so you can inspect login and error flows.

    Args:
        url: The URL to request.
        method: HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD, ...).
        params: Query string parameters.
        headers: Extra headers; these override the browser default headers.
        cookies: Cookies to send with this request only (not stored in the jar).
        data: Form fields (dict, sent urlencoded) or a raw request body (str).
        json: JSON body; sets Content-Type: application/json automatically.
        auth: HTTP basic auth as (username, password).
        timeout: Seconds before giving up on the whole request.
        allow_redirects: Whether to follow redirects.
        max_redirects: Cap on followed redirects; -1 for unlimited.
        proxy: Single proxy URL, e.g. "http://user:pass@host:port".
        proxies: Per-scheme proxies, e.g. {"http": ..., "https": ...}.
            Cannot be combined with proxy.
        proxy_auth: HTTP basic auth for the proxy as (username, password).
        verify: Verify HTTPS certificates.
        referer: Shortcut for the Referer header.
        accept_encoding: Shortcut for the Accept-Encoding header; defaults
            to what the impersonated browser sends.
        impersonate: Browser fingerprint, e.g. "chrome" (latest Chrome),
            "safari", "chrome124", "safari17_0".
        default_headers: Send the impersonated browser's default headers.
        ja3: Custom JA3 TLS fingerprint string (overrides impersonate).
        akamai: Custom Akamai HTTP/2 fingerprint string.
        extra_fp: Extra fingerprint options complementing ja3/akamai.
        http_version: Force an HTTP version: "1.0", "1.1", "2", "2tls",
            "2-prior-knowledge", "3", or "3-only".
        default_encoding: Charset for decoding bodies when the response
            headers do not declare one.
        interface: Network interface name or local IP to bind to.
        dns: DNS server IP(s); requires a c-ares build.
        doh_url: DNS-over-HTTPS resolver URL.
        cert: Client certificate as (cert path, key path).
        max_recv_speed: Maximum receive speed in bytes per second; 0 is
            unlimited.
        quote: Extra characters to percent-encode in the URL; False keeps
            the URL exactly as given.
        discard_cookies: Discard cookies the server sets in this response
            instead of storing them in the jar.
        raw: Return HTML as-is instead of converting it to markdown.
        max_chars: Truncation threshold for text bodies; longer bodies get
            a `[truncated at N chars]` marker. Does not apply to base64
            bodies.

    Returns:
        A dict with status, reason, ok, the final url, the redirects
        chain, http_version, elapsed_ms, response headers, cookies set by
        this response, content_type, and the body — text (markdown for
        HTML unless raw) or base64 for binary bodies, indicated by the
        encoding field. Text bodies also carry a truncated flag.

    Raises:
        ValueError: If http_version is unknown or the response body
            exceeds the 25 MiB limit.
        curl_cffi.requests.errors.RequestException: On transport failures
            (DNS, TLS, timeout, proxy, too many redirects).
    """
    options: dict[str, Any] = {
        "params": params,
        "headers": headers,
        "cookies": cookies,
        "data": data,
        "json": json,
        "auth": tuple(auth) if auth is not None else None,
        "timeout": timeout,
        "allow_redirects": allow_redirects,
        "max_redirects": max_redirects,
        "proxy": proxy,
        "proxies": proxies,
        "proxy_auth": tuple(proxy_auth) if proxy_auth is not None else None,
        "verify": verify,
        "referer": referer,
        "accept_encoding": accept_encoding,
        "impersonate": impersonate,
        "default_headers": default_headers,
        "ja3": ja3,
        "akamai": akamai,
        "extra_fp": extra_fp,
        "default_encoding": default_encoding,
        "interface": interface,
        "dns": dns,
        "doh_url": doh_url,
        "cert": tuple(cert) if cert is not None else None,
        "max_recv_speed": max_recv_speed,
        "quote": quote,
        "discard_cookies": discard_cookies,
    }
    if http_version is not None:
        try:
            options["http_version"] = _HTTP_VERSIONS[http_version]
        except KeyError:
            raise ValueError(
                f"unknown http_version {http_version!r}; "
                f"expected one of {sorted(_HTTP_VERSIONS)}"
            ) from None
    request = {k: v for k, v in options.items() if v is not None}
    resp = await _browser.request(method, url, **request)

    content_type = resp.headers.get("content-type", "")
    mime = content_type.split(";", 1)[0].strip().lower()
    if len(resp.content) > MAX_BODY_BYTES:
        raise ValueError("response exceeds the 25 MiB body limit")

    truncated = False
    if _is_text_mime(mime):
        encoding = "text"
        body = resp.text
        if "html" in mime and not raw:
            body = _h2t.handle(body)
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n\n[truncated at {max_chars} chars]"
            truncated = True
    else:
        encoding = "base64"
        body = base64.b64encode(resp.content).decode("ascii")

    return {
        "status": resp.status_code,
        "reason": resp.reason,
        "ok": resp.ok,
        "url": str(resp.url),
        "redirects": [
            {"status": hop.status_code, "url": str(hop.url)}
            for hop in resp.history
        ],
        "http_version": _HTTP_VERSION_LABELS.get(resp.http_version, str(resp.http_version)),
        "elapsed_ms": int(resp.elapsed.total_seconds() * 1000),
        "headers": dict(resp.headers.multi_items()),
        "cookies": dict(resp.cookies),
        "content_type": content_type,
        "encoding": encoding,
        "body": body,
        "truncated": truncated,
    }


@mcp.tool
async def browser_cookies(
    action: Literal["list", "set", "delete", "clear"],
    name: str | None = None,
    value: str = "",
    domain: str = "",
    path: str | None = None,
    secure: bool = False,
) -> dict[str, Any]:
    """Inspect and manage the shared browser session cookie jar.

    The jar belongs to the browser_request session: cookies here are sent
    automatically with matching requests, exactly like a real browser
    profile.

    Args:
        action: "list" to dump the jar, "set" to add or replace a cookie,
            "delete" to remove cookies by name, "clear" to empty the jar.
        name: Cookie name; required for "set" and "delete".
        value: Cookie value for "set".
        domain: Domain the cookie belongs to; "" (host-only) for "set",
            and a filter for "delete"/"clear" where "" matches all
            domains.
        path: Cookie path; defaults to "/" for "set", and filters
            "delete"/"clear" where None matches all paths.
        secure: Whether the cookie is restricted to HTTPS for "set".

    Returns:
        "list": {"cookies": [{name, value, domain, path, secure,
        expires}, ...]}. "set": {"set": {...}} the stored cookie.
        "delete": {"deleted": name}. "clear": {"cleared": True}.

    Raises:
        ValueError: If name is missing for "set"/"delete" or the action
            is unknown.
    """
    jar = _browser.cookies
    if action == "list":
        return {"cookies": [
            {
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path,
                "secure": c.secure,
                "expires": c.expires,
            }
            for c in jar.jar
        ]}
    if action == "set":
        if not name:
            raise ValueError("name is required to set a cookie")
        jar.set(name, value, domain=domain, path=path or "/", secure=secure)
        return {"set": {
            "name": name, "value": value,
            "domain": domain, "path": path or "/", "secure": secure,
        }}
    if action == "delete":
        if not name:
            raise ValueError("name is required to delete a cookie")
        jar.delete(name, domain=domain or None, path=path)
        return {"deleted": name}
    if action == "clear":
        jar.clear(domain=domain or None, path=path)
        return {"cleared": True}
    raise ValueError(f"unknown action {action!r}")


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8002)
