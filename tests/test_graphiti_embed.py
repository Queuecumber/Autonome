"""Embedding HTTP contracts and graceful failure without a graph database."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from graphiti_mcp import embed


@pytest.fixture
async def endpoint(monkeypatch):
    """Provide a real SDK client backed by a controlled embedding HTTP endpoint."""
    state = {"requests": [], "vector": [0.6, 0.8, 0.0], "status": 200}

    def respond(request):
        """Capture the outbound request and return the configured embedding response."""
        state["requests"].append(request)
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": {"message": "unavailable"}})
        return httpx.Response(200, json={
            "object": "list", "model": "nvidia/test-model",
            "data": [{"object": "embedding", "index": 0, "embedding": state["vector"]}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        })

    transport = httpx.MockTransport(respond)
    http_client = httpx.AsyncClient(transport=transport)

    def client(**kwargs):
        """Construct the SDK client with the test transport and production options."""
        state["client_options"] = kwargs
        return AsyncOpenAI(http_client=http_client, **kwargs)

    for name, value in {
        "MODEL": "nvidia/nvidia/test-model", "BASE_URL": "https://embedding.test/v1",
        "API_KEY": "test-embedding-key", "DIM": 0, "PROVIDER": "nvidia", "TIMEOUT": 7,
        "MIN_SCORE": 0.6,
        "_client": None, "_checked": False,
    }.items():
        monkeypatch.setattr(embed, name, value)
    monkeypatch.setattr(embed, "AsyncOpenAI", client)
    yield state
    await http_client.aclose()


@pytest.mark.parametrize("is_query,input_type", [(False, "passage"), (True, "query")])
async def test_nvidia_request_distinguishes_query_and_passage(endpoint, is_query, input_type):
    """NVIDIA receives the exact model alias, input role, and float encoding."""
    assert await embed.embed("A memory", is_query=is_query) == endpoint["vector"]
    request = endpoint["requests"][0]
    assert request.url == "https://embedding.test/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer test-embedding-key"
    assert json.loads(request.content) == {
        "model": "nvidia/nvidia/test-model", "input": "A memory",
        "encoding_format": "float", "input_type": input_type, "truncate": "END",
    }
    assert endpoint["client_options"]["timeout"] == 7
    assert endpoint["client_options"]["max_retries"] == 0


async def test_explicit_dimensions_are_requested_from_the_endpoint(endpoint, monkeypatch):
    """Dimension reduction is requested from the model, never silently sliced locally."""
    monkeypatch.setattr(embed, "DIM", 3)
    assert await embed.embed("A memory") == endpoint["vector"]
    assert json.loads(endpoint["requests"][0].content)["dimensions"] == 3


async def test_wrong_dimension_response_is_rejected(endpoint, monkeypatch):
    """A server that ignores the requested dimensions cannot corrupt stored vectors."""
    monkeypatch.setattr(embed, "DIM", 2)
    assert await embed.embed("A memory") is None


async def test_openai_compatible_mode_omits_nvidia_fields(endpoint, monkeypatch):
    """Generic compatible endpoints receive only standard embedding parameters."""
    monkeypatch.setattr(embed, "PROVIDER", "openai")
    assert await embed.embed("A query", is_query=True) == endpoint["vector"]
    body = json.loads(endpoint["requests"][0].content)
    assert "input_type" not in body and "truncate" not in body


async def test_disabled_embeddings_do_not_construct_a_client(endpoint, monkeypatch):
    """An empty model disables embeddings without creating an HTTP client."""
    monkeypatch.setattr(embed, "MODEL", "")
    assert await embed.embed("A memory") is None
    assert embed.enabled() is False
    assert "client_options" not in endpoint


@pytest.mark.parametrize("text", ["", " \n\t "])
async def test_blank_input_does_not_call_the_endpoint(endpoint, text):
    """Blank text cannot consume an embedding request."""
    assert await embed.embed(text) is None
    assert endpoint["requests"] == []


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_endpoint_failure_degrades_without_retries(endpoint, status):
    """An embedding outage returns None promptly so memory writes can continue."""
    endpoint["status"] = status
    assert await embed.embed("A memory") is None
    assert len(endpoint["requests"]) == 1


@pytest.mark.parametrize("vector", [[], [0.0, 0.0]])
async def test_unusable_vectors_are_rejected(endpoint, vector):
    """Empty and zero vectors cannot enter semantic retrieval."""
    endpoint["vector"] = vector
    assert await embed.embed("A memory") is None


@pytest.mark.parametrize("setting,value", [
    ("PROVIDER", "unsupported"), ("DIM", -1), ("TIMEOUT", 0), ("TIMEOUT", float("inf")),
    ("MIN_SCORE", 1.1), ("MIN_SCORE", -0.1),
])
async def test_invalid_configuration_does_not_block_memory_writes(endpoint, monkeypatch, setting, value):
    """Bad optional embedding settings fail closed without issuing network requests."""
    monkeypatch.setattr(embed, setting, value)
    assert await embed.embed("A memory") is None
    assert endpoint["requests"] == []


async def test_client_is_reused(endpoint):
    """Multiple embeddings reuse one SDK connection pool."""
    assert embed.enabled()
    client = embed._client
    await embed.embed("First")
    await embed.embed("Second")
    assert embed._client is client
    assert len(endpoint["requests"]) == 2
