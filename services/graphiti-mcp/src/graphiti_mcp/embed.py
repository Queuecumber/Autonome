"""Optional text embeddings for stored memory and semantic queries."""

import logging
import math
import os

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "")
MODEL = os.environ.get("EMBEDDING_MODEL", "")
API_KEY = os.environ.get("EMBEDDING_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
DIM = int(os.environ.get("EMBEDDING_DIM", "0"))
PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
TIMEOUT = float(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "20"))
MIN_SCORE = float(os.environ.get("EMBEDDING_MIN_SCORE", "0.6"))

_client: AsyncOpenAI | None = None
_checked = False


def _embedder() -> AsyncOpenAI | None:
    """Return a cached client, or None when no embedding model is configured.

    Raises:
        ValueError: If provider, dimensions, timeout, or similarity threshold are invalid.
    """
    global _client, _checked
    if not _checked:
        if MODEL:
            if PROVIDER not in {"openai", "nvidia"}:
                raise ValueError("EMBEDDING_PROVIDER must be openai or nvidia")
            if DIM < 0 or TIMEOUT <= 0 or not math.isfinite(TIMEOUT):
                raise ValueError("Embedding dimensions must be nonnegative and timeout positive")
            if not 0 <= MIN_SCORE <= 1:
                raise ValueError("EMBEDDING_MIN_SCORE must be between 0 and 1")
            _client = AsyncOpenAI(
                api_key=API_KEY or "unset", base_url=BASE_URL or None,
                timeout=TIMEOUT, max_retries=0)
        else:
            logger.warning("No EMBEDDING_MODEL set; semantic search disabled")
        _checked = True
    return _client


def enabled() -> bool:
    """Return whether an embedding client is configured; invalid settings raise ValueError."""
    return _embedder() is not None


async def embed(text: str, *, is_query: bool = False) -> list[float] | None:
    """Embed stored text or a retrieval query using the configured endpoint.

    Args:
        text: The text to embed.
        is_query: Use query encoding instead of document encoding. NVIDIA
            requests send input_type=query or passage accordingly.

    Returns:
        A finite, nonzero vector, or None for blank input, disabled embeddings,
        or an endpoint/configuration failure. DIM=0 requests native dimensions;
        a positive DIM is requested from the endpoint and checked on return.
        Vectors are never silently truncated. Failures do not block memory writes.
    """
    if not text.strip():
        return None
    try:
        client = _embedder()
        if client is None:
            return None
        options = {}
        if DIM:
            options["dimensions"] = DIM
        if PROVIDER == "nvidia":
            options["extra_body"] = {
                "input_type": "query" if is_query else "passage", "truncate": "END"}
        result = await client.embeddings.create(
            model=MODEL, input=text, encoding_format="float", **options)
        vector = result.data[0].embedding
        if DIM and len(vector) != DIM:
            raise ValueError(f"Embedding endpoint returned {len(vector)} dimensions; expected {DIM}")
        if not vector or not all(math.isfinite(value) for value in vector) or not any(vector):
            raise ValueError("Embedding endpoint returned an empty, non-finite, or zero vector")
        return vector
    except Exception as e:
        logger.error("Embedding unavailable: %r", e)
        return None
