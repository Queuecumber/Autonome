"""Embeddings, computed only for text the agent chose to store.

There is no background indexing pass here. Deciding to save a fact *is* the
decision to embed it, which keeps the number of embedding calls equal to the
number of deliberate writes.

Unconfigured, everything degrades to structural retrieval — entities, relations
and traversal still work, semantic search returns nothing. That is a usable
service, not a broken one, so the absence of an embedding endpoint is not fatal.
"""

import logging
import os

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "")
MODEL = os.environ.get("EMBEDDING_MODEL", "")
API_KEY = os.environ.get("EMBEDDING_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))

_client = None
_checked = False


def _embedder():
    global _client, _checked
    if not _checked:
        _checked = True
        if MODEL:
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            _client = OpenAIEmbedder(config=OpenAIEmbedderConfig(
                embedding_model=MODEL, embedding_dim=DIM,
                api_key=API_KEY or "unset",
                base_url=BASE_URL or None))
        else:
            logger.warning("No EMBEDDING_MODEL set; semantic search disabled")
    return _client


def enabled() -> bool:
    return _embedder() is not None


async def embed(text: str) -> list[float] | None:
    """Embed one string, or None if embeddings are not configured.

    A failure here must not lose the write: the fact is still worth storing
    without a vector, and can be re-embedded later.
    """
    client = _embedder()
    if client is None or not text:
        return None
    try:
        return await client.create(input_data=text)
    except Exception as e:
        logger.error("Embedding failed, storing without vector: %r", e)
        return None
