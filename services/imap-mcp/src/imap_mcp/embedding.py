"""Optional, independently configured embeddings for the local mail index."""

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from typing import Literal
from urllib.parse import urlparse

from openai import OpenAI


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding endpoint policy; an empty model disables all embedding requests.

    Args:
        model: Exact model ID served by the operator-configured endpoint.
        base_url: OpenAI-compatible endpoint URL, or the SDK default when empty.
        api_key: Credential, excluded from representation and index identity.
        provider: NVIDIA adds query/passage encoding parameters.
        dimensions: Requested dimensions, or zero for the model's native size.
        timeout: Per-request HTTP timeout in seconds.
    """

    model: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    provider: Literal["nvidia", "openai"] = "nvidia"
    dimensions: int = 0
    timeout: float = 20

    def __post_init__(self):
        """Reject invalid policy before opening the database or contacting a provider."""
        if self.provider not in {"nvidia", "openai"}:
            raise ValueError("IMAP_EMBEDDING_PROVIDER must be nvidia or openai")
        url = urlparse(self.base_url)
        if self.model and (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment):
            raise ValueError("An enabled mail embedder requires an explicit HTTP(S) base URL without credentials/query/fragment")
        if type(self.dimensions) is not int or not 0 <= self.dimensions <= 16384:
            raise ValueError("IMAP_EMBEDDING_DIM must be from 0 to 16384")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("IMAP_EMBEDDING_TIMEOUT_SECONDS must be finite and positive")

    @property
    def profile(self) -> str:
        """Return an internal vector-space version, never a replacement mail identifier."""
        values = [self.model, self.base_url, self.provider, self.dimensions, "mail-chunks-v1"]
        return hashlib.sha256(json.dumps(values).encode()).hexdigest()

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        """Read only IMAP_EMBEDDING_* settings; malformed values raise ValueError."""
        return cls(model=os.getenv("IMAP_EMBEDDING_MODEL", ""),
                   base_url=os.getenv("IMAP_EMBEDDING_BASE_URL", ""),
                   api_key=os.getenv("IMAP_EMBEDDING_API_KEY", ""),
                   provider=os.getenv("IMAP_EMBEDDING_PROVIDER", "nvidia"),
                   dimensions=int(os.getenv("IMAP_EMBEDDING_DIM", "0")),
                   timeout=float(os.getenv("IMAP_EMBEDDING_TIMEOUT_SECONDS", "20")))


class Embedder:
    """Encode queries and document chunks; callers own retries and coverage reporting.

    Args:
        settings: Explicit operator opt-in and endpoint configuration.
    """

    def __init__(self, settings: EmbeddingSettings):
        """Create a client only when enabled; construction makes no network requests."""
        self.settings = settings
        self.client = (OpenAI(api_key=settings.api_key or "unset", base_url=settings.base_url or None,
                              timeout=settings.timeout, max_retries=0) if settings.model else None)

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Embed a bounded batch in input order, validating dimensions and finite vectors.

        Args:
            texts: One to sixteen nonempty text chunks, each at most 4096 characters.
            query: Whether these are retrieval queries instead of stored passages.

        Returns:
            One nonzero finite vector per text, with a consistent dimension.

        Raises:
            ValueError: Invalid input, disabled embeddings, or invalid provider output.
            openai.APIError: Provider failure; no implicit SDK retries are performed.
        """
        if self.client is None or not 1 <= len(texts) <= 16 or any(not text.strip() or len(text) > 4096 for text in texts):
            raise ValueError("Embeddings require an enabled model and 1..16 nonempty bounded texts")
        options = {"dimensions": self.settings.dimensions} if self.settings.dimensions else {}
        if self.settings.provider == "nvidia":
            options["extra_body"] = {"input_type": "query" if query else "passage", "truncate": "NONE"}
        result = self.client.embeddings.create(model=self.settings.model, input=texts, encoding_format="float", **options)
        ordered = sorted(result.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ValueError("Embedding response omitted or duplicated an input")
        vectors = [item.embedding for item in ordered]
        dimension = self.settings.dimensions or len(vectors[0])
        if not 1 <= dimension <= 16384 or any(len(v) != dimension or not all(math.isfinite(x) for x in v)
                                             or not any(v) for v in vectors):
            raise ValueError("Embedding response has invalid dimensions or values")
        return vectors

    def close(self) -> None:
        """Release the embedding HTTP client after background workers and tools stop."""
        if self.client is not None:
            self.client.close()
