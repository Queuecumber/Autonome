"""Session manager: JSONL history per contact with token-based truncation."""

import json
from pathlib import Path
from typing import Any

import litellm


class SessionManager:
    def __init__(self, store_dir: Path, max_history_tokens: int = 100000):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_tokens = max_history_tokens

    def _session_path(self, contact: str) -> Path:
        return self.store_dir / f"{contact}.jsonl"

    def load(self, contact: str) -> list[dict[str, Any]]:
        """Load conversation history for a contact."""
        path = self._session_path(contact)
        if not path.exists():
            return []
        messages = []
        for line in path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        return messages

    def append(self, contact: str, messages: list[dict[str, Any]]) -> None:
        """Append messages to a contact's session file."""
        path = self._session_path(contact)
        with path.open("a") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    def load_truncated(self, contact: str, model: str = "claude-opus-4-6") -> list[dict[str, Any]]:
        """Load history, truncating oldest complete exchanges if over token budget."""
        messages = self.load(contact)
        if not messages:
            return messages

        token_count = self._count_tokens(messages, model)
        if token_count <= self.max_history_tokens:
            return messages

        exchanges = self._group_exchanges(messages)

        while exchanges and self._count_tokens(
            [m for ex in exchanges for m in ex], model
        ) > self.max_history_tokens:
            exchanges.pop(0)

        return [m for ex in exchanges for m in ex]

    def _group_exchanges(self, messages: list[dict]) -> list[list[dict]]:
        """Group messages into exchanges, each starting with a user message."""
        exchanges: list[list[dict]] = []
        current: list[dict] = []
        for msg in messages:
            if msg.get("role") == "user" and current:
                exchanges.append(current)
                current = []
            current.append(msg)
        if current:
            exchanges.append(current)
        return exchanges

    def _count_tokens(self, messages: list[dict], model: str) -> int:
        """Count tokens using LiteLLM's token counter with fallback."""
        try:
            return litellm.token_counter(model=model, messages=messages)
        except Exception:
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            return total_chars // 4
