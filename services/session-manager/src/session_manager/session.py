"""Session manager: JSONL history per (channel, session_id) with token-based truncation."""

import json
from pathlib import Path
from typing import Any


class SessionManager:
    def __init__(self, store_dir: Path, max_history_tokens: int = 100000):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_tokens = max_history_tokens

    def _session_path(self, channel: str, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return self.store_dir / f"{channel}_{safe_id}.jsonl"

    def load(self, channel: str, session_id: str) -> list[dict[str, Any]]:
        path = self._session_path(channel, session_id)
        if not path.exists():
            return []
        messages = []
        for line in path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        return messages

    def append(self, channel: str, session_id: str, messages: list[dict[str, Any]]) -> None:
        path = self._session_path(channel, session_id)
        with path.open("a") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    def load_truncated(self, channel: str, session_id: str) -> list[dict[str, Any]]:
        messages = self.load(channel, session_id)
        if not messages:
            return messages

        token_count = self._count_tokens(messages)
        if token_count <= self.max_history_tokens:
            return messages

        exchanges = self._group_exchanges(messages)
        while exchanges and self._count_tokens(
            [m for ex in exchanges for m in ex]
        ) > self.max_history_tokens:
            exchanges.pop(0)

        return [m for ex in exchanges for m in ex]

    def _group_exchanges(self, messages: list[dict]) -> list[list[dict]]:
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

    def _count_tokens(self, messages: list[dict]) -> int:
        """Estimate token count. Rough heuristic: ~4 chars per token."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4
