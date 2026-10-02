"""Session storage: versioned JSONL files with reported-token compaction signals.

Each session_id has one or more `<safe_id>.<N>.jsonl` files, where the highest
N is the active file. Compaction writes a new version N+1 with a summary at
the top followed by the recent recency-window messages. Older versions stay
on disk as an audit trail.

Token accounting reads `{"type": "comment", "kind": "usage", "input_tokens": …}`
entries written by the orchestrator after each LLM call — ground truth from
the model, not a char-count heuristic.
"""

import json
import os
import re
from pathlib import Path
import tempfile
from typing import Any

from session_manager.context import paired_cutoff


class SessionManager:
    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_unversioned()

    # ── path helpers ─────────────────────────────────────────

    @staticmethod
    def _safe_id(session_id: str) -> str:
        return session_id.replace("/", "_").replace("\\", "_")

    _VERSION_RE = re.compile(r"^(?P<id>.+)\.(?P<n>\d+)\.jsonl$")

    def _versioned_files(self, session_id: str) -> list[tuple[int, Path]]:
        """Return [(version, path), …] sorted ascending. Empty if none exist."""
        safe = self._safe_id(session_id)
        out: list[tuple[int, Path]] = []
        for p in self.store_dir.iterdir():
            m = self._VERSION_RE.match(p.name)
            if m and m.group("id") == safe:
                out.append((int(m.group("n")), p))
        out.sort()
        return out

    def _active_path(self, session_id: str) -> Path:
        """Highest-version file, or version 0 if none yet."""
        versions = self._versioned_files(session_id)
        if versions:
            return versions[-1][1]
        return self.store_dir / f"{self._safe_id(session_id)}.0.jsonl"

    def _migrate_unversioned(self) -> None:
        """One-time rename of legacy `<id>.jsonl` files to `<id>.0.jsonl`."""
        for path in self.store_dir.glob("*.jsonl"):
            if self._VERSION_RE.match(path.name):
                continue
            new_path = path.with_name(f"{path.stem}.0.jsonl")
            if not new_path.exists():
                path.rename(new_path)

    # ── read / write ─────────────────────────────────────────

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Read all messages from the active version."""
        path = self._active_path(session_id)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append messages to the active version."""
        path = self._active_path(session_id)
        with path.open("a") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def bump_version(self, session_id: str, messages: list[dict[str, Any]]) -> Path:
        """Write a new version N+1 with the given messages and return its path.

        Previous versions are kept on disk as audit trail. The new version
        becomes the active file for subsequent `load` / `append` calls.
        Publication is atomic: serialization/write failures leave the previous
        version active. Filesystem and serialization errors propagate to the caller.
        """
        versions = self._versioned_files(session_id)
        next_n = (versions[-1][0] + 1) if versions else 0
        new_path = self.store_dir / f"{self._safe_id(session_id)}.{next_n}.jsonl"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.store_dir,
                                             prefix=".compaction-", suffix=".tmp", delete=False) as f:
                temporary = Path(f.name)
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, new_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return new_path

    # ── compaction support ───────────────────────────────────

    @staticmethod
    def latest_input_tokens(messages: list[dict[str, Any]]) -> int | None:
        """Walk messages newest-to-oldest, return the most recent usage
        comment's `input_tokens`. None if no usage data is present.
        """
        for m in reversed(messages):
            if (m.get("type") == "comment"
                    and m.get("kind") == "usage"
                    and m.get("input_tokens") is not None):
                return m["input_tokens"]
        return None

    @staticmethod
    def recency_split(messages: list[dict[str, Any]], recency_tokens: int) -> int:
        """Choose a paired recent tail using reported prompt-token growth.

        Args:
            messages: Persisted items, including usage from every tool iteration.
            recency_tokens: Soft token target for the retained recent window.

        Returns:
            A safe cutoff, or zero if no meaningful prefix can be folded. The
            crossing delta stays in the tail. Decreases (for example when old
            transient images disappear) do not cancel later observed growth.
            If growth is insufficient but the oldest recorded prompt already
            exceeds the target, its preceding history can be folded while all
            subsequently observed conversation stays verbatim.

        Prompt usage includes fixed instructions and transient content, so this
        is an approximate recency window, not exact per-message token accounting.
        """
        usages: list[tuple[int, int]] = []
        for i, m in enumerate(messages):
            if (m.get("type") == "comment"
                    and m.get("kind") == "usage"
                    and m.get("input_tokens") is not None):
                usages.append((i, m["input_tokens"]))

        cumulative = 0
        cutoff = 0
        for i in range(len(usages) - 1, 0, -1):
            _line_curr, tok_curr = usages[i]
            line_prev, tok_prev = usages[i - 1]
            cumulative += max(0, tok_curr - tok_prev)
            if cumulative >= recency_tokens:
                cutoff = line_prev + 1
                break
        if not cutoff and usages and usages[0][1] > recency_tokens:
            cutoff = usages[0][0] + 1
        cutoff = paired_cutoff(messages, cutoff)
        if not any(m.get("type") not in {"comment", "reasoning"} for m in messages[:cutoff]):
            return 0
        return cutoff

    @staticmethod
    def strip_usage_comments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop `usage` comments. The recorded `input_tokens` only reflect
        the file state at the original call site; once messages are carried
        into a new version they're misleading. Call before writing kept
        messages to the next version."""
        return [m for m in messages
                if not (m.get("type") == "comment" and m.get("kind") == "usage")]
