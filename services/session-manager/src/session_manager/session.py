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
import re
from pathlib import Path
from typing import Any


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
        """
        versions = self._versioned_files(session_id)
        next_n = (versions[-1][0] + 1) if versions else 0
        new_path = self.store_dir / f"{self._safe_id(session_id)}.{next_n}.jsonl"
        with new_path.open("w") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
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
        """Find the index where `messages[index:]` covers ~recency_tokens.

        Walks `usage` comments backward, summing deltas between consecutive
        comments. Each delta is the token cost of everything appended
        between those two LLM calls. When a single delta would push the
        cumulative *past* recency by more than it currently undershoots,
        prefer the earlier cutoff (less kept) — keeps the post-compaction
        state from blowing up when one iteration emitted a huge tool
        result.

        Returns 0 (keep all) if there isn't enough usage data to identify
        a cutoff.
        """
        usages: list[tuple[int, int]] = []
        for i, m in enumerate(messages):
            if (m.get("type") == "comment"
                    and m.get("kind") == "usage"
                    and m.get("input_tokens") is not None):
                usages.append((i, m["input_tokens"]))

        if len(usages) < 2:
            return 0

        cumulative = 0
        for i in range(len(usages) - 1, 0, -1):
            line_curr, tok_curr = usages[i]
            line_prev, tok_prev = usages[i - 1]
            cumulative += tok_curr - tok_prev
            if cumulative >= recency_tokens:
                # Always include the crossing delta in keep. Overshooting
                # recency_tokens is fine (keep is slightly larger than
                # target); undershooting means fold balloons toward the
                # summarize call's context window limit.
                return line_prev + 1
        return 0

    @staticmethod
    def strip_usage_comments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop `usage` comments. The recorded `input_tokens` only reflect
        the file state at the original call site; once messages are carried
        into a new version they're misleading. Call before writing kept
        messages to the next version."""
        return [m for m in messages
                if not (m.get("type") == "comment" and m.get("kind") == "usage")]
