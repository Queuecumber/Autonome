"""Binary store — filename-addressed files with collision-avoidance suffixes.

Tool outputs containing binary content get saved here and referenced by
"pointer" (the filename). When the agent passes a pointer back to a tool
that accepts bytes, the orchestrator resolves it before dispatching.
"""

import logging
import re
import time
from pathlib import Path

import filetype

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400


def _sanitize(name: str) -> str:
    """Strip path separators and non-safe characters from a filename."""
    name = Path(name).name  # drop any path components
    return re.sub(r"[^\w.\-]", "_", name) or "blob"


def _extension_for(mime_type: str) -> str:
    """Return the canonical extension for a MIME type (e.g. 'jpg'), or 'bin'."""
    kind = filetype.get_type(mime=mime_type)
    return kind.extension if kind else "bin"


class BinaryStore:
    """Persistent store for tool binaries, keyed by filename."""

    def __init__(self, store_dir: Path, retention_days: int = 30):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days

    def save(self, content: bytes, mime_type: str, filename: str | None = None) -> str:
        """Write bytes to disk and return the pointer (the on-disk filename).

        If a filename is provided it's preferred (sanitized). On collision, a
        numeric suffix is appended (foo.jpg, foo_0.jpg, foo_1.jpg, ...).
        """
        base = _sanitize(filename) if filename else f"blob.{_extension_for(mime_type)}"
        path = self.store_dir / base
        counter = 0
        while path.exists():
            stem, dot, ext = base.partition(".")
            path = self.store_dir / f"{stem}_{counter}{dot}{ext}"
            counter += 1

        path.write_bytes(content)
        logger.info(f"Saved binary {path.name} ({len(content)} bytes, {mime_type})")
        return path.name

    def load(self, pointer: str) -> tuple[bytes, str]:
        """Read a pointer back to (bytes, mime_type). Raises FileNotFoundError."""
        path = self.store_dir / _sanitize(pointer)
        if not path.is_file():
            raise FileNotFoundError(f"No binary with pointer {pointer!r}")
        raw = path.read_bytes()
        kind = filetype.guess(raw)
        return raw, kind.mime if kind else "application/octet-stream"

    def gc(self) -> int:
        """Delete files older than retention_days. Returns number removed."""
        cutoff = time.time() - (self.retention_days * SECONDS_PER_DAY)
        removed = 0
        for path in self.store_dir.iterdir():
            if not path.is_file():
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        if removed:
            logger.info(f"GC: removed {removed} binaries older than {self.retention_days}d")
        return removed
