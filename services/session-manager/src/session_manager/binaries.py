"""Binary store — counter-named files with optional filename preserved.

Tool outputs containing binary content get saved here and referenced by
"pointer" (the filename). When the agent passes a pointer back to a tool
that accepts bytes, the orchestrator resolves it before dispatching.
"""

import logging
import mimetypes
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    """Strip path separators and non-safe characters from a filename."""
    name = Path(name).name  # drop any path components
    return re.sub(r"[^\w.\-]", "_", name) or "blob"


class BinaryStore:
    """Counter-named persistent store for tool binaries."""

    def __init__(self, store_dir: Path, retention_days: int = 30):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._counter = self._scan_max_counter() + 1

    def _scan_max_counter(self) -> int:
        """Find the highest counter prefix in existing files."""
        highest = 0
        for path in self.store_dir.iterdir():
            if not path.is_file():
                continue
            prefix = path.name.split("-", 1)[0].split(".", 1)[0]
            if prefix.isdigit():
                highest = max(highest, int(prefix))
        return highest

    def save(self, content: bytes, mime_type: str, filename: str | None = None) -> str:
        """Write bytes to disk and return the pointer (the filename)."""
        counter = self._counter
        self._counter += 1

        if filename:
            pointer = f"{counter}-{_sanitize(filename)}"
        else:
            ext = mimetypes.guess_extension(mime_type or "") or ".bin"
            pointer = f"{counter}{ext}"

        (self.store_dir / pointer).write_bytes(content)
        logger.info(f"Saved binary {pointer} ({len(content)} bytes, {mime_type})")
        return pointer

    def load(self, pointer: str) -> tuple[bytes, str]:
        """Read a pointer back to (bytes, mime_type). Raises FileNotFoundError."""
        path = self.store_dir / _sanitize(pointer)
        if not path.is_file():
            raise FileNotFoundError(f"No binary with pointer {pointer!r}")
        mime_type, _ = mimetypes.guess_type(path.name)
        return path.read_bytes(), mime_type or "application/octet-stream"

    def gc(self) -> int:
        """Delete files older than retention_days. Returns number removed."""
        cutoff = time.time() - (self.retention_days * 86400)
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
