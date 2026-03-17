"""LiteLLM callback: auto-discovers workspace *.md files and injects as system prompt."""

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union


def discover_workspace_md(workspace_dir: Path) -> list[Path]:
    """Find all .md files in workspace root (not subdirectories), sorted alphabetically."""
    workspace = Path(workspace_dir)
    return sorted(f for f in workspace.glob("*.md") if f.is_file())


def build_system_message(workspace_dir: Path) -> str:
    """Read and concatenate all workspace markdown files into a system prompt."""
    files = discover_workspace_md(workspace_dir)
    parts = []
    for f in files:
        parts.append(f.read_text())
    return "\n\n".join(parts)


# Attempt real inheritance when litellm is available (i.e., in production).
# Falls back to a plain class so the module can be imported in test environments
# where libstdc++.so is not present.
try:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.caching.dual_cache import DualCache
    _BASE = CustomLogger
except Exception:  # pragma: no cover – only happens in test environments
    _BASE = object  # type: ignore[assignment,misc]


class SystemPromptInjector(_BASE):  # type: ignore[valid-type,misc]
    """LiteLLM callback that prepends workspace markdown as system prompt.

    Reads WORKSPACE_DIR from environment. Set this in docker-compose.
    """

    def __init__(self):
        super().__init__()
        self.workspace_dir = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Literal[
            "completion", "text_completion", "embeddings",
            "image_generation", "moderation", "audio_transcription",
            "pass_through_endpoint", "rerank",
        ],
    ) -> Optional[Union[dict, str]]:
        if call_type != "completion":
            return None

        system_content = build_system_message(self.workspace_dir)
        if not system_content:
            return None

        system_msg = {"role": "system", "content": system_content}
        data["messages"] = [system_msg] + data.get("messages", [])
        return None


# Module-level instance — LiteLLM imports this via the callbacks config
proxy_handler_instance = SystemPromptInjector()
