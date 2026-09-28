"""Context-overflow recognition and safe transcript boundaries for recovery."""

import json
from typing import Any

from openai import APIStatusError


def context_limit_error(error: Exception) -> bool:
    """Recognize context-specific HTTP failures, not arbitrary invalid requests.

    Args:
        error: Exception raised while establishing a model request.

    Returns:
        True for known context-limit codes/messages on HTTP 400/413, including
        the nested LiteLLM message used by the configured NVIDIA gateway.
    """
    if not isinstance(error, APIStatusError) or error.status_code not in {400, 413}:
        return False
    if isinstance(error.code, str) and error.code in {"context_length_exceeded", "context_window_exceeded", "prompt_too_long"}:
        return True
    text = str(error).casefold()
    return any(message in text for message in (
        "input exceeds the context window", "maximum context length",
        "context length exceeded", "context window exceeded", "prompt is too long"))


def split_for_recovery(items: list[dict[str, Any]], keep_chars: int) -> tuple[list[dict], list[dict]]:
    """Choose a bounded recent tail without cutting a tool-call/result exchange.

    Args:
        items: Persisted-format transcript, including complete current-turn work.
        keep_chars: Soft JSON-character budget for verbatim recent items.

    Returns:
        Older items to summarize and the retained tail. An oversized newest
        exchange is summarized as a whole instead of leaving orphaned tool results.
        Event metadata stays with its following user text; reasoning and usage
        markers stay with their following assistant response when possible.
    """
    boundaries = [0]
    pending = set()
    for offset, item in enumerate(items):
        kind = item.get("type")
        if kind == "function_call":
            pending.add(item.get("call_id"))
        elif kind == "function_call_output":
            pending.discard(item.get("call_id"))
        following = items[offset + 1] if offset + 1 < len(items) else {}
        paired = ((item.get("role") == "developer" and following.get("role") == "user")
                  or kind in {"reasoning", "comment"}
                  or (item.get("role") == "assistant" and following.get("type") == "function_call"))
        if not pending and not paired:
            boundaries.append(offset + 1)
    if boundaries[-1] != len(items):
        boundaries.append(len(items))
    size, split = 0, len(items)
    for start, end in reversed(list(zip(boundaries, boundaries[1:]))):
        length = len(json.dumps(items[start:end], ensure_ascii=False))
        if size + length > keep_chars:
            break
        size += length
        split = start
    return items[:split], items[split:]


def recovery_source(items: list[dict[str, Any]]) -> str:
    """Serialize conversation/action transcript items for a tool-free summarizer.

    Reasoning and usage comments are excluded, as in normal compaction. Source
    items are left unchanged; the complete originals remain in the session audit.
    """
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items
                     if item.get("type") not in {"reasoning", "comment"})


def recovery_media(messages: list[dict[str, Any]], *, discard: bool = False) -> tuple[list[dict], int]:
    """Keep only the most recent image batch, or discard it on a stronger recovery.

    Args:
        messages: Actual outgoing Chat Completions messages, including transient images.
        discard: Omit all images, with the omission count recorded in the checkpoint.

    Returns:
        Image content parts to retain and a count of omitted image parts. No image
        data is persisted in the compacted transcript or sent to the summarizer.
    """
    batches = [[part for part in message["content"] if isinstance(part, dict) and part.get("type") == "image_url"]
               for message in messages if isinstance(message.get("content"), list)]
    batches = [batch for batch in batches if batch]
    kept = batches[-1] if batches and not discard else []
    return kept, sum(map(len, batches)) - len(kept)
