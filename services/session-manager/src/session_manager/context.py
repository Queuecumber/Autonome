"""Context-overflow recognition and paired transcript boundaries for compaction."""

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


def paired_cutoff(items: list[dict[str, Any]], cutoff: int) -> int:
    """Move a proposed cutoff backwards to a complete conversation boundary.

    Args:
        items: Persisted-format transcript.
        cutoff: Proposed start of the retained recent context.

    Returns:
        A cutoff no later than requested. Tool batches, event metadata/text pairs,
        and reasoning/assistant records stay together. Zero means retain all.
    """
    boundary = 0
    pending = set()
    for offset, item in enumerate(items):
        if offset >= cutoff:
            break
        kind = item.get("type")
        if kind == "function_call":
            pending.add(item.get("call_id"))
        elif kind == "function_call_output":
            pending.discard(item.get("call_id"))
        following = items[offset + 1] if offset + 1 < len(items) else {}
        paired = ((item.get("role") == "developer" and following.get("role") == "user")
                  or kind == "reasoning"
                  or (item.get("role") == "assistant" and following.get("type") == "function_call"))
        if not pending and not paired:
            boundary = offset + 1
    return boundary


def retained_media(batches: dict[str, dict[str, Any]], call_ids: set[str]) -> tuple[dict[str, dict], int]:
    """Retain images from tool batches still present after compaction.

    Args:
        batches: Live image messages keyed by their completed tool batch's last call ID.
        call_ids: IDs of tool results in the retained transcript.

    Returns:
        Retained batches with their ownership intact and the omitted image count.
        Source data is unchanged; the mapping is never sent to the provider.
    """
    kept, omitted = {}, 0
    for call_id, message in batches.items():
        if call_id in call_ids:
            kept[call_id] = message
        else:
            omitted += sum(part.get("type") == "image_url" for part in message["content"])
    return kept, omitted
