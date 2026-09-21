"""Bounded model-request retries with a process-local provider cooldown."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import logging
import math
import random
import re
import time
from typing import TypeVar

from openai import APIConnectionError, APIStatusError

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Configure attempts and elapsed retry budget for each model request.

    Args:
        max_attempts: Total attempts, including the initial request; at least one.
        max_elapsed_seconds: Budget for starting retries, not an in-flight timeout.
        initial_delay_seconds: First fallback backoff before positive jitter.
        max_delay_seconds: Cap on fallback backoff, not on server-requested waits.

    Raises:
        ValueError: Limits are non-finite, non-positive, or inconsistent.
    """

    max_attempts: int = 8
    max_elapsed_seconds: float = 300
    initial_delay_seconds: float = 2
    max_delay_seconds: float = 60

    def __post_init__(self) -> None:
        """Reject invalid limits before any provider request is made."""
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 100:
            raise ValueError("model.retry.max_attempts must be an integer from 1 to 100")
        for name in ("max_elapsed_seconds", "initial_delay_seconds", "max_delay_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"model.retry.{name} must be a finite positive number")
        if self.initial_delay_seconds > self.max_delay_seconds:
            raise ValueError("model.retry.initial_delay_seconds must not exceed max_delay_seconds")


class RetryBudgetExceeded(RuntimeError):
    """A shared provider cooldown exceeds this request's remaining retry budget."""


def _positive_seconds(value: str | None) -> float | None:
    """Return a finite positive numeric delay, or None for an unusable value."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def retry_hint(error: APIStatusError) -> tuple[float, str] | None:
    """Read a server delay from response headers or an explicit rate-limit message.

    Args:
        error: Failed SDK request retaining its HTTP response and provider message.

    Returns:
        Positive seconds and a safe diagnostic source, or None. Header precedence
        is retry-after-ms, then Retry-After (seconds or HTTP date). The message
        fallback accepts only an explicit limit reset interval on HTTP 429.
    """
    headers = error.response.headers
    milliseconds = _positive_seconds(headers.get("retry-after-ms"))
    if milliseconds is not None:
        return milliseconds / 1000, "retry-after-ms"
    value = headers.get("retry-after")
    seconds = _positive_seconds(value)
    if seconds is not None:
        return seconds, "retry-after"
    if value:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
            if math.isfinite(seconds) and seconds > 0:
                return seconds, "retry-after-date"
        except (TypeError, ValueError, OverflowError):
            pass
    if error.status_code == 429:
        match = re.search(r"\blimit (?:will )?resets? in (\d+(?:\.\d+)?)\s*"
                          r"(milliseconds?|seconds?|minutes?)\b", str(error)[:32768], re.IGNORECASE)
        if match:
            seconds = _positive_seconds(match[1])
            if seconds is not None:
                scale = 0.001 if match[2].lower().startswith("milli") else 60 if match[2].lower().startswith("minute") else 1
                return seconds * scale, "reset-message"
    return None


def retryable(error: APIConnectionError | APIStatusError) -> bool:
    """Whether an SDK failure is transient; explicit refusal and quota errors stop retries."""
    if isinstance(error, APIConnectionError):
        return True
    if error.response.headers.get("x-should-retry", "").lower() == "false":
        return False
    if error.code in ("insufficient_quota", "billing_hard_limit_reached"):
        return False
    return error.status_code in (408, 409, 429) or error.status_code >= 500


class ModelRequests:
    """Share a cooldown across requests to one orchestrator's configured provider.

    Args:
        policy: Validated retry limits. SDK retries must be disabled on the client.
    """

    def __init__(self, policy: RetryPolicy):
        """Initialize retry policy without delaying the first request."""
        self.policy = policy
        self.not_before = 0.0

    async def _wait(self, seconds: float, cancel: asyncio.Event | None) -> bool:
        """Wait for a cooldown; return False if cooperative cancellation arrives."""
        if cancel is None:
            await asyncio.sleep(seconds)
            return True
        try:
            await asyncio.wait_for(cancel.wait(), timeout=seconds)
            return False
        except TimeoutError:
            return True

    async def run(self, request: Callable[[], Awaitable[T]], cancel: asyncio.Event | None = None) -> T | None:
        """Retry a single request without repeating any caller-side tool execution.

        Args:
            request: Factory for a fresh awaitable, including stream establishment
                but never iteration over a returned stream.
            cancel: Optional cooperative cancellation signal, checked during waits.

        Returns:
            The response, or None if cancelled before an attempt or during backoff.

        Raises:
            APIConnectionError, APIStatusError: Non-retryable or exhausted failure.
            RetryBudgetExceeded: Another request's cooldown exceeds this budget.
            asyncio.CancelledError: The caller task was cancelled.
            Exception: Other request failures propagate without retry.
        """
        deadline = time.monotonic() + self.policy.max_elapsed_seconds
        attempts = 0
        last_error = None
        while True:
            if cancel is not None and cancel.is_set():
                return None
            now = time.monotonic()
            delay = max(0.0, self.not_before - now)
            if now + delay >= deadline:
                if last_error is not None:
                    raise last_error
                raise RetryBudgetExceeded("Provider cooldown exceeds model retry budget")
            if delay:
                if not await self._wait(delay, cancel):
                    return None
                continue
            attempts += 1
            try:
                return await request()
            except (APIConnectionError, APIStatusError) as error:
                if not retryable(error):
                    raise
                hint = retry_hint(error) if isinstance(error, APIStatusError) else None
                delay, source = hint or (
                    min(self.policy.initial_delay_seconds * 2 ** (attempts - 1), self.policy.max_delay_seconds),
                    "exponential-backoff")
                delay += random.uniform(0, min(delay * 0.25, 1))
                self.not_before = max(self.not_before, time.monotonic() + delay)
                if attempts >= self.policy.max_attempts or self.not_before >= deadline:
                    raise
                logger.warning("Model request throttled or unavailable: status=%s attempt=%d/%d retry_in=%.2fs source=%s",
                               getattr(error, "status_code", "connection"), attempts,
                               self.policy.max_attempts, delay, source)
                last_error = error
