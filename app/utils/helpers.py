"""
app/utils/helpers.py
─────────────────────
Shared utility functions.
Keep this thin — if a helper grows complex, move it to a dedicated service.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ── ID / hashing ──────────────────────────────────────────────────

def new_id() -> str:
    """Generate a compact unique ID (no dashes)."""
    return uuid.uuid4().hex


def stable_hash(text: str, length: int = 16) -> str:
    """Deterministic short hash — used for cache keys."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


# ── Text utilities ────────────────────────────────────────────────

def truncate(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_len chars, appending suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Basic text normalisation:
    - Collapse multiple whitespace/newlines
    - Strip leading/trailing whitespace
    - Remove null bytes (common in bad PDFs)
    """
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate without calling a tokenizer.
    Rule of thumb: ~0.75 tokens per word, ~4 chars per token.
    """
    return max(len(text) // 4, 1)


# ── Cost estimation ───────────────────────────────────────────────

# Approximate costs per 1M tokens (USD) — update as pricing changes
_TOKEN_COSTS: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate API cost in USD for a request."""
    costs = _TOKEN_COSTS.get(model, {"input": 3.0, "output": 15.0})
    return (
        input_tokens * costs["input"] / 1_000_000
        + output_tokens * costs["output"] / 1_000_000
    )


# ── Timing ────────────────────────────────────────────────────────

class Timer:
    """Context manager for measuring elapsed time in ms."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = int((time.monotonic() - self._start) * 1000)


# ── Validation helpers ────────────────────────────────────────────

def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email))


def sanitize_filename(filename: str) -> str:
    """
    Remove dangerous characters from filename.
    Keeps alphanumeric, dots, dashes, underscores.
    Prevents directory traversal.
    """
    # Replace path separators and null bytes first
    base = filename.replace("/", "_").replace("\\", "_").replace("\x00", "_")
    # Remove any remaining dangerous chars (keep alphanum, dot, dash, underscore)
    base = re.sub(r"[^\w.\-]", "_", base)
    # Prevent directory traversal: collapse consecutive dots
    base = re.sub(r"\.{2,}", "_", base)
    return base[:255]  # filesystem limit


# ── Response helpers ──────────────────────────────────────────────

def paginate(
    items: list[Any],
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Simple offset pagination."""
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "has_next": end < total,
        "has_prev": page > 1,
    }
