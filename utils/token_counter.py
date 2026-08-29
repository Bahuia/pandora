"""
Token counting utilities for prompt length estimation.
"""

from typing import Optional

# Lazy-loaded tiktoken encodings
_encoding_cache = {}


def count_prompt_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in a prompt string.

    Uses tiktoken for OpenAI-compatible models; falls back to
    a character-based heuristic (~3.5 chars/token) if unavailable.

    Args:
        text: The prompt text to count tokens for.
        model: Model name hint (used to select tiktoken encoding).
               Ignored if tiktoken is not installed.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Try tiktoken first
    encoding = _get_encoding(model)
    if encoding is not None:
        return len(encoding.encode(text))

    # Fallback: ~3.5 chars per token (works well for mixed EN/ZH)
    return max(1, len(text) // 3 + 1)


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in a list of OpenAI-style chat messages.

    Also accounts for per-message overhead (~3 tokens per message,
    ~1 token for assistant priming, per OpenAI's token calculation).

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Model name hint.

    Returns:
        Total token count.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += count_prompt_tokens(content, model)
        # Per-message overhead
        total += 3  # role + formatting tokens

    total += 1  # assistant priming token
    return total


def _get_encoding(model: str):
    """
    Get a tiktoken encoding for the given model name.
    Returns None if tiktoken is not installed.
    """
    if model in _encoding_cache:
        return _encoding_cache[model]

    try:
        import tiktoken

        # Map common model names to tiktoken encodings
        if "gpt-4o" in model:
            enc_name = "o200k_base"
        elif "gpt-4" in model or "gpt-3.5" in model:
            enc_name = "cl100k_base"
        elif "davinci" in model:
            enc_name = "p50k_base"
        else:
            enc_name = "cl100k_base"  # default for newer models

        encoding = tiktoken.get_encoding(enc_name)
        _encoding_cache[model] = encoding
        return encoding
    except (ImportError, KeyError):
        _encoding_cache[model] = None
        return None
