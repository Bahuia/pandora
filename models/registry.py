"""
Pandora Model Registry

Factory for creating model clients by name.

Auto-detects provider from model name prefix:
  - gpt-4, gpt-3.5 → OpenAI
  - deepseek       → DeepSeek
  - qwen           → Qwen (通义千问)

Usage:
    model = ModelRegistry.create(model_name="qwen-plus", api_key="...")
    # No need to specify provider -- detected automatically.
    # OpenAI-compatible endpoints can be supplied explicitly with base_url.
"""

from typing import Optional

from .base import BaseModel
from .openai_client import OpenAIClient
from .deepseek_client import DeepSeekClient
from .qwen_client import QwenClient


class ModelRegistry:
    """
    Registry and factory for model clients.

    Creates appropriate model client based on model name prefix.
    """

    # Provider prefixes — order matters: longer/more specific prefixes first
    PROVIDERS = {
        "gpt-4": "openai",
        "gpt-3.5": "openai",
        "deepseek": "deepseek",
        "qwen": "qwen",
    }

    @classmethod
    def create(
        cls,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> BaseModel:
        """
        Create a model client instance.

        Args:
            model_name: Model name (e.g., "gpt-4o-mini", "deepseek-chat", "qwen3-8b")
            api_key: API key (uses env var if not provided)
            base_url: Optional custom API base URL (for direct providers)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds

        Returns:
            Model client instance

        Raises:
            ValueError: If model provider is not supported
        """
        provider = cls._detect_provider(model_name)

        if provider == "openai":
            return OpenAIClient(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif provider == "deepseek":
            return DeepSeekClient(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif provider == "qwen":
            return QwenClient(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unsupported model provider for '{model_name}'. Supported: openai, deepseek, qwen")

    @classmethod
    def _detect_provider(cls, model_name: str) -> str:
        """Detect provider from model name prefix."""
        model_lower = model_name.lower()

        for prefix, provider in cls.PROVIDERS.items():
            if model_lower.startswith(prefix):
                return provider

        # Default to OpenAI
        return "openai"

    @classmethod
    def register_provider(cls, prefix: str, provider: str) -> None:
        """
        Register a new provider prefix.

        Args:
            prefix: Model name prefix
            provider: Provider name
        """
        cls.PROVIDERS[prefix] = provider
