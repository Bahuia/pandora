"""
Pandora Base Model Interface

Abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseModel(ABC):
    """
    Abstract base class for language model clients.

    All model providers (OpenAI, DeepSeek, etc.) must implement this interface.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key_from_env()
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    def _get_api_key_from_env(self) -> Optional[str]:
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the model.
        Must implement automatic retry on transient failures.
        """
        pass

    def _build_messages(
        self,
        prompt: str,
        system_message: Optional[str] = None,
    ) -> list[dict[str, str]]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return messages
