"""
Pandora DeepSeek Client

DeepSeek API client for model inference.
"""

import os
import time
from typing import Any, Optional
import sys
sys.path.append("..")
import requests

from .base import BaseModel
from utils.logger import setup_logger


def _is_transient_error(e: Exception) -> bool:
    """Check if an exception is a transient/network error worth retrying."""
    error_str = str(e).lower()
    return any([
        isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
        "429" in error_str,
        "rate limit" in error_str,
        "timeout" in error_str,
        "connection" in error_str,
        "500" in error_str,
        "502" in error_str,
        "503" in error_str,
        "504" in error_str,
    ])




class DeepSeekClient(BaseModel):
    """
    DeepSeek API client for chat completion.

    Supports deepseek-chat, deepseek-coder, and other DeepSeek models.
    """

    def __init__(
        self,
        model_name: str = "deepseek-reasoner",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        """
        Initialize DeepSeek client.

        Args:
            model_name: DeepSeek model name (e.g., "deepseek-chat")
            api_key: DeepSeek API key
            base_url: Optional custom API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        self.base_url = base_url or "https://api.deepseek.com"
        self.endpoint = f"{self.base_url}/chat/completions"
        self.logger = setup_logger("pandora.deepseek")

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get DeepSeek API key from environment."""
        return os.environ.get("DEEPSEEK_API_KEY")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from DeepSeek with automatic retry on transient failures.

        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional API parameters
                max_retries (int): Max retry attempts, default 3
                retry_delay (float): Base delay between retries in seconds, default 5.0

        Returns:
            Generated text response
        """
        max_retries = kwargs.pop("max_retries", 3)
        retry_delay = kwargs.pop("retry_delay", 5.0)

        return self._generate_with_retry(prompt, system_message, max_retries, retry_delay, **kwargs)

    def _generate_once(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Single attempt (no retry)."""
        messages = self._build_messages(prompt, system_message)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        self.logger.debug(f"DeepSeek response: {content[:200]}...")
        return content

    def _generate_with_retry(
        self,
        prompt: str,
        system_message: Optional[str],
        max_retries: int,
        retry_delay: float,
        **kwargs: Any,
    ) -> str:
        """Internal retry loop with exponential backoff."""
        last_error = None

        for attempt in range(max_retries):
            try:
                return self._generate_once(prompt, system_message, **kwargs)
            except Exception as e:
                last_error = e

                if not _is_transient_error(e):
                    self.logger.error(f"DeepSeek API error (non-retriable): {e}")
                    raise

                self.logger.warning(
                    f"DeepSeek API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {retry_delay * (attempt + 1)}s..."
                )

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        raise last_error

    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        **kwargs: Any,
    ) -> str:
        """
        Generate with automatic retry on failure.

        Args:
            prompt: User prompt
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries in seconds
            **kwargs: Additional arguments

        Returns:
            Generated text response
        """
        return self._generate_with_retry(prompt, None, max_retries, retry_delay, **kwargs)
