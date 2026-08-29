"""
Pandora Qwen Client

Qwen (通义千问) API client for model inference.
Compatible with DashScope and OpenAI-compatible endpoints.
"""

import os
import time
from typing import Any, Optional
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


class QwenClient(BaseModel):
    """
    Qwen API client for chat completion.

    Supports qwen-plus, qwen-max, qwen-turbo, and other Qwen models.
    Uses DashScope-compatible OpenAI-style endpoint.
    """

    def __init__(
        self,
        model_name: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        """
        Initialize Qwen client.

        Args:
            model_name: Qwen model name (e.g., "qwen-plus", "qwen-max")
            api_key: Qwen/DashScope API key
            base_url: Optional custom API base URL
                      (default: DashScope OpenAI-compatible endpoint)
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

        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.endpoint = f"{self.base_url}/chat/completions"
        self.logger = setup_logger("pandora.qwen")

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get Qwen API key from environment."""
        return os.environ.get("DASHSCOPE_API_KEY")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from Qwen with automatic retry on transient failures.

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

        if response.status_code != 200:
            self.logger.error(f"Qwen API error [{response.status_code}]: {response.text[:1000]}")
            self.logger.error(f"  Request payload: model={payload.get('model')}, "
                            f"temperature={payload.get('temperature')}, "
                            f"max_tokens={payload.get('max_tokens')}, "
                            f"messages_count={len(payload.get('messages', []))}")

        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        self.logger.debug(f"Qwen response: {content[:200]}...")
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
                    self.logger.error(f"Qwen API error (non-retriable): {e}")
                    raise

                self.logger.warning(
                    f"Qwen API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
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
