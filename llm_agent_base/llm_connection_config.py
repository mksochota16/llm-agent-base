import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class LLMConnectionConfig:
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    embedding_model: str = "openai/text-embedding-3-small"
    max_retries: int = 3
    timeout: Optional[float] = None

    def get_api_key(self) -> str:
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("API key required: pass api_key or set OPENROUTER_API_KEY")
        return key

    def build_client(self) -> OpenAI:
        """Build the OpenAI client.

        ``max_retries`` uses the SDK's own exponential backoff with jitter,
        which honours ``Retry-After`` and covers connection errors, timeouts,
        429s and 5xx. The SDK default is 2; this raises it to 3.
        """
        kwargs: dict = {
            "api_key": self.get_api_key(),
            "base_url": self.base_url,
            "max_retries": self.max_retries,
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return OpenAI(**kwargs)
