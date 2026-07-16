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

    def get_api_key(self) -> str:
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("API key required: pass api_key or set OPENROUTER_API_KEY")
        return key

    def build_client(self) -> OpenAI:
        return OpenAI(api_key=self.get_api_key(), base_url=self.base_url)
