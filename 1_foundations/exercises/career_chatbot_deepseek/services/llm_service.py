"""LLM service wrapper for DeepSeek.

Provides a clean interface for LLM operations,
abstracting away API details.
"""


from typing import Any

from openai import OpenAI

from config.settings import settings


class LLMService:

  def __init__(self, api_key: str | None = None, model: str | None = None):
   self.client = OpenAI(
    api_key=api_key or settings.deepseek_api_key,
    base_url="https://api.deepseek.com/v1"
   )
   self.model = model or settings.primary_model

  def chat_completion(
    self,
    messages: list[dict[str, str]],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None) -> Any:

    kwargs = {
      "model": self.model,
      "messages": messages,
      "temperature": temperature
    }

    if tools:
      kwargs["tools"] = tools

    if max_tokens:
      kwargs["max_tokens"] = max_tokens

    return self.client.chat.completions.create(**kwargs)
