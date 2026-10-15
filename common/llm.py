"""LLM factory for OpenRouter or Ollama's hosted chat API."""

import json
import os

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class _OllamaStructured:
    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def _request(self, messages: list[dict]) -> dict:
        schema_text = json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
        return {
            "model": os.environ.get("LLM_MODEL", "gpt-oss:120b"),
            "messages": [{
                "role": "system",
                "content": (
                    "Return only one valid JSON object matching this JSON Schema. "
                    "Do not add Markdown fences or prose. Schema: " + schema_text
                ),
            }, *messages],
            "format": self.schema.model_json_schema(),
            "stream": False,
        }

    def _parse(self, response: httpx.Response) -> BaseModel:
        response.raise_for_status()
        content = response.json()["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            return self.schema.model_validate(json.loads(content))
        except (ValueError, TypeError) as exc:
            raise ValueError("Ollama did not return a valid structured review") from exc

    def invoke(self, messages: list[dict]) -> BaseModel:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://ollama.com/api/chat",
                headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
                json=self._request(messages),
            )
        return self._parse(response)

    async def ainvoke(self, messages: list[dict]) -> BaseModel:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://ollama.com/api/chat",
                headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
                json=self._request(messages),
            )
        return self._parse(response)


class _OllamaModel:
    def with_structured_output(self, schema: type[BaseModel]) -> _OllamaStructured:
        return _OllamaStructured(schema)


def get_llm(temperature: float = 0.2) -> ChatOpenAI | _OllamaModel:
    if os.environ.get("LLM_PROVIDER") == "ollama":
        if not os.environ.get("OLLAMA_API_KEY"):
            raise RuntimeError("OLLAMA_API_KEY is not set")
        return _OllamaModel()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — copy .env.example to .env")
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
    )
