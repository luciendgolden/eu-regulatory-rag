"""LLM integration — OpenAI GPT-4o and Ollama local fallback.

Configured via environment variables:
  LLM_PROVIDER=openai|ollama   (default: openai)
  LLM_MODEL=gpt-4o|llama3      (default: gpt-4o)
  OLLAMA_HOST=http://localhost:11434
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


class LLMService:
    """Abstract base for LLM backends."""

    def complete(self, system: str, user: str) -> str:
        """Return a single complete response string."""
        raise NotImplementedError

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield response chunks as they arrive."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


class OpenAILLMService(LLMService):
    """OpenAI chat-completion backend (default: gpt-4o)."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self.api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from exc
        return self._client

    def complete(self, system: str, user: str) -> str:
        client = self._get_client()
        logger.debug("OpenAI complete: model=%s", self.model)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        client = self._get_client()
        logger.debug("OpenAI stream: model=%s", self.model)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


class OllamaLLMService(LLMService):
    """Ollama local LLM backend via HTTP API."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3") -> None:
        self.host = host.rstrip("/")
        self.model = model

    def _get_client(self):
        try:
            import httpx  # type: ignore

            return httpx.Client(timeout=120.0)
        except ImportError as exc:
            raise ImportError(
                "httpx is required for Ollama. Install with: pip install httpx"
            ) from exc

    def complete(self, system: str, user: str) -> str:
        logger.debug("Ollama complete: model=%s host=%s", self.model, self.host)
        with self._get_client() as client:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            }
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

    def stream(self, system: str, user: str) -> Iterator[str]:
        logger.debug("Ollama stream: model=%s host=%s", self.model, self.host)
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "httpx is required for Ollama. Install with: pip install httpx"
            ) from exc

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
        }
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{self.host}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        break


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_service(
    provider: str | None = None,
    model: str | None = None,
    openai_api_key: str = "",
    ollama_host: str = "",
) -> LLMService:
    """Return the configured LLM service.

    Reads LLM_PROVIDER / LLM_MODEL from config if not passed explicitly.
    """
    if provider is None or model is None:
        try:
            from config import settings  # type: ignore

            provider = provider or settings.llm_provider
            model = model or settings.llm_model
            openai_api_key = openai_api_key or settings.openai_api_key
            ollama_host = ollama_host or settings.ollama_host
        except Exception:  # noqa: BLE001
            provider = provider or "openai"
            model = model or "gpt-4o"

    provider = (provider or "openai").lower()
    model = model or "gpt-4o"

    if provider == "openai":
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when using LLM_PROVIDER=openai."
            )
        logger.info("Using OpenAI LLM service (model=%s)", model)
        return OpenAILLMService(api_key=openai_api_key, model=model)

    if provider == "ollama":
        host = ollama_host or "http://localhost:11434"
        logger.info("Using Ollama LLM service (model=%s, host=%s)", model, host)
        return OllamaLLMService(host=host, model=model)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Expected 'openai' or 'ollama'."
    )
