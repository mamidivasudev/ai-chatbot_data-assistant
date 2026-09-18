"""
LLM provider layer for the /summarize endpoints.

Ollama on the local GPU is the default. Switching to Google's Gemini API is a
deployment setting, not a code change:

  LLM_PROVIDER          ollama | gemini                 (default: ollama)
  OLLAMA_HOST           default http://127.0.0.1:11434
  OLLAMA_SUMMARY_MODEL  default llama3:latest
  OLLAMA_NUM_CTX        context window requested per call (default 8192)
  GEMINI_API_KEY        required when LLM_PROVIDER=gemini
  GEMINI_MODEL          default gemini-2.5-flash

Only text in, text out: callers build the prompts and parse the replies.
"""

import os
from typing import Optional

import httpx


class LLMError(RuntimeError):
    """The provider could not produce an answer (unreachable, bad status, empty reply)."""


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("OLLAMA_SUMMARY_MODEL", "llama3:latest")
        self.host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        # Ollama silently truncates any prompt longer than num_ctx, and its own
        # default window is small, so every call asks for this size explicitly.
        self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

    @property
    def max_input_chars(self) -> int:
        # About 4 characters per English token, keeping 40% of the window for
        # the instructions and the reply.
        return int(self.num_ctx * 4 * 0.6)

    async def complete(self, system: str, user: str, *, json_mode: bool = False,
                       temperature: float = 0.2, timeout: float = 300.0) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.host}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        content = (data.get("message") or {}).get("content", "")
        if not content.strip():
            raise LLMError("Ollama returned an empty reply.")
        return content

    async def status(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                installed = [m.get("model") or m.get("name") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as exc:
            return {"host": self.host, "reachable": False, "model_available": False, "detail": str(exc)}
        # Ollama lists "llama3" as "llama3:latest".
        tagged = self.model if ":" in self.model else f"{self.model}:latest"
        return {"host": self.host, "reachable": True, "model_available": tagged in installed, "installed_models": installed}


class GeminiProvider:
    name = "gemini"
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set.")

    # Gemini's window is large enough that almost every document fits in a
    # single call; this cap only bounds request size and cost.
    max_input_chars = 400_000

    async def complete(self, system: str, user: str, *, json_mode: bool = False,
                       temperature: float = 0.2, timeout: float = 300.0) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.API_URL.format(model=self.model),
                    json=body,
                    # Header, not query string, so the key never lands in logs or error text.
                    headers={"x-goog-api-key": self.api_key},
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Gemini returned no content: {data.get('promptFeedback') or data}")
        content = "".join(p.get("text", "") for p in parts)
        if not content.strip():
            raise LLMError("Gemini returned an empty reply.")
        return content

    async def status(self) -> dict:
        return {"reachable": None, "model_available": None, "detail": "API key configured; not probed."}


def get_provider(model: Optional[str] = None):
    """Build the provider named by LLM_PROVIDER; `model` overrides its default model."""
    name = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    if name == "gemini":
        return GeminiProvider(model)
    if name == "ollama":
        return OllamaProvider(model)
    raise LLMError(f"Unknown LLM_PROVIDER '{name}'. Use 'ollama' or 'gemini'.")
