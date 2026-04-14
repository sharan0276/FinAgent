from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "openai/gpt-4.1-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REPO_ROOT = Path(__file__).resolve().parent.parent


load_dotenv(REPO_ROOT / ".env")


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter configuration or calls fail."""


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: int = 90,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], str]:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_error = "OpenRouter request failed."
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    OPENROUTER_URL,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                body = response.json()
                if "error" in body:
                    raise OpenRouterError(str(body["error"]))
                choice = body.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise OpenRouterError("OpenRouter returned an empty message.")
                parsed = json.loads(content)
                model_name = str(body.get("model") or self.model)
                return parsed, model_name
            except (requests.RequestException, json.JSONDecodeError, OpenRouterError) as exc:
                last_error = str(exc)
                if attempt == self.max_retries:
                    break
                time.sleep(min(2 * attempt, 4))

        raise OpenRouterError(last_error)
