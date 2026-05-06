from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import error, request


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    call_id: str | None = None


class LLMClient:
    """Optional OpenAI-compatible chat client.

    The optimization loop must work without an API key. When configured, the
    LLM is used as an advisory agent for mutation planning; validation still
    comes from compile, correctness, and benchmark agents.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
        self.base_url = self._normalize_base_url(os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "")
        self.model = os.getenv("LLM_MODEL") or os.getenv("LLM_MODE") or os.getenv("MODEL") or "gpt-4o-mini"
        self.timeout = float(os.getenv("LLM_TIMEOUT_SEC", "45"))
        self._client = None
        self._backend: str | None = None
        self._error: str | None = None
        self._last_request_error: str | None = None

        if not self.api_key:
            self._error = "LLM disabled: no API key in OPENAI_API_KEY or API_KEY."
            return

        try:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
            self._backend = "openai-sdk"
        except ModuleNotFoundError:
            self._backend = "http"
        except Exception as exc:  # pragma: no cover - depends on optional package
            self._backend = "http"
            self._error = f"OpenAI SDK unavailable, using HTTP fallback: {type(exc).__name__}: {exc}"

    @property
    def enabled(self) -> bool:
        return self._client is not None or self._backend == "http"

    @property
    def status(self) -> str:
        if self._client is not None:
            return f"LLM enabled with model {self.model} via OpenAI SDK."
        if self._backend == "http":
            suffix = f" ({self._error})" if self._error else ""
            return f"LLM enabled with model {self.model} via OpenAI-compatible HTTP fallback.{suffix}"
        return self._error or "LLM disabled."

    @property
    def last_request_error(self) -> str | None:
        return self._last_request_error

    def complete(self, system: str, user: str) -> LLMResponse | None:
        if not self.enabled:
            self._last_request_error = self._error or "LLM disabled."
            return None
        try:
            if self._client is not None:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                )
                content = response.choices[0].message.content or ""
                self._last_request_error = None
                return LLMResponse(content=content, model=self.model)
            return self._complete_http(system, user)
        except Exception as exc:  # pragma: no cover - depends on network
            self._error = f"LLM request failed: {type(exc).__name__}: {exc}"
            self._last_request_error = self._error
            return None

    def _complete_http(self, system: str, user: str) -> LLMResponse | None:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        req = request.Request(
            self._chat_completions_url(),
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            self._error = f"LLM HTTP request failed: HTTP {exc.code}: {detail}"
            self._last_request_error = self._error
            return None
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        self._last_request_error = None
        return LLMResponse(content=content, model=self.model)

    def _chat_completions_url(self) -> str:
        base = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _normalize_base_url(self, base_url: str) -> str:
        base = base_url.strip().rstrip("/")
        if not base:
            return ""
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"
