"""LLM annotators used to validate candidate gaps.

Every provider is reached through an OpenAI-compatible or provider-native
chat endpoint at ``temperature=0``, which is what makes the validation
reproducible.  Credentials are read from environment variables; nothing is
persisted.

Some current models reject a ``temperature`` parameter outright (it is
deprecated for them) or accept only ``1``.  Pass ``temperature=None`` to omit
the field; the annotations are then no longer strictly deterministic, so the
protocol fidelity of such a model should be reported alongside its results.

Registered providers
--------------------
============ ==================================== =========================
Name         Endpoint                             Key variable (default)
============ ==================================== =========================
``openai``   api.openai.com                       ``OPENAI_API_KEY``
``anthropic``api.anthropic.com                    ``ANTHROPIC_API_KEY``
``deepseek`` api.deepseek.com                     ``DEEPSEEK_API_KEY``
``moonshot`` api.moonshot.ai (Kimi)               ``MOONSHOT_API_KEY``
``xai``      api.x.ai (Grok)                      ``XAI_API_KEY``
``google``   generativelanguage.googleapis.com    ``GOOGLE_API_KEY``
``callable`` any local function                   --
============ ==================================== =========================

``CallableAnnotator`` wraps an arbitrary ``str -> str`` function, which keeps
the validation module testable offline and lets users plug in a model that is
only reachable from their own environment.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .protocol import Annotation, SYSTEM_PROMPT, build_prompt, parse_annotation_payload

__all__ = [
    "Annotator",
    "ChatAnnotator",
    "CallableAnnotator",
    "PROVIDERS",
    "get_annotator",
    "available_providers",
]


@dataclass
class ProviderSpec:
    """How to talk to one chat provider."""

    url: str
    default_model: str
    key_env: str
    style: str = "openai"  # 'openai' | 'anthropic' | 'google'
    extra_headers: dict[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "https://api.openai.com/v1/chat/completions", "gpt-4o", "OPENAI_API_KEY"
    ),
    "deepseek": ProviderSpec(
        "https://api.deepseek.com/chat/completions", "deepseek-chat", "DEEPSEEK_API_KEY"
    ),
    "moonshot": ProviderSpec(
        "https://api.moonshot.ai/v1/chat/completions", "kimi-k2-0905-preview", "MOONSHOT_API_KEY"
    ),
    "xai": ProviderSpec(
        "https://api.x.ai/v1/chat/completions", "grok-4", "XAI_API_KEY"
    ),
    "anthropic": ProviderSpec(
        "https://api.anthropic.com/v1/messages",
        "claude-sonnet-4-5",
        "ANTHROPIC_API_KEY",
        style="anthropic",
        extra_headers={"anthropic-version": "2023-06-01"},
    ),
    "google": ProviderSpec(
        "https://generativelanguage.googleapis.com/v1beta/models",
        "gemini-2.5-flash",
        "GOOGLE_API_KEY",
        style="google",
    ),
}


def available_providers() -> list[str]:
    return sorted(PROVIDERS)


class Annotator(ABC):
    """Labels keywords as GAP / EXPLORED / LOW IMPORTANCE / METHOD / NOISE."""

    def __init__(self, name: str, batch_size: int = 25):
        self.name = name
        self.batch_size = batch_size

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send one prompt and return the raw reply text."""

    def annotate(
        self,
        keywords: Sequence[str],
        problem_description: str,
        domain: str | None = None,
        keyword_set: str = "",
    ) -> list[Annotation]:
        """Annotate ``keywords`` in batches of ``batch_size``.

        A batch whose reply yields no usable entry is requested once more
        before the error is raised: an unparseable reply is an occasional
        accident of generation rather than a property of the batch, and losing
        a whole study to one of them would be wasteful.
        """
        keywords = [str(kw) for kw in keywords]
        annotations: list[Annotation] = []
        for start in range(0, len(keywords), self.batch_size):
            batch = keywords[start : start + self.batch_size]
            prompt = build_prompt(batch, problem_description, domain)
            parsed: list[Annotation] = []
            error: Exception | None = None
            for _ in range(2):
                try:
                    parsed = parse_annotation_payload(
                        self.complete(prompt), batch, self.name, keyword_set
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    error, parsed = exc, []
                if parsed:
                    break
            if not parsed and error is not None:
                raise error
            annotations.extend(parsed)
        return annotations


class ChatAnnotator(Annotator):
    """Annotator backed by a hosted chat-completions endpoint."""

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        key_env: str | None = None,
        temperature: float | None = 0.0,
        batch_size: int = 25,
        timeout: int = 300,
        name: str | None = None,
        max_retries: int = 4,
        retry_delay: float = 2.0,
        max_output_tokens: int = 16384,
    ):
        if provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; available: {available_providers()}"
            )
        self.spec = PROVIDERS[provider]
        self.provider = provider
        self.model = model or self.spec.default_model
        self.key_env = key_env or self.spec.key_env
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_output_tokens = max_output_tokens
        super().__init__(name or f"{provider}:{self.model}", batch_size)

    def _key(self) -> str:
        key = os.environ.get(self.key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"environment variable {self.key_env!r} is empty; export the "
                f"{self.provider} API key before running the validation"
            )
        return key

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, object]):
        """POST with bounded exponential back-off on rate limits and 5xx."""
        import time

        import requests

        delay = self.retry_delay
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    last = RuntimeError(
                        f"{self.name} returned HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        delay *= 2
                        continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:  # transport-level failure
                last = exc
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise last if last else RuntimeError(f"{self.name}: request failed")

    def complete(self, prompt: str) -> str:
        key = self._key()
        if self.spec.style == "anthropic":
            headers = {"x-api-key": key, "content-type": "application/json", **self.spec.extra_headers}
            payload = {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            response = self._post(self.spec.url, headers, payload)
            blocks = response.json().get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

        if self.spec.style == "google":
            url = f"{self.spec.url}/{self.model}:generateContent"
            payload = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": self.max_output_tokens,
                    "responseMimeType": "application/json",
                },
            }
            if self.temperature is not None:
                payload["generationConfig"]["temperature"] = self.temperature
            response = self._post(
                url, {"content-type": "application/json", "x-goog-api-key": key}, payload
            )
            candidates = response.json().get("candidates", [])
            if not candidates:
                raise RuntimeError(f"{self.name} returned no candidates")
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(part.get("text", "") for part in parts)

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        response = self._post(self.spec.url, headers, payload)
        return response.json()["choices"][0]["message"]["content"]


class CallableAnnotator(Annotator):
    """Annotator wrapping any ``str -> str`` completion function.

    Useful for models reachable only from the caller's own runtime, and for
    unit tests, which can supply a deterministic stub.
    """

    def __init__(self, name: str, completion: Callable[[str], str], batch_size: int = 25):
        super().__init__(name, batch_size)
        self._completion = completion

    def complete(self, prompt: str) -> str:
        return self._completion(prompt)


def get_annotator(provider: str, **kwargs: object) -> Annotator:
    """Instantiate a :class:`ChatAnnotator` for ``provider``."""
    return ChatAnnotator(provider, **kwargs)  # type: ignore[arg-type]
