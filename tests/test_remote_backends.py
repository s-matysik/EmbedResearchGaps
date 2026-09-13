"""Request and response handling of the hosted embedding and chat backends.

``requests.post`` is replaced by a recorder, so these tests verify the payload
each backend builds and how it reads the reply without contacting any API.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from embedresearchgaps.config import EncoderConfig
from embedresearchgaps.encoders import get_encoder
from embedresearchgaps.validation import ChatAnnotator

KEY_VARIABLE = "ERG_TEST_KEY"


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class Recorder:
    """Captures every request and replies with a queued payload."""

    def __init__(self, *payloads: dict[str, Any]):
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, headers: dict[str, str] | None = None,
                 json: dict[str, Any] | None = None, timeout: int | None = None) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        return FakeResponse(payload)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_VARIABLE, "test-key-not-a-real-secret")


def patch_requests(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> None:
    import requests

    monkeypatch.setattr(requests, "post", recorder)


class TestEmbeddingBackends:
    def test_openai_payload_and_ordering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder(
            {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
        )
        patch_requests(monkeypatch, recorder)
        encoder = get_encoder(
            EncoderConfig(name="openai", api_key_env=KEY_VARIABLE, dimensions=2, normalize=False)
        )
        matrix = encoder.encode(["first", "second"])
        np.testing.assert_allclose(matrix, [[1.0, 0.0], [0.0, 1.0]])
        call = recorder.calls[0]
        assert call["url"].endswith("/v1/embeddings")
        assert call["json"]["model"] == "text-embedding-3-large"
        assert call["json"]["dimensions"] == 2
        assert call["headers"]["Authorization"].startswith("Bearer ")

    def test_openai_replaces_blank_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder({"data": [{"index": 0, "embedding": [1.0]}]})
        patch_requests(monkeypatch, recorder)
        get_encoder(EncoderConfig(name="openai", api_key_env=KEY_VARIABLE)).encode([""])
        assert recorder.calls[0]["json"]["input"] == [" "]

    def test_openai_batches_large_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder({"data": [{"index": 0, "embedding": [1.0]}]})
        patch_requests(monkeypatch, recorder)
        encoder = get_encoder(
            EncoderConfig(name="openai", api_key_env=KEY_VARIABLE, batch_size=2)
        )
        with pytest.raises(ValueError, match="for 3 texts"):
            encoder.encode(["a", "b", "c"])  # one vector per call -> 2 for 3 texts
        assert len(recorder.calls) == 2

    def test_nomic_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder({"embeddings": [[1.0, 0.0], [0.0, 1.0]]})
        patch_requests(monkeypatch, recorder)
        encoder = get_encoder(
            EncoderConfig(name="nomic", api_key_env=KEY_VARIABLE, dimensions=2, normalize=False)
        )
        matrix = encoder.encode(["a", "b"])
        assert matrix.shape == (2, 2)
        payload = recorder.calls[0]["json"]
        assert payload["model"] == "nomic-embed-text-v1.5"
        assert payload["task_type"] == "search_document"
        assert payload["dimensionality"] == 2

    def test_jina_payload_and_error_propagation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
        patch_requests(monkeypatch, recorder)
        encoder = get_encoder(EncoderConfig(name="jina", api_key_env=KEY_VARIABLE, normalize=False))
        encoder.encode(["a"])
        assert recorder.calls[0]["json"]["input"] == [{"text": "a"}]

        patch_requests(monkeypatch, Recorder({"error": "quota exceeded"}))
        with pytest.raises(RuntimeError, match="Jina API error"):
            encoder.encode(["a"])

    def test_cohere_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder({"embeddings": {"float": [[1.0, 0.0]]}})
        patch_requests(monkeypatch, recorder)
        encoder = get_encoder(EncoderConfig(name="cohere", api_key_env=KEY_VARIABLE, normalize=False))
        np.testing.assert_allclose(encoder.encode(["a"]), [[1.0, 0.0]])
        assert recorder.calls[0]["json"]["input_type"] == "classification"


class TestChatBackends:
    REPLY = {"annotations": [{"keyword": "alpha term", "novelty": 7, "label": "GAP"}]}

    def test_openai_style_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder(
            {"choices": [{"message": {"content": json.dumps(self.REPLY)}}]}
        )
        patch_requests(monkeypatch, recorder)
        annotator = ChatAnnotator("deepseek", key_env=KEY_VARIABLE)
        annotations = annotator.annotate(["alpha term"], "problem")
        assert annotations[0].label == "GAP"
        payload = recorder.calls[0]["json"]
        assert payload["temperature"] == 0.0
        assert payload["messages"][0]["role"] == "system"

    def test_anthropic_style_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder(
            {"content": [{"type": "text", "text": json.dumps(self.REPLY)}]}
        )
        patch_requests(monkeypatch, recorder)
        annotator = ChatAnnotator("anthropic", key_env=KEY_VARIABLE)
        assert annotator.annotate(["alpha term"], "problem")[0].novelty == 7.0
        headers = recorder.calls[0]["headers"]
        assert "x-api-key" in headers and headers["anthropic-version"] == "2023-06-01"

    def test_google_style_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder(
            {"candidates": [{"content": {"parts": [{"text": json.dumps(self.REPLY)}]}}]}
        )
        patch_requests(monkeypatch, recorder)
        annotator = ChatAnnotator("google", key_env=KEY_VARIABLE)
        assert annotator.annotate(["alpha term"], "problem")[0].is_gap
        assert recorder.calls[0]["url"].endswith(":generateContent")
        assert recorder.calls[0]["headers"]["x-goog-api-key"]

    def test_google_without_candidates_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_requests(monkeypatch, Recorder({"candidates": []}))
        with pytest.raises(RuntimeError, match="no candidates"):
            ChatAnnotator("google", key_env=KEY_VARIABLE).complete("prompt")

    @pytest.mark.parametrize(
        ("provider", "payload"),
        [
            ("deepseek", {"choices": [{"message": {"content": '{"annotations": []}'}}]}),
            ("anthropic", {"content": [{"type": "text", "text": '{"annotations": []}'}]}),
            ("google", {"candidates": [{"content": {"parts": [{"text": '{"annotations": []}'}]}}]}),
        ],
    )
    def test_temperature_none_omits_the_field(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, payload: dict[str, Any]
    ) -> None:
        recorder = Recorder(payload)
        patch_requests(monkeypatch, recorder)
        ChatAnnotator(provider, key_env=KEY_VARIABLE, temperature=None).complete("prompt")
        sent = recorder.calls[0]["json"]
        assert "temperature" not in sent
        assert "temperature" not in sent.get("generationConfig", {})

    def test_temperature_zero_is_sent_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder({"candidates": [{"content": {"parts": [{"text": '{"annotations": []}'}]}}]})
        patch_requests(monkeypatch, recorder)
        ChatAnnotator("google", key_env=KEY_VARIABLE).complete("prompt")
        assert recorder.calls[0]["json"]["generationConfig"]["temperature"] == 0.0

    def test_rate_limit_is_retried_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        monkeypatch.setattr(time, "sleep", lambda seconds: None)

        class Flaky(Recorder):
            def __call__(self, *args, **kwargs):
                super().__call__(*args, **kwargs)
                if len(self.calls) == 1:
                    return FakeResponse({"error": "rate limited"}, status=429)
                return FakeResponse({"choices": [{"message": {"content": '{"annotations": []}'}}]})

        recorder = Flaky({})
        patch_requests(monkeypatch, recorder)
        annotator = ChatAnnotator("deepseek", key_env=KEY_VARIABLE, retry_delay=0.0)
        annotator.complete("prompt")
        assert len(recorder.calls) == 2

    def test_persistent_rate_limit_raises_after_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time

        monkeypatch.setattr(time, "sleep", lambda seconds: None)

        class AlwaysLimited(Recorder):
            def __call__(self, *args, **kwargs):
                super().__call__(*args, **kwargs)
                return FakeResponse({"error": "rate limited"}, status=429)

        recorder = AlwaysLimited({})
        patch_requests(monkeypatch, recorder)
        annotator = ChatAnnotator(
            "deepseek", key_env=KEY_VARIABLE, max_retries=2, retry_delay=0.0
        )
        with pytest.raises(RuntimeError, match="HTTP 429"):
            annotator.complete("prompt")
        assert len(recorder.calls) == 3

    def test_custom_model_name_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder({"choices": [{"message": {"content": json.dumps(self.REPLY)}}]})
        patch_requests(monkeypatch, recorder)
        ChatAnnotator("xai", model="grok-test", key_env=KEY_VARIABLE).complete("prompt")
        assert recorder.calls[0]["json"]["model"] == "grok-test"
