"""Encoder registry, contract and the offline backend."""

from __future__ import annotations

import numpy as np
import pytest

from embedresearchgaps.config import EncoderConfig
from embedresearchgaps.encoders import (
    Encoder,
    PrecomputedEncoder,
    TfidfSvdEncoder,
    available_encoders,
    batched,
    get_encoder,
    l2_normalize,
    register_encoder,
)

TEXTS = [
    "supply chain resilience",
    "supply chain disruption",
    "employee engagement",
    "credit risk modelling",
    "bank capital buffer",
]


def test_batched_partitions_without_loss() -> None:
    assert [list(b) for b in batched(list(range(5)), 2)] == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError, match="batch size"):
        list(batched([1], 0))


def test_l2_normalize_leaves_zero_rows_intact() -> None:
    matrix = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(matrix[0], [0.6, 0.8])
    np.testing.assert_allclose(matrix[1], [0.0, 0.0])


class TestTfidfSvdEncoder:
    def test_shape_and_normalisation(self) -> None:
        encoder = TfidfSvdEncoder(EncoderConfig(name="tfidf-svd", dimensions=4))
        matrix = encoder.encode(TEXTS)
        assert matrix.shape[0] == len(TEXTS)
        assert matrix.shape[1] <= 4
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-9)

    def test_is_deterministic(self) -> None:
        encoder = TfidfSvdEncoder(EncoderConfig(name="tfidf-svd", dimensions=4))
        np.testing.assert_allclose(encoder.encode(TEXTS), encoder.encode(TEXTS))

    def test_lexical_overlap_raises_similarity(self) -> None:
        matrix = TfidfSvdEncoder(EncoderConfig(name="tfidf-svd", dimensions=8)).encode(TEXTS)
        related = float(matrix[0] @ matrix[1])      # both "supply chain ..."
        unrelated = float(matrix[0] @ matrix[2])    # "employee engagement"
        assert related > unrelated

    def test_normalisation_can_be_disabled(self) -> None:
        encoder = TfidfSvdEncoder(EncoderConfig(name="tfidf-svd", dimensions=4, normalize=False))
        norms = np.linalg.norm(encoder.encode(TEXTS), axis=1)
        assert not np.allclose(norms, 1.0)


class TestEncoderContract:
    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            TfidfSvdEncoder().encode([])

    def test_wrong_row_count_is_detected(self) -> None:
        class Broken(Encoder):
            name = "broken"

            def _encode(self, texts):  # type: ignore[no-untyped-def]
                return np.ones((len(texts) - 1, 3))

        with pytest.raises(ValueError, match="for 3 texts"):
            Broken().encode(["a", "b", "c"])

    def test_non_finite_output_is_detected(self) -> None:
        class Nan(Encoder):
            name = "nan"

            def _encode(self, texts):  # type: ignore[no-untyped-def]
                return np.full((len(texts), 2), np.nan)

        with pytest.raises(ValueError, match="non-finite"):
            Nan().encode(["a"])

    def test_none_values_are_coerced_to_empty_strings(self) -> None:
        matrix = TfidfSvdEncoder(EncoderConfig(name="tfidf-svd", dimensions=3)).encode(
            ["alpha beta", None, "gamma delta"]  # type: ignore[list-item]
        )
        assert matrix.shape[0] == 3


class TestPrecomputedEncoder:
    def test_returns_supplied_vectors(self) -> None:
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        encoder = PrecomputedEncoder(
            EncoderConfig(name="precomputed", normalize=False, extra={"vectors": vectors})
        )
        np.testing.assert_allclose(encoder.encode(["b", "a"]), [[0.0, 1.0], [1.0, 0.0]])

    def test_unknown_text_raises_rather_than_zero_filling(self) -> None:
        encoder = PrecomputedEncoder(
            EncoderConfig(name="precomputed", extra={"vectors": {"a": [1.0]}})
        )
        with pytest.raises(KeyError, match="no precomputed embedding"):
            encoder.encode(["a", "missing"])

    def test_missing_vectors_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="extra\\['vectors'\\]"):
            PrecomputedEncoder(EncoderConfig(name="precomputed")).encode(["a"])


class TestRegistry:
    def test_known_backends_are_listed(self) -> None:
        assert {"tfidf-svd", "openai", "nomic", "jina", "cohere", "sbert", "precomputed"} <= set(
            available_encoders()
        )

    def test_get_encoder_accepts_a_bare_name(self) -> None:
        assert isinstance(get_encoder("tfidf-svd"), TfidfSvdEncoder)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown encoder"):
            get_encoder("word2vec")

    def test_custom_backend_can_be_registered(self) -> None:
        class Constant(Encoder):
            name = "constant-test"

            def _encode(self, texts):  # type: ignore[no-untyped-def]
                return np.ones((len(texts), 2))

        register_encoder("constant-test", Constant)
        assert isinstance(get_encoder("constant-test"), Constant)

    def test_remote_backend_reports_a_missing_key(self) -> None:
        encoder = get_encoder(EncoderConfig(name="openai", api_key_env="ERG_ABSENT_KEY"))
        with pytest.raises(RuntimeError, match="ERG_ABSENT_KEY"):
            encoder.encode(["text"])

    def test_remote_backend_without_key_env_is_rejected(self) -> None:
        encoder = get_encoder(EncoderConfig(name="openai"))
        with pytest.raises(ValueError, match="api_key_env"):
            encoder.encode(["text"])
