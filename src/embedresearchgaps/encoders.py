"""Embedding backends.

Every backend implements :class:`Encoder` and is reachable by name through
:func:`get_encoder`.  Remote backends read their credentials from an
environment variable named in :class:`~embedresearchgaps.config.EncoderConfig`;
API keys are never stored in a configuration object or on disk.

Backends
--------
``tfidf-svd``
    Offline, dependency-light and fully deterministic.  Fits a word plus
    character n-gram TF-IDF on the texts of the run and reduces it with
    truncated SVD.  Requires no network access, which makes it the default
    for tests and for reproducing a run without API credentials.
``sbert``
    Any ``sentence-transformers`` checkpoint, e.g. ``all-mpnet-base-v2``.
``openai``, ``cohere``, ``jina``, ``nomic``
    Hosted embedding APIs; ``nomic-embed-text-v1.5`` is the model used in the
    two conference papers.
``precomputed``
    Looks vectors up in a user-supplied mapping, for corpora that already
    carry an embedding column.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import EncoderConfig

__all__ = [
    "Encoder",
    "TfidfSvdEncoder",
    "SentenceTransformerEncoder",
    "OpenAIEncoder",
    "CohereEncoder",
    "JinaEncoder",
    "NomicEncoder",
    "PrecomputedEncoder",
    "get_encoder",
    "register_encoder",
    "available_encoders",
    "l2_normalize",
    "batched",
]


def batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Yield consecutive slices of ``items`` of at most ``size`` elements."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit Euclidean norm, leaving zero rows untouched."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class Encoder(ABC):
    """Turns texts into a dense matrix of shape ``(len(texts), dim)``."""

    name: str = "encoder"

    def __init__(self, config: EncoderConfig | None = None):
        self.config = config or EncoderConfig(name=self.name)

    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Backend-specific encoding, without normalisation or validation."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode ``texts``, optionally L2-normalising the result.

        Raises
        ------
        ValueError
            If ``texts`` is empty or the backend returns the wrong number of
            vectors -- a silent length mismatch would misalign every keyword
            with its embedding.
        """
        texts = [("" if t is None else str(t)) for t in texts]
        if not texts:
            raise ValueError("cannot encode an empty list of texts")
        matrix = np.asarray(self._encode(texts), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(
                f"{self.name} returned {matrix.shape} for {len(texts)} texts"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{self.name} returned non-finite embedding values")
        return l2_normalize(matrix) if self.config.normalize else matrix

    def describe(self) -> dict[str, object]:
        return {"encoder": self.name, "model": self.config.model}

    def _api_key(self) -> str:
        variable = self.config.api_key_env
        if not variable:
            raise ValueError(
                f"{self.name} needs EncoderConfig.api_key_env to name the "
                "environment variable holding the API key"
            )
        key = os.environ.get(variable, "").strip()
        if not key:
            raise RuntimeError(
                f"environment variable {variable!r} is empty; export the API "
                f"key before using the {self.name} encoder"
            )
        return key


class TfidfSvdEncoder(Encoder):
    """Deterministic offline encoder: word+char TF-IDF followed by SVD.

    The representation is fitted on the texts passed to :meth:`encode`, so it
    is transductive: encoding the same texts always yields the same matrix,
    but vectors are not comparable across separate calls with different
    inputs.  Both pipelines encode articles once and keywords once, so this
    is sufficient for a self-contained, network-free run.
    """

    name = "tfidf-svd"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_union

        target_dim = int(self.config.dimensions or 128)
        vectoriser = make_union(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True),
        )
        sparse = vectoriser.fit_transform(texts)
        max_components = max(1, min(sparse.shape) - 1)
        n_components = max(1, min(target_dim, max_components))
        if n_components < 2:
            return np.asarray(sparse.todense())
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=int(self.config.extra.get("random_state", 42)),
            algorithm="arpack",
        )
        return svd.fit_transform(sparse)


class SentenceTransformerEncoder(Encoder):
    """Local ``sentence-transformers`` checkpoint."""

    name = "sbert"
    default_model = "sentence-transformers/all-mpnet-base-v2"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model_name = self.config.model or self.default_model
        model = SentenceTransformer(model_name)
        return model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            show_progress_bar=bool(self.config.extra.get("progress", False)),
            convert_to_numpy=True,
        )


class OpenAIEncoder(Encoder):
    """OpenAI embeddings endpoint (``text-embedding-3-large`` by default)."""

    name = "openai"
    default_model = "text-embedding-3-large"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        import requests

        url = self.config.extra.get("base_url", "https://api.openai.com/v1") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        vectors: list[list[float]] = []
        for batch in batched(list(texts), min(self.config.batch_size, 256)):
            payload: dict[str, object] = {
                "model": self.config.model or self.default_model,
                "input": [t if t.strip() else " " for t in batch],
            }
            if self.config.dimensions:
                payload["dimensions"] = self.config.dimensions
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in data)
        return np.asarray(vectors)


class CohereEncoder(Encoder):
    """Cohere ``embed`` endpoint."""

    name = "cohere"
    default_model = "embed-english-v3.0"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        import requests

        url = "https://api.cohere.com/v2/embed"
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        vectors: list[list[float]] = []
        for batch in batched(list(texts), min(self.config.batch_size, 96)):
            payload = {
                "model": self.config.model or self.default_model,
                "texts": list(batch),
                "input_type": self.config.extra.get("input_type", "classification"),
                "embedding_types": ["float"],
            }
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"]["float"])
        return np.asarray(vectors)


class JinaEncoder(Encoder):
    """Jina AI embeddings endpoint."""

    name = "jina"
    default_model = "jina-embeddings-v3"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        import requests

        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        vectors: list[list[float]] = []
        for batch in batched(list(texts), min(self.config.batch_size, 100)):
            payload = {
                "model": self.config.model or self.default_model,
                "task": self.config.extra.get("task", "text-matching"),
                "dimensions": self.config.dimensions or 1024,
                "embedding_type": "float",
                "input": [{"text": t} for t in batch],
            }
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(f"Jina API error: {body['error']}")
            data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
            vectors.extend(item["embedding"] for item in data)
        return np.asarray(vectors)


class NomicEncoder(Encoder):
    """Nomic Atlas embeddings endpoint (model of the source publications)."""

    name = "nomic"
    default_model = "nomic-embed-text-v1.5"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        import requests

        url = "https://api-atlas.nomic.ai/v1/embedding/text"
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        vectors: list[list[float]] = []
        for batch in batched(list(texts), min(self.config.batch_size, 100)):
            payload: dict[str, object] = {
                "texts": list(batch),
                "model": self.config.model or self.default_model,
                "task_type": self.config.extra.get("task_type", "search_document"),
                "long_text_mode": self.config.extra.get("long_text_mode", "truncate"),
            }
            if self.config.dimensions:
                payload["dimensionality"] = self.config.dimensions
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        return np.asarray(vectors)


class PrecomputedEncoder(Encoder):
    """Look vectors up in a mapping from text to embedding.

    Pass the mapping as ``EncoderConfig.extra['vectors']``.  Unknown texts
    raise :class:`KeyError` rather than silently returning a zero vector.
    """

    name = "precomputed"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: Mapping[str, Sequence[float]] | None = self.config.extra.get("vectors")
        if not vectors:
            raise ValueError("PrecomputedEncoder needs extra['vectors']")
        missing = [t for t in texts if t not in vectors]
        if missing:
            raise KeyError(
                f"{len(missing)} text(s) have no precomputed embedding, "
                f"first: {missing[0]!r}"
            )
        return np.asarray([list(vectors[t]) for t in texts])


_REGISTRY: dict[str, Callable[[EncoderConfig], Encoder]] = {
    TfidfSvdEncoder.name: TfidfSvdEncoder,
    SentenceTransformerEncoder.name: SentenceTransformerEncoder,
    OpenAIEncoder.name: OpenAIEncoder,
    CohereEncoder.name: CohereEncoder,
    JinaEncoder.name: JinaEncoder,
    NomicEncoder.name: NomicEncoder,
    PrecomputedEncoder.name: PrecomputedEncoder,
}


def register_encoder(name: str, factory: Callable[[EncoderConfig], Encoder]) -> None:
    """Register a custom encoder factory under ``name``."""
    _REGISTRY[name] = factory


def available_encoders() -> list[str]:
    return sorted(_REGISTRY)


def get_encoder(config: EncoderConfig | str) -> Encoder:
    """Instantiate the encoder named by ``config``."""
    if isinstance(config, str):
        config = EncoderConfig(name=config)
    try:
        factory = _REGISTRY[config.name]
    except KeyError as exc:
        raise ValueError(
            f"unknown encoder {config.name!r}; available: {available_encoders()}"
        ) from exc
    return factory(config)
