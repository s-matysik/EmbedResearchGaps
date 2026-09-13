"""The keyword semantic space: embeddings, frequencies, centrality, sub-clusters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .clustering import ClusterAssignment, distances_to_centroid, fit_kmeans
from .config import ClusteringConfig
from .encoders import Encoder
from .text import normalize_keyword

__all__ = ["KeywordSpace", "build_keyword_space", "default_subcluster_count"]


def default_subcluster_count(n_keywords: int, cap: int = 5) -> int:
    """Heuristic number of keyword sub-clusters: ``min(cap, max(2, sqrt(n)))``."""
    if n_keywords < 2:
        return 1
    return int(min(cap, max(2, int(math.sqrt(n_keywords)))))


@dataclass
class KeywordSpace:
    """Author keywords embedded in a semantic space.

    Attributes
    ----------
    keywords:
        Normalised keywords, unique and in order of first appearance.
    embeddings:
        Row-aligned embedding matrix.
    frequency:
        Document frequency of each keyword inside the scope the space was
        built for (a whole corpus, or a single article cluster).
    centrality:
        Distance of each keyword from the centroid of the whole keyword
        space -- the global-centrality component of equation (7).
    assignment:
        Keyword sub-clustering, when one was fitted.
    """

    keywords: list[str]
    embeddings: np.ndarray
    frequency: np.ndarray
    centrality: np.ndarray
    assignment: ClusterAssignment | None = None
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {kw: i for i, kw in enumerate(self.keywords)}
        n = len(self.keywords)
        for name, array in (
            ("embeddings", self.embeddings),
            ("frequency", self.frequency),
            ("centrality", self.centrality),
        ):
            if len(array) != n:
                raise ValueError(
                    f"{name} has {len(array)} rows but there are {n} keywords"
                )

    def __len__(self) -> int:
        return len(self.keywords)

    def position(self, keyword: str) -> int | None:
        return self.index.get(normalize_keyword(keyword))

    def embedding(self, keyword: str) -> np.ndarray | None:
        position = self.position(keyword)
        return None if position is None else self.embeddings[position]

    @property
    def subcluster_labels(self) -> np.ndarray:
        if self.assignment is None:
            return np.zeros(len(self.keywords), dtype=int)
        return self.assignment.labels

    @property
    def subcluster_distances(self) -> np.ndarray:
        """Distance of each keyword from its own sub-cluster centroid."""
        if self.assignment is None:
            return self.centrality.copy()
        return self.assignment.distances

    def fit_subclusters(
        self,
        config: ClusteringConfig,
        n_clusters: int | None = None,
    ) -> "KeywordSpace":
        """Sub-cluster the keyword space in place and return ``self``.

        Spaces with fewer than four keywords are left unclustered: k-means on
        such a handful produces centroids that sit on the points themselves,
        which would make every within-cluster distance zero.
        """
        if len(self.keywords) < 4:
            self.assignment = None
            return self
        k = n_clusters if n_clusters is not None else (
            config.n_keyword_subclusters or default_subcluster_count(len(self.keywords))
        )
        self.assignment = fit_kmeans(self.embeddings, config, n_clusters=k)
        return self

    def to_frame(self) -> pd.DataFrame:
        """Tabular view of the keyword space, one row per keyword."""
        return pd.DataFrame(
            {
                "keyword": self.keywords,
                "frequency": self.frequency.astype(int),
                "n_tokens": [len(kw.split()) for kw in self.keywords],
                "subcluster": self.subcluster_labels,
                "distance_from_subcentroid": self.subcluster_distances,
                "centrality": self.centrality,
            }
        )


def build_keyword_space(
    keyword_lists: Sequence[Sequence[str]],
    encoder: Encoder,
    metric: str = "cosine",
    cache: Mapping[str, Sequence[float]] | None = None,
) -> KeywordSpace:
    """Embed the unique author keywords of ``keyword_lists``.

    Parameters
    ----------
    keyword_lists:
        One sequence of keywords per record; keywords are normalised and
        de-duplicated, and their document frequency is counted.
    encoder:
        Embedding backend used for keywords.
    metric:
        Distance used for the global-centrality component.
    cache:
        Optional mapping of already-embedded keywords, which lets both modes
        and several article clusters share one round of embedding calls.
    """
    ordered: dict[str, int] = {}
    for keywords in keyword_lists:
        # dict.fromkeys, not a set: set iteration order depends on the
        # interpreter's hash seed, which would make the row order of the
        # embedding matrix -- and therefore the k-means++ initialisation --
        # differ between processes started with different seeds.
        for keyword in dict.fromkeys(normalize_keyword(kw) for kw in keywords):
            if keyword:
                ordered[keyword] = ordered.get(keyword, 0) + 1
    if not ordered:
        raise ValueError("no author keywords found in the supplied records")

    keywords = list(ordered)
    frequency = np.asarray([ordered[kw] for kw in keywords], dtype=int)

    if cache is not None and all(kw in cache for kw in keywords):
        embeddings = np.asarray([list(cache[kw]) for kw in keywords], dtype=np.float64)
    else:
        embeddings = encoder.encode(keywords)

    global_centroid = embeddings.mean(axis=0)
    centrality = distances_to_centroid(embeddings, global_centroid, metric)  # type: ignore[arg-type]
    return KeywordSpace(
        keywords=keywords,
        embeddings=embeddings,
        frequency=frequency,
        centrality=centrality,
    )
